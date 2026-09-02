"""
Focused tests for the /upload submission path in src/routers/process.py.

Stubs DINOv2 + PyMuPDF + vLLM so it runs with no GPU and no network:

    python test_upload.py
"""
import io
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

# --- point the app at throwaway dirs + a known key BEFORE importing it -------
TMP = Path(tempfile.mkdtemp(prefix="pnrc_test_"))
os.environ["API_KEY"] = "testkey"
os.environ["UPLOAD_TMP_DIR"] = str(TMP / "uploads")
os.environ["OUTPUT_DIR"] = str(TMP / "output")
UP = TMP / "uploads"

# --- stub the heavy deps before importing the router -----------------------
_emb = types.ModuleType("src.services.embedding_service")
_emb.get_embedding = lambda image: None
_emb.DEVICE = "cpu"
sys.modules["src.services.embedding_service"] = _emb

_cls = types.ModuleType("src.services.classification_service")
_cls.classify_pdf = lambda pdf: ([], {})
_cls.iter_classified_pages = lambda pdf: iter(())   # overridden per-test below
sys.modules["src.services.classification_service"] = _cls

from fastapi.testclient import TestClient  # noqa: E402
import src.routers.process as P  # noqa: E402
from src.main import app  # noqa: E402

client = TestClient(app)
H = {"X-API-Key": "testkey"}
FAKE_PDF = b"%PDF-1.4 not really a pdf but non-empty"


class _Pix:
    def tobytes(self, fmt):
        return b"png-bytes"


def _pages_ok(pdf):
    # /upload hands the worker the streamed-to-disk path; /process hands it bytes.
    if isinstance(pdf, str):
        assert Path(pdf).is_file(), f"expected live temp path, got {pdf!r}"
    else:
        assert isinstance(pdf, (bytes, bytearray)) and pdf, f"expected bytes, got {pdf!r}"
    return iter([(1, {"prediction": "filled"}, _Pix()),
                 (2, {"prediction": "empty"}, _Pix())])


def _pages_raise(pdf):
    raise RuntimeError("corrupt pdf: cannot open")


def _ocr_one_stub(item, params):
    return {"page": item["page"], "status": "filled", "text": f"t{item['page']}",
            "confidence": {"is_low_confidence": False}, "need_review": False}


P._ocr_one = _ocr_one_stub


def _wait(job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/process/{job_id}", headers=H)
        if r.status_code == 200 and r.json().get("status") in ("done", "error"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def _no_temp_pdfs():
    return not any(UP.glob("*.pdf"))


# --------------------------------------------------------------------------- #
def test_healthz_needs_no_key():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_creates_job_streams_to_disk_and_cleans_up():
    P.iter_classified_pages = _pages_ok
    r = client.post("/upload", headers={**H, "Idempotency-Key": "k1"},
                    files={"file": ("doc one.pdf", io.BytesIO(FAKE_PDF), "application/pdf")})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["job_id"] and body["status"] == "queued"
    assert body["filename"] == "doc_one.pdf"       # sanitised
    got = _wait(body["job_id"])
    assert got["status"] == "done"
    assert [p["page"] for p in got["result"]["pages"]] == [1, 2]
    assert got["result"]["pages"][0]["text"] == "t1"
    assert got["result"]["pages"][1]["status"] == "empty"
    assert _no_temp_pdfs(), "temp PDF not cleaned after finalize"


def test_idempotent_retry_returns_same_job_no_duplicate():
    P.iter_classified_pages = _pages_ok
    before = len(P._jobs)
    r1 = client.post("/upload", headers={**H, "Idempotency-Key": "dup"},
                     files={"file": ("d.pdf", io.BytesIO(FAKE_PDF), "application/pdf")})
    r2 = client.post("/upload", headers={**H, "Idempotency-Key": "dup"},
                     files={"file": ("d.pdf", io.BytesIO(FAKE_PDF), "application/pdf")})
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r2.json().get("duplicate") is True
    assert len(P._jobs) == before + 1, "duplicate upload created a second job"
    _wait(r1.json()["job_id"])


def test_empty_file_rejected_and_cleaned():
    r = client.post("/upload", headers=H,
                    files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")})
    assert r.status_code == 422
    assert _no_temp_pdfs()


def test_malformed_pdf_marks_error_and_cleans():
    P.iter_classified_pages = _pages_raise
    r = client.post("/upload", headers={**H, "Idempotency-Key": "bad"},
                    files={"file": ("bad.pdf", io.BytesIO(b"junk"), "application/pdf")})
    got = _wait(r.json()["job_id"])
    assert got["status"] == "error"
    assert "corrupt" in (got["error"] or "")
    assert _no_temp_pdfs(), "temp PDF not cleaned after error"


def test_upload_requires_key():
    r = client.post("/upload",
                    files={"file": ("x.pdf", io.BytesIO(FAKE_PDF), "application/pdf")})
    assert r.status_code == 401


def test_process_status_disk_fallback_after_restart():
    out = Path(os.environ["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "thing_deadbeef.json").write_text(
        json.dumps({"pdf_name": "thing.pdf", "pages": [], "metrics": {}}), encoding="utf-8")
    r = client.get("/process/deadbeef", headers=H)
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    assert r.json()["result"]["pdf_name"] == "thing.pdf"


def test_metrics_shape():
    m = client.get("/metrics", headers=H).json()
    for k in ("jobs_total", "queued", "running", "done", "error",
              "classify_queue_depth", "ocr_active", "ocr_pool_max", "idempotency_keys"):
        assert k in m, k


def test_legacy_process_multi_file_still_works():
    P.iter_classified_pages = _pages_ok
    r = client.post("/process", headers=H, files=[
        ("files", ("a.pdf", io.BytesIO(FAKE_PDF), "application/pdf")),
        ("files", ("b.pdf", io.BytesIO(FAKE_PDF), "application/pdf")),
    ])
    assert r.status_code == 202
    jobs = r.json()["jobs"]
    assert len(jobs) == 2 and all(j.get("job_id") for j in jobs)
    for j in jobs:
        assert _wait(j["job_id"])["status"] == "done"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as exc:
            failed += 1
            print("FAIL", fn.__name__, "->", repr(exc))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
