"""Self-check for the streaming classify -> OCR pipeline in src/routers/process.py.

Stubs out DINOv2 + vLLM so it runs with no GPU and no network:

    python test_pipeline.py
"""
import sys
import time
import types

# --- stub the heavy deps before importing the router ------------------------
_emb = types.ModuleType("src.services.embedding_service")
_emb.get_embedding = lambda image: None
_emb.DEVICE = "cpu"
sys.modules["src.services.embedding_service"] = _emb

_cls = types.ModuleType("src.services.classification_service")
_cls.classify_pdf = lambda pdf_bytes: ([], {})
_cls.iter_classified_pages = lambda pdf_bytes: iter(())  # overridden below
sys.modules["src.services.classification_service"] = _cls

import src.routers.process as P  # noqa: E402

# 3 pages: filled, empty, filled. Page 3 will "fail" OCR.
FAKE_PAGES = [
    (1, {"prediction": "filled"}, types.SimpleNamespace(tobytes=lambda fmt: b"png1")),
    (2, {"prediction": "empty"}, types.SimpleNamespace(tobytes=lambda fmt: b"png2")),
    (3, {"prediction": "filled"}, types.SimpleNamespace(tobytes=lambda fmt: b"png3")),
]
P.iter_classified_pages = lambda pdf_bytes: iter(FAKE_PAGES)


def fake_ocr_one(item, params):
    time.sleep(0.05)  # let classification get ahead -> exercises streaming path
    if item["page"] == 3:
        return {"page": 3, "status": "error", "text": None,
                "confidence": {"is_low_confidence": True}, "need_review": True, "error": "boom"}
    return {"page": item["page"], "status": "filled", "text": f"text {item['page']}",
            "confidence": {"is_low_confidence": False}, "need_review": False}


P._ocr_one = fake_ocr_one


def main():
    job_id = "test-job"
    P._jobs[job_id] = {
        "job_id": job_id, "pdf_name": "doc.pdf",
        "status": "queued", "stage": "queued",
        "pages_done": 0, "pages_total": 0, "result": None, "error": None,
        "_pdf_bytes": b"x", "_params": P.resolve_params(), "_start": time.time(),
        "_filled_total": 0, "_results": [], "_empty_pages": [], "_classified": False,
        "classified": [],
    }
    P._classify_q.put(job_id)

    deadline = time.time() + 10
    while P._jobs[job_id]["status"] not in ("done", "error"):
        assert time.time() < deadline, "pipeline never finished"
        time.sleep(0.05)

    job = P._jobs[job_id]
    assert job["status"] == "done", job
    pages = job["result"]["pages"]
    assert [p["page"] for p in pages] == [1, 2, 3], pages
    assert pages[0]["text"] == "text 1"
    assert pages[1]["status"] == "empty"          # page 2 merged in from _empty_pages
    assert pages[2]["status"] == "error"          # page 3 OCR failure survives
    m = job["result"]["metrics"]
    assert m == {"sheet_count": 3, "filled_sheet_count": 2, "error_page_count": 1,
                 **{k: m[k] for k in ("processing_seconds",)}}, m
    print("OK", m)


if __name__ == "__main__":
    main()
