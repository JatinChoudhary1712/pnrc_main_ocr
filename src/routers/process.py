import asyncio
import atexit
import json
import logging
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from src.config import MAX_FILES_PER_BATCH, OCR_CONCURRENCY, OUTPUT_DIR, UPLOAD_TMP_DIR
from src.services.classification_service import classify_pdf, iter_classified_pages
from src.services.ocr_service import _ocr_one, empty_page_result, resolve_params

logger = logging.getLogger("pnrc")

# In-process job store: job_id -> job dict. Single process, so a plain dict is
# enough. ponytail: grows unbounded; add TTL eviction if this stays up under load.
_jobs: dict[str, dict] = {}

# idempotency-key -> job_id. Lets a client retry a timed-out upload without
# spawning a second (expensive) OCR job. Grows with _jobs; same trade-off.
_idem: dict[str, str] = {}

# Pipeline: one classify worker feeds a shared OCR pool.
#   * ONE classify thread — DINOv2 is a single model instance; serializing it
#     avoids GPU/GIL contention no matter how many PDFs are queued.
#   * ONE shared OCR pool sized to OCR_CONCURRENCY — vLLM sees a steady bounded
#     stream instead of (n_pdfs * concurrency) simultaneous requests.
# A filled page is handed to the OCR pool the instant it's classified; each
# job's JSON is written as soon as its last page comes back.
_classify_q: "queue.Queue[str]" = queue.Queue()
_ocr_pool = ThreadPoolExecutor(max_workers=OCR_CONCURRENCY, thread_name_prefix="ocr")
_finalize_lock = threading.Lock()  # ponytail: one global lock; per-job if it gets hot

_ocr_active = 0                     # in-flight OCR requests, for /metrics
_ocr_active_lock = threading.Lock()

_UPLOAD_CHUNK = 1 << 20            # 1 MiB streaming chunk


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _safe_name(name: str) -> str:
    """Strip any path and reduce to a filesystem-safe basename."""
    base = str(name or "uploaded.pdf").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._") or "uploaded.pdf"
    return base[:200]


def _new_job(job_id, pdf_name, params, *, pdf_bytes=None, pdf_path=None, idempotency_key=None):
    return {
        "job_id": job_id,
        "pdf_name": pdf_name,
        "status": "queued",
        "stage": "queued",
        "pages_done": 0,
        "pages_total": 0,
        "result": None,
        "error": None,
        "idempotency_key": idempotency_key,
        "_pdf_bytes": pdf_bytes,
        "_pdf_path": pdf_path,
        "_params": params,
        "_start": time.time(),
        "_filled_total": 0,
        "_results": [],
        "_empty_pages": [],
        "_classified": False,
        "classified": [],  # public: [{page, prediction}] as pages are scanned
    }


def _cleanup_upload(job):
    """Delete the streamed-to-disk PDF once the job can't need it any more.
    Idempotent."""
    path = job.get("_pdf_path")
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
        logger.info("cleanup job=%s removed %s", job.get("job_id"), path)
    except Exception:
        logger.warning("cleanup job=%s failed to remove %s", job.get("job_id"), path)
    job["_pdf_path"] = None


def _save_json(job_id, pdf_name, payload):
    """Persist a job's output as soon as it's ready. Best-effort."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{Path(pdf_name).stem}_{job_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("save_json job=%s failed", job_id)


def _public(job):
    """Job dict minus the private _ pipeline fields (bytes, futures, counters)."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _finalize(job):
    """Write JSON + mark done once classification finished AND every filled page
    has an OCR result. Idempotent; safe to call after each page."""
    with _finalize_lock:
        if job["status"] in ("done", "error"):
            return
        if not job["_classified"] or len(job["_results"]) < job["_filled_total"]:
            return

        pages = {n: empty_page_result(n) for n in job["_empty_pages"]}
        for r in job["_results"]:
            pages[r["page"]] = {
                k: r[k] for k in ("page", "status", "text", "confidence", "need_review")
            }
        ordered = [pages[p] for p in sorted(pages)]

        err_pages = sum(1 for p in ordered if p["status"] == "error")
        job["result"] = {
            "pdf_name": job["pdf_name"],
            "pages": ordered,
            "metrics": {
                "sheet_count": len(ordered),
                "filled_sheet_count": job["_filled_total"],
                "error_page_count": err_pages,
                "processing_seconds": round(time.time() - job["_start"], 2),
            },
        }
        job["stage"] = "done"
        job["status"] = "done"
        _save_json(job["job_id"], job["pdf_name"], job["result"])
        _cleanup_upload(job)
        logger.info(
            "finalize job=%s status=done pages=%d filled=%d errpages=%d dur=%.1fs",
            job["job_id"], len(ordered), job["_filled_total"], err_pages,
            job["result"]["metrics"]["processing_seconds"],
        )


def _on_ocr_done(job, fut):
    global _ocr_active
    with _ocr_active_lock:
        _ocr_active -= 1
    try:
        job["_results"].append(fut.result())
    except Exception as exc:  # _ocr_one already swallows page errors; belt-and-braces
        logger.warning("ocr callback job=%s raised: %s", job["job_id"], exc)
        job["_results"].append(
            {"page": -1, "status": "error", "text": None,
             "confidence": empty_page_result(-1)["confidence"], "need_review": True,
             "error": str(exc)}
        )
    job["pages_done"] = len(job["_results"])
    _finalize(job)


def _classify_worker():
    """Pull PDFs off the queue, classify page by page, stream each filled page
    straight to the OCR pool. One thread, runs forever."""
    global _ocr_active
    while True:
        job_id = _classify_q.get()
        job = _jobs[job_id]
        try:
            job["status"] = "running"
            job["stage"] = "classifying"
            src = job.get("_pdf_path") or job["_pdf_bytes"]
            logger.info("classify start job=%s file=%s", job_id, job["pdf_name"])
            for page_number, result, pix in iter_classified_pages(src):
                job["classified"].append({"page": page_number, "prediction": result["prediction"]})
                if result["prediction"] == "empty":
                    job["_empty_pages"].append(page_number)
                    continue
                job["_filled_total"] += 1
                job["pages_total"] = job["_filled_total"]
                item = {
                    "page": page_number,
                    "image_png": pix.tobytes("png"),
                    "pdf_name": job["pdf_name"],
                }
                with _ocr_active_lock:
                    _ocr_active += 1
                fut = _ocr_pool.submit(_ocr_one, item, job["_params"])
                fut.add_done_callback(lambda f, j=job: _on_ocr_done(j, f))
            job["_pdf_bytes"] = None  # free ~MBs per job once rendered
            job["_classified"] = True
            job["stage"] = "ocr"
            logger.info(
                "classify done job=%s pages=%d filled=%d -> ocr",
                job_id, len(job["classified"]), job["_filled_total"],
            )
            _finalize(job)  # covers the all-empty / zero-filled case
        except Exception as exc:
            logger.exception("classify failed job=%s file=%s", job_id, job["pdf_name"])
            job["status"] = "error"
            job["stage"] = "error"
            job["error"] = str(exc)
            _save_json(job_id, job["pdf_name"],
                       {"job_id": job_id, "status": "error", "error": str(exc)})
            _cleanup_upload(job)
        finally:
            _classify_q.task_done()


threading.Thread(target=_classify_worker, name="classify", daemon=True).start()
atexit.register(_ocr_pool.shutdown, wait=False)


# --------------------------------------------------------------------------- #
#  request helpers
# --------------------------------------------------------------------------- #
async def _read_pdf(file: UploadFile) -> tuple[str, bytes]:
    pdf_name = file.filename or "uploaded.pdf"
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail=f"{pdf_name} is empty.")
    return pdf_name, pdf_bytes


async def _stream_to_disk(file: UploadFile, dest: Path) -> int:
    """Write an UploadFile to `dest` in bounded chunks. Returns bytes written.
    Never holds the whole file in memory."""
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            size += len(chunk)
    return size


def _params_from_form(temperature, top_p, top_k, max_tokens, reasoning, concurrency, page_retries):
    return resolve_params(
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "reasoning": reasoning,
            "concurrency": concurrency,
            "page_retries": page_retries,
        }
    )


# --------------------------------------------------------------------------- #
#  routes
# --------------------------------------------------------------------------- #
def build_process_router():
    router = APIRouter()

    @router.post("/upload")
    async def upload(
        file: UploadFile = File(...),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        temperature: float | None = Form(None),
        top_p: float | None = Form(None),
        top_k: int | None = Form(None),
        max_tokens: int | None = Form(None),
        reasoning: str | None = Form(None),
        concurrency: int | None = Form(None),
        page_retries: int | None = Form(None),
    ):
        """Accept ONE PDF, stream it to disk, create a job, return 202 immediately.

        This is the reliable submission path: each request carries a single small
        file, so it completes fast enough that no proxy/duration limit is hit, and
        an interrupted upload never takes a whole batch down with it.
        """
        fname = _safe_name(file.filename)
        t0 = time.time()

        # Retry of an upload that already produced a job → hand back the same job.
        if idempotency_key and idempotency_key in _idem:
            jid = _idem[idempotency_key]
            job = _jobs.get(jid)
            logger.info("upload dedup key=%s -> job=%s file=%s", idempotency_key, jid, fname)
            return JSONResponse(
                status_code=202,
                content={
                    "job_id": jid,
                    "filename": fname,
                    "status": (job["status"] if job else "unknown"),
                    "duplicate": True,
                },
            )

        job_id = str(uuid.uuid4())
        dest = UPLOAD_TMP_DIR / f"{job_id}.pdf"
        try:
            size = await _stream_to_disk(file, dest)
        except ClientDisconnect:
            dest.unlink(missing_ok=True)
            logger.warning("upload aborted (client disconnect) file=%s dur=%.1fs", fname, time.time() - t0)
            raise HTTPException(status_code=499, detail="client disconnected during upload")
        except Exception as exc:
            dest.unlink(missing_ok=True)
            logger.exception("upload stream failed file=%s", fname)
            raise HTTPException(status_code=500, detail=f"upload failed: {exc}")

        if size == 0:
            dest.unlink(missing_ok=True)
            logger.warning("upload empty file=%s", fname)
            raise HTTPException(status_code=422, detail=f"{fname} is empty.")

        params = _params_from_form(temperature, top_p, top_k, max_tokens, reasoning, concurrency, page_retries)
        job = _new_job(job_id, fname, params, pdf_path=str(dest), idempotency_key=idempotency_key)
        _jobs[job_id] = job
        if idempotency_key:
            _idem[idempotency_key] = job_id
        _classify_q.put(job_id)
        logger.info(
            "upload ok file=%s size=%dB dur=%.1fs job=%s key=%s path=%s",
            fname, size, time.time() - t0, job_id, idempotency_key, dest,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "filename": fname,
                "status": "queued",
                "idempotency_key": idempotency_key,
            },
        )

    @router.post("/process")
    async def process(
        files: list[UploadFile] = File(...),
        temperature: float | None = Form(None),
        top_p: float | None = Form(None),
        top_k: int | None = Form(None),
        max_tokens: int | None = Form(None),
        reasoning: str | None = Form(None),
        concurrency: int | None = Form(None),
        page_retries: int | None = Form(None),
        max_files: int | None = Form(None),
    ):
        """Legacy multi-file path. Kept for existing callers (smoke tests,
        backend_test.html). New clients should use /upload — one file per
        request — which does not depend on a long-lived multipart POST."""
        cap = max_files or MAX_FILES_PER_BATCH
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF is required.")
        if len(files) > cap:
            raise HTTPException(
                status_code=413,
                detail=f"Too many PDFs in one request ({len(files)}). Submit at most {cap}.",
            )

        params = _params_from_form(temperature, top_p, top_k, max_tokens, reasoning, concurrency, page_retries)

        jobs = []
        for file in files:
            try:
                pdf_name, pdf_bytes = await _read_pdf(file)
            except HTTPException as exc:
                jobs.append(
                    {"pdf_name": file.filename or "uploaded.pdf", "status": "error", "error": exc.detail}
                )
                continue

            job_id = str(uuid.uuid4())
            _jobs[job_id] = _new_job(job_id, pdf_name, params, pdf_bytes=pdf_bytes)
            _classify_q.put(job_id)
            logger.info("process queued job=%s file=%s size=%dB", job_id, pdf_name, len(pdf_bytes))
            jobs.append({"pdf_name": pdf_name, "job_id": job_id, "status": "queued"})

        return JSONResponse(status_code=202, content={"jobs": jobs})

    @router.get("/process/{job_id}")
    async def process_status(job_id: str):
        job = _jobs.get(job_id)
        if job is not None:
            return _public(job)
        # Fallback: job not in memory (e.g. pod restarted) but its result JSON
        # may still be on disk.
        try:
            hit = next(iter(OUTPUT_DIR.glob(f"*_{job_id}.json")), None)
            if hit:
                payload = json.loads(hit.read_text(encoding="utf-8"))
                status = "error" if payload.get("status") == "error" else "done"
                return {"job_id": job_id, "status": status, "stage": status,
                        "result": None if status == "error" else payload,
                        "error": payload.get("error")}
        except Exception:
            logger.warning("disk fallback failed for job=%s", job_id)
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}.")

    @router.get("/metrics")
    async def metrics():
        from collections import Counter

        c = Counter(j["status"] for j in _jobs.values())
        return {
            "jobs_total": len(_jobs),
            "queued": c.get("queued", 0),
            "running": c.get("running", 0),
            "done": c.get("done", 0),
            "error": c.get("error", 0),
            "classify_queue_depth": _classify_q.qsize(),
            "ocr_active": _ocr_active,
            "ocr_pool_max": OCR_CONCURRENCY,
            "idempotency_keys": len(_idem),
        }

    @router.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @router.post("/classify")
    async def classify(file: UploadFile = File(...)):
        """DINOv2 filled/empty classification only — no OCR."""
        pdf_name, pdf_bytes = await _read_pdf(file)
        page_results, _ = await asyncio.to_thread(classify_pdf, pdf_bytes)
        return {"pdf_name": pdf_name, "pages": page_results}

    return router
