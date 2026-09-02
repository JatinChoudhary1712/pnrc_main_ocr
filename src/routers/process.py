import asyncio
import atexit
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import MAX_FILES_PER_BATCH, OCR_CONCURRENCY, OUTPUT_DIR
from src.services.classification_service import classify_pdf, iter_classified_pages
from src.services.ocr_service import _ocr_one, empty_page_result, resolve_params

# In-process job store: job_id -> job dict. Single process, so a plain dict is
# enough. ponytail: grows unbounded; add TTL eviction if this stays up under load.
_jobs: dict[str, dict] = {}

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


def _save_json(job_id, pdf_name, payload):
    """Persist a job's output as soon as it's ready. Best-effort."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{Path(pdf_name).stem}_{job_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


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

        job["result"] = {
            "pdf_name": job["pdf_name"],
            "pages": ordered,
            "metrics": {
                "sheet_count": len(ordered),
                "filled_sheet_count": job["_filled_total"],
                "error_page_count": sum(1 for p in ordered if p["status"] == "error"),
                "processing_seconds": round(time.time() - job["_start"], 2),
            },
        }
        job["stage"] = "done"
        job["status"] = "done"
        _save_json(job["job_id"], job["pdf_name"], job["result"])


def _on_ocr_done(job, fut):
    try:
        job["_results"].append(fut.result())
    except Exception as exc:  # _ocr_one already swallows page errors; belt-and-braces
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
    while True:
        job_id = _classify_q.get()
        job = _jobs[job_id]
        try:
            job["status"] = "running"
            job["stage"] = "classifying"
            for page_number, result, pix in iter_classified_pages(job["_pdf_bytes"]):
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
                fut = _ocr_pool.submit(_ocr_one, item, job["_params"])
                fut.add_done_callback(lambda f, j=job: _on_ocr_done(j, f))
            job["_pdf_bytes"] = None  # free ~MBs per job once rendered
            job["_classified"] = True
            job["stage"] = "ocr"
            _finalize(job)  # covers the all-empty / zero-filled case
        except Exception as exc:
            job["status"] = "error"
            job["stage"] = "error"
            job["error"] = str(exc)
            _save_json(job_id, job["pdf_name"],
                       {"job_id": job_id, "status": "error", "error": str(exc)})
        finally:
            _classify_q.task_done()


threading.Thread(target=_classify_worker, name="classify", daemon=True).start()
atexit.register(_ocr_pool.shutdown, wait=False)


async def _read_pdf(file: UploadFile) -> tuple[str, bytes]:
    pdf_name = file.filename or "uploaded.pdf"
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail=f"{pdf_name} is empty.")
    return pdf_name, pdf_bytes


def build_process_router():
    router = APIRouter()

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
        cap = max_files or MAX_FILES_PER_BATCH
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF is required.")
        if len(files) > cap:
            raise HTTPException(
                status_code=413,
                detail=f"Too many PDFs in one request ({len(files)}). Submit at most {cap}.",
            )

        params = resolve_params(
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
            _jobs[job_id] = {
                "job_id": job_id,
                "pdf_name": pdf_name,
                "status": "queued",
                "stage": "queued",
                "pages_done": 0,
                "pages_total": 0,
                "result": None,
                "error": None,
                "_pdf_bytes": pdf_bytes,
                "_params": params,
                "_start": time.time(),
                "_filled_total": 0,
                "_results": [],
                "_empty_pages": [],
                "_classified": False,
                "classified": [],  # public: [{page, prediction}] as pages are scanned
            }
            _classify_q.put(job_id)
            jobs.append({"pdf_name": pdf_name, "job_id": job_id, "status": "queued"})

        return JSONResponse(status_code=202, content={"jobs": jobs})

    @router.get("/process/{job_id}")
    async def process_status(job_id: str):
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}.")
        return _public(job)

    @router.post("/classify")
    async def classify(file: UploadFile = File(...)):
        """DINOv2 filled/empty classification only — no OCR."""
        pdf_name, pdf_bytes = await _read_pdf(file)
        page_results, _ = await asyncio.to_thread(classify_pdf, pdf_bytes)
        return {"pdf_name": pdf_name, "pages": page_results}

    return router
