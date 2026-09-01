import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import MAX_FILES_PER_BATCH, OUTPUT_DIR
from src.services.classification_service import classify_pdf
from src.services.ocr_service import empty_page_result, ocr_image_payloads

# In-process job store: job_id -> job dict. Single process now, so a plain dict
# is enough. ponytail: grows unbounded; add TTL eviction if this stays up under
# heavy traffic.
_jobs: dict[str, dict] = {}

# Keep strong refs to running tasks so the event loop doesn't GC (and cancel) them.
_tasks: set = set()


def _save_json(job_id, pdf_name, payload):
    """Persist a job's output as soon as it's ready. Best-effort: a write failure
    must not fail the job."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{Path(pdf_name).stem}_{job_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _merge_pages(page_results, ocr_results):
    pages = {
        r["page"]: empty_page_result(r["page"])
        for r in page_results
        if r["prediction"] == "empty"
    }
    for item in ocr_results:
        pages[item["page"]] = {
            "page": item["page"],
            "status": item["status"],
            "text": item["text"],
            "confidence": item["confidence"],
            "need_review": item["need_review"],
        }
    return [pages[p] for p in sorted(pages)]


def _process_sync(job, pdf_bytes):
    """Blocking classify + OCR. Runs in a worker thread; mutates `job` for progress."""
    start = time.time()

    job["status"] = "running"
    job["stage"] = "classifying"
    page_results, page_images = classify_pdf(pdf_bytes)

    filled = [r["page"] for r in page_results if r["prediction"] == "filled"]
    payload = [
        {"page": p, "image_png": page_images[p].tobytes("png"), "pdf_name": job["pdf_name"]}
        for p in filled
    ]

    job["stage"] = "ocr"
    job["pages_total"] = len(payload)

    def on_progress(done, total):
        job["pages_done"] = done
        job["pages_total"] = total

    ocr_results = ocr_image_payloads(payload, on_progress=on_progress)
    pages = _merge_pages(page_results, ocr_results)

    job["result"] = {
        "pdf_name": job["pdf_name"],
        "pages": pages,
        "metrics": {
            "sheet_count": len(page_results),
            "filled_sheet_count": len(filled),
            "error_page_count": sum(1 for p in pages if p["status"] == "error"),
            "processing_seconds": round(time.time() - start, 2),
        },
    }
    job["stage"] = "done"
    job["status"] = "done"
    _save_json(job["job_id"], job["pdf_name"], job["result"])


async def _run_job(job_id, pdf_bytes):
    job = _jobs[job_id]
    try:
        await asyncio.to_thread(_process_sync, job, pdf_bytes)
    except Exception as exc:
        job["status"] = "error"
        job["stage"] = "error"
        job["error"] = str(exc)
        _save_json(job_id, job["pdf_name"], {"job_id": job_id, "status": "error", "error": str(exc)})


async def _read_pdf(file: UploadFile) -> tuple[str, bytes]:
    pdf_name = file.filename or "uploaded.pdf"
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail=f"{pdf_name} is empty.")
    return pdf_name, pdf_bytes


def build_process_router():
    router = APIRouter()

    @router.post("/process")
    async def process(files: list[UploadFile] = File(...)):
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF is required.")
        if len(files) > MAX_FILES_PER_BATCH:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Too many PDFs in one request ({len(files)}). "
                    f"Submit at most {MAX_FILES_PER_BATCH} per request."
                ),
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
            }
            task = asyncio.create_task(_run_job(job_id, pdf_bytes))
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
            jobs.append({"pdf_name": pdf_name, "job_id": job_id, "status": "queued"})

        return JSONResponse(status_code=202, content={"jobs": jobs})

    @router.get("/process/{job_id}")
    async def process_status(job_id: str):
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}.")
        return job

    @router.post("/classify")
    async def classify(file: UploadFile = File(...)):
        """DINOv2 filled/empty classification only — no OCR."""
        pdf_name, pdf_bytes = await _read_pdf(file)
        page_results, _ = await asyncio.to_thread(classify_pdf, pdf_bytes)
        return {"pdf_name": pdf_name, "pages": page_results}

    return router
