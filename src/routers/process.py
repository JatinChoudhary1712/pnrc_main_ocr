from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import modal

from src.config import MAX_FILES_PER_BATCH


async def _read_pdf(file: UploadFile) -> tuple[str, bytes]:
    pdf_name = file.filename or "uploaded.pdf"
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail=f"{pdf_name} is empty.")
    return pdf_name, pdf_bytes


def build_process_router(worker_cls):
    router = APIRouter()

    @router.post("/process")
    async def process_document(file: UploadFile = File(...)):
        pdf_name, pdf_bytes = await _read_pdf(file)
        worker = worker_cls()
        result = await worker.process_document.remote.aio(pdf_bytes, pdf_name)
        return {"pdf_name": pdf_name, **result}

    @router.post("/process/batch")
    async def process_batch(files: list[UploadFile] = File(...)):
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF is required.")
        if len(files) > MAX_FILES_PER_BATCH:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Too many PDFs in one request ({len(files)}). "
                    f"Submit at most {MAX_FILES_PER_BATCH} PDFs per batch. "
                    "Send another request for the remaining files."
                ),
            )

        worker = worker_cls()
        jobs = []
        for file in files:
            try:
                pdf_name, pdf_bytes = await _read_pdf(file)
                call = await worker.process_document.spawn.aio(pdf_bytes, pdf_name)
                jobs.append({"pdf_name": pdf_name, "call_id": call.object_id, "status": "queued"})
            except HTTPException as exc:
                jobs.append({"pdf_name": file.filename or "uploaded.pdf", "status": "error", "error": exc.detail})

        queued = sum(job["status"] == "queued" for job in jobs)
        return JSONResponse(
            status_code=202,
            content={
                "jobs": jobs,
                "message": "Jobs queued. Poll /process/result/{call_id} for each.",
                "metrics": {
                    "files_received": len(files),
                    "files_queued": queued,
                    "files_failed_to_queue": len(files) - queued,
                },
            },
        )

    @router.get("/process/result/{call_id}")
    async def process_result(call_id: str):
        try:
            function_call = modal.FunctionCall.from_id(call_id)
            result = await function_call.get.aio(timeout=0)
        except TimeoutError:
            return JSONResponse(status_code=202, content={"call_id": call_id, "status": "running"})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not retrieve job {call_id}: {exc}") from exc

        return {"call_id": call_id, "status": "completed", **result}

    return router
