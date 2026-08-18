from fastapi import APIRouter, File, HTTPException, UploadFile

from src.config import AVG_SECONDS_PER_PAGE, L4_HOURLY_RATE


def build_cost_router(worker_cls):
    router = APIRouter()

    @router.post("/cost-check")
    async def cost_check(files: list[UploadFile] = File(...)):
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF is required.")

        worker = worker_cls()
        documents = []
        total_filled = 0

        for file in files:
            pdf_name = file.filename or "uploaded.pdf"
            pdf_bytes = await file.read()
            if not pdf_bytes:
                documents.append({"pdf_name": pdf_name, "error": "empty file"})
                continue

            result = await worker.classify_only.remote.aio(pdf_bytes, pdf_name)
            total_filled += result["filled_pages"]
            documents.append(result)

        estimated_seconds = total_filled * AVG_SECONDS_PER_PAGE
        estimated_cost = (estimated_seconds / 3600) * L4_HOURLY_RATE

        return {
            "documents": documents,
            "summary": {
                "total_filled_pages": total_filled,
                "estimated_ocr_seconds": round(estimated_seconds, 2),
                "estimated_cost_usd": round(estimated_cost, 4),
            },
        }

    return router
