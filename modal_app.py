import os
import time
import modal

# Read directly rather than importing src.config: this module is re-imported
# inside the container before src/ is added to sys.path (see _start_ollama),
# so a top-level `from src...` import here would crash-loop every container.
SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN_WINDOW", "15"))

app = modal.App("pnrc-main-ocr")

progress_dict = modal.Dict.from_name("pnrc-ocr-progress", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "httpx", "ollama", "python-dotenv", "tqdm", "fastapi", "python-multipart",
        "pymupdf", "pydantic", "numpy", "pillow", "scikit-learn",
        "torch", "torchvision", "transformers",
    )
    .run_commands(
        "nohup ollama serve > /tmp/ollama.log 2>&1 & "
        "sleep 5 && ollama pull glm-ocr"
    )
    .run_commands(
        "python -c \"from transformers import AutoImageProcessor, AutoModel; "
        "AutoImageProcessor.from_pretrained('facebook/dinov2-base'); "
        "AutoModel.from_pretrained('facebook/dinov2-base')\""
    )
    .add_local_dir("src", remote_path="/root/app/src")
    .add_local_dir("knowledge_base", remote_path="/root/app/knowledge_base")
)


def _start_ollama():
    import os
    import subprocess
    import sys
    import httpx

    env = {**os.environ, "OLLAMA_NO_CLOUD": "1"}
    subprocess.Popen(["ollama", "serve"], env=env)

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            httpx.get("http://127.0.0.1:11434/api/version", timeout=2).raise_for_status()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Ollama did not become ready within 60s")

    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")


def _empty_page_result(page_number):
    return {
        "page": page_number,
        "status": "empty",
        "text": None,
        "confidence": {"is_low_confidence": False, "has_low_confidence_run": False},
        "need_review": False,
    }


def _merge_pages(page_results, ocr_filled):
    pages = {}
    for result in page_results:
        if result["prediction"] == "empty":
            pages[result["page"]] = _empty_page_result(result["page"])

    for item in ocr_filled:
        pages[item["page"]] = {
            "page": item["page"],
            "status": item["status"],
            "text": item["text"],
            "confidence": item["confidence"],
            "need_review": item["need_review"],
        }

    return [pages[p] for p in sorted(pages)]


@app.cls(
    image=image,
    gpu="L4",
    timeout=3600,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Worker:
    @modal.enter()
    def start(self):
        self.container_start_ts = time.time()
        _start_ollama()
        self.ollama_ready_ts = time.time()

        from src.services.classification_service import classify_pdf
        self.classify_pdf = classify_pdf

        self.request_count = 0

    @modal.method()
    def process_document(self, pdf_bytes: bytes, pdf_name: str, job_id: str):
        from src.config import L4_HOURLY_RATE
        from src.services.ocr_service import ocr_image_payloads

        current_stage = "classifying"

        def set_progress(stage, **detail):
            nonlocal current_stage
            current_stage = stage
            progress_dict[job_id] = {"stage": stage, "pdf_name": pdf_name, **detail}

        try:
            is_first_request = self.request_count == 0
            self.request_count += 1

            start_time = time.time()

            set_progress("classifying")
            page_results, page_images = self.classify_pdf(pdf_bytes)

            filled_pages = [r["page"] for r in page_results if r["prediction"] == "filled"]
            pages_payload = [
                {"page": p, "image_png": page_images[p].tobytes("png"), "pdf_name": pdf_name}
                for p in filled_pages
            ]

            set_progress("ocr", pages_done=0, pages_total=len(pages_payload))
            ocr_results = ocr_image_payloads(
                pages_payload,
                on_progress=lambda done, total: set_progress("ocr", pages_done=done, pages_total=total),
            )
            pages = _merge_pages(page_results, ocr_results)

            processing_seconds = time.time() - start_time
            cold_start_seconds = (self.ollama_ready_ts - self.container_start_ts) if is_first_request else 0

            result = {
                "pages": pages,
                "metrics": {
                    "sheet_count": len(page_results),
                    "filled_sheet_count": len(filled_pages),
                    "processing_seconds": round(processing_seconds, 2),
                    "is_first_request": is_first_request,
                    "cold_start_seconds": round(cold_start_seconds, 2),
                    "approx_cost_usd": round(((processing_seconds + cold_start_seconds) / 3600) * L4_HOURLY_RATE, 4),
                },
            }
            set_progress("done")
            return result
        except Exception as exc:
            progress_dict[job_id] = {
                "stage": "error",
                "pdf_name": pdf_name,
                "error": str(exc),
                "failed_stage": current_stage,
            }
            raise


web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "python-multipart", "pydantic", "python-dotenv")
    .add_local_dir("src", remote_path="/root/app/src")
)

@app.function(image=web_image, secrets=[modal.Secret.from_name("pnrc-api-key")])
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root/app")

    from src.main import create_app

    return create_app(Worker)
