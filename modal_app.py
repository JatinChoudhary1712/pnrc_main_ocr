import time
import modal

app = modal.App("pnrc-main-ocr")

MAX_FILES_PER_BATCH = 8

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
    scaledown_window=80,
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
    def process_document(self, pdf_bytes: bytes, pdf_name: str):
        from src.services.ocr_service import ocr_image_payloads

        is_first_request = self.request_count == 0
        self.request_count += 1

        start_time = time.time()

        page_results, page_images = self.classify_pdf(pdf_bytes)

        filled_pages = [r["page"] for r in page_results if r["prediction"] == "filled"]
        pages_payload = [
            {"page": p, "image_png": page_images[p].tobytes("png"), "pdf_name": pdf_name}
            for p in filled_pages
        ]

        ocr_results = ocr_image_payloads(pages_payload)
        pages = _merge_pages(page_results, ocr_results)

        processing_seconds = time.time() - start_time
        cold_start_seconds = (self.ollama_ready_ts - self.container_start_ts) if is_first_request else 0

        return {
            "pages": pages,
            "metrics": {
                "sheet_count": len(page_results),
                "filled_sheet_count": len(filled_pages),
                "processing_seconds": round(processing_seconds, 2),
                "is_first_request": is_first_request,
                "cold_start_seconds": round(cold_start_seconds, 2),
                "approx_cost_usd": round(((processing_seconds + cold_start_seconds) / 3600) * 0.80, 4),
            },
        }


web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi", "python-multipart", "pydantic")

@app.function(image=web_image)
@modal.asgi_app()
def fastapi_app():
    from src.main import app as web_app

    return web_app
