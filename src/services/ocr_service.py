import os
import tempfile

import httpx
from tqdm import tqdm

from src.config import CPU_CORES, OCR_MODEL_NAME, OCR_PROMPT
from src.dependencies.ollama_client import ollama_client
from src.services.confidence_service import score_page_confidence
from src.utils.text_utils import clean_repetitions


def empty_page_result(page_number):
    conf = score_page_confidence(None)
    return {
        "page": page_number,
        "status": "empty",
        "text": None,
        "confidence": conf,
        "need_review": conf["is_low_confidence"],
    }


def ocr_page(image_path):
    resp = ollama_client.chat(
        model=OCR_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": OCR_PROMPT,
                "images": [str(image_path)],
            }
        ],
        options={
            "temperature": 0,
            "repeat_penalty": 1.1,
            "repeat_last_n": 128,
            "num_predict": 2048,
            "num_thread": CPU_CORES,
            "cache_prompt": False,
            "stop": [
                "The text in the image is as follows",
                "The image contains",
                "I see",
                "Wait,",
                "Here is the transcription",
            ],
        },
        logprobs=True,
        top_logprobs=1,
    )

    return resp["message"]["content"], resp.logprobs


def get_ocr_device():
    for model in ollama_client.ps().get("models", []):
        if model.get("model") != OCR_MODEL_NAME:
            continue

        size_vram = model.get("size_vram", 0)
        if size_vram == 0:
            return "CPU"

        return "GPU" if size_vram >= model.get("size", 1) else "cpu+gpu (partial offload)"

    return "not loaded"


def ocr_image_payloads(pages_payload):
    """OCR filled page PNG bytes. Runs on the Modal GPU worker."""

    results = []
    pbar = tqdm(pages_payload, desc=f"OCR - Model name : {OCR_MODEL_NAME}")
    for i, item in enumerate(pbar):
        fd, image_path = tempfile.mkstemp(suffix=".png")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(item["image_png"])

            try:
                text, logprobs = ocr_page(image_path)
            except httpx.RemoteProtocolError:
                text, logprobs = ocr_page(image_path)
        finally:
            os.unlink(image_path)

        cleaned = clean_repetitions(text).strip()
        conf = score_page_confidence(logprobs)
        result = {
            "page": item["page"],
            "status": "filled",
            "text": cleaned if cleaned else None,
            "confidence": conf,
            "need_review": conf["is_low_confidence"],
        }
        if "pdf_name" in item:
            result["pdf_name"] = item["pdf_name"]
        results.append(result)

        if i == 0:
            device = get_ocr_device()
            pbar.set_description(f"OCR - Model name : {OCR_MODEL_NAME} and on Device {device}")

    return results
