import base64
import time

import httpx

from src.config import (
    OCR_CONCURRENCY,
    OCR_MAX_TOKENS,
    OCR_MODEL_NAME,
    OCR_PAGE_RETRIES,
    OCR_PROMPT,
    OCR_REASONING,
    OCR_TEMPERATURE,
    OCR_TOP_K,
    OCR_TOP_P,
)
from src.dependencies.vllm_client import vllm_client
from src.services.confidence_service import score_page_confidence
from src.utils.text_utils import clean_repetitions

_DEFAULTS = {
    "temperature": OCR_TEMPERATURE,
    "top_p": OCR_TOP_P,
    "top_k": OCR_TOP_K,
    "max_tokens": OCR_MAX_TOKENS,
    "reasoning": OCR_REASONING,
    "concurrency": OCR_CONCURRENCY,
    "page_retries": OCR_PAGE_RETRIES,
}


def resolve_params(overrides=None):
    """Merge per-request overrides onto the .env defaults; None/unknown keys ignored."""
    p = dict(_DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is not None and k in p:
            p[k] = v
    return p


def empty_page_result(page_number):
    conf = score_page_confidence(None)
    return {
        "page": page_number,
        "status": "empty",
        "text": None,
        "confidence": conf,
        "need_review": conf["is_low_confidence"],
    }


def ocr_page(image_png: bytes, params=None):
    """OCR one page PNG against the vLLM server. Returns (text, [logprob floats]).

    Relies on --reasoning-parser muse_glimmer: the transcription comes back in
    message.content, the model's chain-of-thought in reasoning_content (ignored).
    """
    p = params or _DEFAULTS
    data_url = "data:image/png;base64," + base64.b64encode(image_png).decode()
    payload = {
        "model": OCR_MODEL_NAME,
        "temperature": p["temperature"],
        "top_p": p["top_p"],
        "top_k": p["top_k"],
        "max_tokens": p["max_tokens"],
        "logprobs": True,
        "top_logprobs": 1,
        # muse_glimmer reasoning parser requires special tokens in the stream to
        # split reasoning_content from content.
        "skip_special_tokens": False,
        "messages": [
            {"role": "system", "content": f"Reasoning strength: {p['reasoning']}"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    last_err = None
    for attempt in range(4):
        try:
            resp = vllm_client.post("/chat/completions", json=payload)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_err = exc
            time.sleep(min(20, 5 * (attempt + 1)))
    else:
        raise RuntimeError(f"vLLM OCR failed after retries: {last_err}")

    choice = resp.json()["choices"][0]
    text = choice["message"].get("content") or ""
    entries = (choice.get("logprobs") or {}).get("content") or []
    return text, [e["logprob"] for e in entries]


def _ocr_one(item, params=None):
    """OCR one page. Retries any failure page_retries times, then returns an
    error page rather than raising so one bad page can't sink the whole job."""
    p = params or _DEFAULTS
    last_err = None
    for _ in range(p["page_retries"] + 1):
        try:
            text, logprobs = ocr_page(item["image_png"], p)
            cleaned = clean_repetitions(text).strip()
            conf = score_page_confidence(logprobs)
            result = {
                "page": item["page"],
                "status": "filled",
                "text": cleaned or None,
                "confidence": conf,
                "need_review": conf["is_low_confidence"],
            }
            if "pdf_name" in item:
                result["pdf_name"] = item["pdf_name"]
            return result
        except Exception as exc:
            last_err = exc

    result = {
        "page": item["page"],
        "status": "error",
        "text": None,
        "confidence": score_page_confidence(None),
        "need_review": True,
        "error": str(last_err),
    }
    if "pdf_name" in item:
        result["pdf_name"] = item["pdf_name"]
    return result
