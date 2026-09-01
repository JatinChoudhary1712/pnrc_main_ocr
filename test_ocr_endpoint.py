"""Manual OCR check against a Muse-Glimmer vLLM (OpenAI-compatible) endpoint.

Muse-Glimmer is a reasoning VLM: the server must run with
    --reasoning-parser muse_glimmer --tool-call-parser muse_glimmer
so the final transcription lands in message.content and the model's
thinking lands in message.reasoning_content (which we ignore by default).

Sampling follows Meta's recipe (temp 1.0 / top_p 0.95 / top_k 64) -- the
model card says do NOT use greedy. Override via env if needed.

Usage:
    export OCR_BASE_URL=http://localhost:8000/v1        # or the RunPod proxy URL
    export OCR_API_KEY=the-secret-from---api-key
    python test_ocr_endpoint.py page1.png scan.jpg doc.pdf ...

PDFs are rendered page-by-page at the pipeline's PDF_DPI; each page's text
is written to <doc>_p<N>.<tag>.txt next to the PDF.

Env knobs:
    OCR_MODEL      served-model-name           (default "muse-glimmer")
    OCR_TAG        output filename suffix      (default "nvfp4")
    OCR_REASONING  "low"|"medium"|"high"|"xhigh" system prompt effort (default "low")
    OCR_TEMP / OCR_TOP_P / OCR_TOP_K           sampling overrides
    OCR_SHOW_THINKING=1                        also print reasoning_content
"""
import base64
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from src.config import OCR_PROMPT, PDF_DPI
from src.utils.text_utils import clean_repetitions

BASE_URL = os.environ["OCR_BASE_URL"].rstrip("/")
API_KEY = os.environ.get("OCR_API_KEY", "")
MODEL = os.environ.get("OCR_MODEL", "muse-glimmer")
TAG = os.environ.get("OCR_TAG", "nvfp4")
REASONING = os.environ.get("OCR_REASONING", "low")
TEMP = float(os.environ.get("OCR_TEMP", "1.0"))
TOP_P = float(os.environ.get("OCR_TOP_P", "0.95"))
TOP_K = int(os.environ.get("OCR_TOP_K", "64"))
SHOW_THINKING = os.environ.get("OCR_SHOW_THINKING") == "1"
CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "1"))
TOKEN_CONF_THRESHOLD = 0.7  # matches src/config.py TOKEN_CONFIDENCE_THRESHOLD

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def pdf_page_pngs(pdf_path: Path) -> list[tuple[str, bytes]]:
    """Render every page to PNG bytes at the pipeline's DPI."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=PDF_DPI)
        pages.append((f"{pdf_path.name}#p{i + 1}", pix.tobytes("png")))
    doc.close()
    return pages


def ocr(png_bytes: bytes, mime: str = "image/png") -> dict:
    data_url = f"data:{mime};base64," + base64.b64encode(png_bytes).decode()
    payload = {
        "model": MODEL,
        "temperature": TEMP,
        "top_p": TOP_P,
        "top_k": TOP_K,  # vLLM accepts this outside the OpenAI spec
        "max_tokens": 4096,
        "logprobs": True,
        "top_logprobs": 1,
        "messages": [
            {"role": "system", "content": f"Reasoning strength: {REASONING}"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    started = time.monotonic()
    last_err = None
    for attempt in range(5):
        try:
            resp = httpx.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=600,
            )
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            break
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            wait = min(30, 5 * (attempt + 1))
            print(f"  retry {attempt + 1}/5 after {type(e).__name__} ({e}); sleeping {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"gave up after 5 attempts: {last_err}")
    elapsed = time.monotonic() - started
    choice = resp.json()["choices"][0]
    msg = choice["message"]

    raw_text = (msg.get("content") or "").strip()
    text = clean_repetitions(raw_text).strip()

    tokens = (choice.get("logprobs") or {}).get("content") or []
    confs = [math.exp(t["logprob"]) for t in tokens]
    low_frac = sum(c < TOKEN_CONF_THRESHOLD for c in confs) / len(confs) if confs else 0.0

    return {
        "text": text,
        "thinking": (msg.get("reasoning_content") or "").strip(),
        "elapsed": elapsed,
        "n_tokens": len(tokens),
        "finish_reason": choice.get("finish_reason"),
        "low_conf_fraction": low_frac,
    }


def iter_inputs(paths: list[str]):
    """Yield (out_stem: Path, label: str, png_bytes: bytes, mime: str)."""
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".pdf":
            for label, png in pdf_page_pngs(path):
                stem = path.with_name(label.replace("#", "_").replace(".pdf", ""))
                yield stem, label, png, "image/png"
        else:
            yield path.with_suffix(""), path.name, path.read_bytes(), MIME.get(
                path.suffix.lower(), "image/png"
            )


def _one(job):
    stem, label, png, mime = job
    r = ocr(png, mime)
    out = stem.with_suffix(f".{TAG}.txt")
    out.write_text(r["text"])
    return label, out, r


def main(paths: list[str]) -> None:
    jobs = list(iter_inputs(paths))
    wall_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = pool.map(_one, jobs)
        for label, out, r in results:
            print(f"\n=== {label}  [{TAG}] ===")
            print(
                f"time {r['elapsed']:.1f}s | final-tokens {r['n_tokens']} | "
                f"finish={r['finish_reason']} | low-confidence {r['low_conf_fraction']:.1%}"
            )
            if SHOW_THINKING and r["thinking"]:
                print("--- reasoning ---")
                print(r["thinking"])
            print("--- transcription ---")
            print(r["text"])
            print(f"-> wrote {out}")

    wall = time.monotonic() - wall_start
    print(f"\n[{len(jobs)} pages | concurrency={CONCURRENCY} | wall {wall:.1f}s | {wall / len(jobs):.1f}s/page]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
