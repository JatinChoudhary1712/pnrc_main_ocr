from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

## CPU usage limit — cap thread pools so this process leaves headroom on shared servers.
## Must be set before torch/numpy spin up their thread pools, so this stays above those imports.
CPU_CORES = int(os.environ.get("CPU_CORES", "12"))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_CORES))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_CORES))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_CORES))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_CORES))

KB_FOLDER = BASE_DIR / "knowledge_base"
EMBEDDINGS_FILE = KB_FOLDER / "embeddings.npy"
METADATA_FILE = KB_FOLDER / "metadata.json"

## DINOv2 model
MODEL_NAME = "facebook/dinov2-base"

## OCR model — vLLM (OpenAI-compatible), e.g. Muse-Glimmer via vllm/vllm-openai:muse-glimmer.
## The server must run with --reasoning-parser muse_glimmer --tool-call-parser muse_glimmer
## so the transcription lands in message.content and the model's thinking in reasoning_content.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "muse-glimmer")

## Sampling — Muse-Glimmer's card says do NOT use greedy; keep reasoning minimal for OCR.
OCR_TEMPERATURE = float(os.environ.get("OCR_TEMPERATURE", "1.0"))
OCR_TOP_P = float(os.environ.get("OCR_TOP_P", "0.95"))
OCR_TOP_K = int(os.environ.get("OCR_TOP_K", "64"))
OCR_MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "4096"))
OCR_REASONING = os.environ.get("OCR_REASONING", "low")  # low | medium | high | xhigh

## How many pages to OCR concurrently against the vLLM server.
OCR_CONCURRENCY = int(os.environ.get("OCR_CONCURRENCY", "12"))

OCR_PROMPT = """
You are an OCR text recognition engine.

Transcribe all visible text in the image exactly as it appears.

Rules:

* Output only the recognized text. Do not add explanations, comments, or descriptions.
* Do not format the output as Markdown.
* Preserve the original line breaks and paragraph structure.
* Preserve the original reading order.
* Preserve spelling, capitalization, punctuation, numbers, symbols, and special characters exactly as visible.
* Do not correct spelling or grammar.
* Do not translate the text.
* Do not summarize, interpret, or infer missing text.
* Do not add text that is not visibly present.
* If a character or word is unclear, make the best OCR transcription based only on what is visible.
* Preserve tables and structured text using plain text while maintaining their visual reading order.
* Include headers, footers, labels, handwritten text, stamps, and other visible text when readable.
* Ignore purely visual elements that contain no text.
* Stop immediately after the last visible character.

Return only the transcription.
"""

def get_device():
    """DINOv2 device: local CUDA when available, otherwise CPU."""
    import torch

    torch.set_num_threads(CPU_CORES)
    torch.set_num_interop_threads(CPU_CORES)
    return "cuda" if torch.cuda.is_available() else "cpu"

os.environ.setdefault("HF_HUB_OFFLINE", "1")

## Classification
TOP_N = 10
MARGIN = 0.0

## OCR confidence
TOKEN_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_FRACTION_THRESHOLD = 0.1
LOW_CONFIDENCE_RUN_LENGTH = 3

## Image rendering
PDF_DPI = 150

## Batching
MAX_FILES_PER_BATCH = int(os.environ.get("MAX_FILES_PER_BATCH", "8"))

## Container lifecycle
SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN_WINDOW", "15"))

## Cost estimation
L4_HOURLY_RATE = 0.80
