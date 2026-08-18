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

## OCR model
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OCR_MODEL_NAME = "GLM-OCR:latest"
OCR_PROMPT = """
Text Recognition.

Output only the recognized text.
Do not format the output as Markdown.
Preserve line breaks.
Stop after the last visible character.
"""

def get_device():
    """DINOv2 device: local CUDA when available, otherwise CPU."""
    import torch

    torch.set_num_threads(CPU_CORES)
    torch.set_num_interop_threads(CPU_CORES)
    return "cuda" if torch.cuda.is_available() else "cpu"

## Local API server (proxy layer)
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))

## API auth
API_KEY_HEADER = "X-API-Key"
API_KEY = os.environ.get("API_KEY")

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
