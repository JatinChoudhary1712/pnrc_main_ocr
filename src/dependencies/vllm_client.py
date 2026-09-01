import httpx

from src.config import OCR_CONCURRENCY, VLLM_API_KEY, VLLM_BASE_URL

# One pooled client, reused across the concurrent OCR fan-out. Pool size tracks
# OCR_CONCURRENCY so threads don't queue on connections.
_pool = OCR_CONCURRENCY + 4

vllm_client = httpx.Client(
    base_url=VLLM_BASE_URL,
    headers={"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {},
    timeout=httpx.Timeout(600.0, connect=15.0),
    limits=httpx.Limits(max_connections=_pool, max_keepalive_connections=_pool),
)
