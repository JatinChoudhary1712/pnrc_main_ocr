import logging
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import src.config  # noqa: F401 - loads .env before the app reads API_KEY
from src.routers.process import build_process_router

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Paths reachable without the X-API-Key header (RunPod health checks, API docs).
_OPEN_PATHS = {"/healthz", "/docs", "/redoc", "/openapi.json"}


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    if request.url.path in _OPEN_PATHS:
        return
    if x_api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def create_app():
    app = FastAPI(title="PNRC OCR", dependencies=[Depends(require_api_key)])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_process_router())
    return app


app = create_app()
