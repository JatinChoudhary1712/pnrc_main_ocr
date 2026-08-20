import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.routers.process import build_process_router


def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def create_app(worker_cls):
    app = FastAPI(title="PNRC OCR", dependencies=[Depends(require_api_key)])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_process_router(worker_cls))
    return app

