import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import src.config  # noqa: F401 - loads .env before the app reads API_KEY
from src.routers.process import build_process_router


def require_api_key(x_api_key: str = Header(...)):
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

