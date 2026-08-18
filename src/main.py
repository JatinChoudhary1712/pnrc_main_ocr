from fastapi import FastAPI

from src.routers.cost import build_cost_router
from src.routers.process import build_process_router


def create_app(worker_cls):
    app = FastAPI(title="PNRC OCR")
    app.include_router(build_process_router(worker_cls))
    app.include_router(build_cost_router(worker_cls))
    return app

