"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from configs.settings import settings
from src.run_store import init_db

# Ensure results directory exists before mounting
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
init_db()

app = FastAPI(title="Carbon Monitor API", version="1.0.0")
app.include_router(router)
app.mount(
    "/static",
    StaticFiles(directory=str(settings.RESULTS_DIR)),
    name="static",
)
