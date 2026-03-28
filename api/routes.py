"""
FastAPI route handlers for the Carbon Monitor API.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import AnalyzeRequest, AnalyzeResponse
from src.pipeline import run_pipeline
from src.run_store import create_run, get_latest_run_for_project

router = APIRouter()

# In-memory results store — keyed by project_id
results_store: dict = {}


@router.post("/projects/{project_id}/analyze", response_model=AnalyzeResponse)
async def analyze(
    project_id: str,
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """Queue a new analysis job."""
    run_id = str(uuid.uuid4())
    created_at = create_run(run_id, project_id, "queued")
    results_store[project_id] = {
        "run_id": run_id,
        "project_id": project_id,
        "status": "queued",
        "created_at": created_at,
    }
    background_tasks.add_task(run_pipeline, run_id, project_id, req, results_store)
    return AnalyzeResponse(
        status="queued",
        poll_url=f"/projects/{project_id}/results",
        run_id=run_id,
    )


@router.get("/projects/{project_id}/results")
async def get_results(project_id: str) -> dict:
    """Poll for analysis results."""
    result = results_store.get(project_id)
    if result is None:
        persisted = get_latest_run_for_project(project_id)
        if persisted is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return persisted
    return result
