"""
Pydantic v2 schemas for the Carbon Monitor API.
"""

from typing import Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    geojson: dict
    start_year: int
    end_year: int
    annual_offset_tco2: Optional[float] = None


class AnalyzeResponse(BaseModel):
    status: str        # "queued"
    poll_url: str
    run_id: Optional[str] = None


class ResultsResponse(BaseModel):
    run_id: Optional[str] = None
    project_id: str
    status: str        # "running" | "complete" | "failed"
    created_at: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    segmentation_method: Optional[str] = None   # "prithvi" | "ndvi"
    biome: Optional[str] = None
    ndvi_threshold_used: Optional[float] = None
    sequestration_rate_used: Optional[float] = None
    forest_loss_ha: Optional[float] = None
    forest_loss_pct: Optional[float] = None
    risk_score: Optional[float] = None
    risk_flag: Optional[str] = None             # "HIGH" | "LOW" | "DATA_MISSING"
    iou_score: Optional[float] = None
    f1_score: Optional[float] = None
    ndvi_before_mean: Optional[float] = None
    ndvi_after_mean: Optional[float] = None
    forest_loss_map_url: Optional[str] = None
    ndvi_overlay_url: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    warnings: list[str] = []
