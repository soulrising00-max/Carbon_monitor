"""
MLflow experiment logging for Carbon Monitor pipeline runs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

from configs.settings import settings

logger = logging.getLogger(__name__)


def _tracking_uri_candidates() -> list[str]:
    return [
        f"sqlite:///{(settings.REPO_ROOT / 'mlflow.db').as_posix()}",
        f"file:///{(settings.REPO_ROOT / 'generated' / 'mlruns').as_posix()}",
    ]


def log_run(
    params: dict,
    metrics: dict,
    artifacts: list[Path],
    run_name: str,
    tags: dict | None = None,
    extra_json: dict | None = None,
) -> tuple[str, str]:
    """Create an MLflow run, log params/metrics/artifacts, and return the run_id.

    Expected params keys:
        project_id, segmentation_method, biome, ndvi_threshold,
        cloud_cover_threshold, sequestration_rate, scenes_per_year

    Expected metrics keys:
        iou_score, f1_score, forest_loss_ha, risk_score, unusable_pixel_pct

    Artifacts: list of Path objects (e.g. forest_loss.png, ndvi_overlay.png)
    """
    last_exc: Exception | None = None

    for tracking_uri in _tracking_uri_candidates():
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters (mlflow requires string values for params)
                for key, value in params.items():
                    if value is None:
                        continue
                    mlflow.log_param(key, str(value))

                # Log metrics (must be numeric; skip None values)
                for key, value in metrics.items():
                    if value is not None:
                        try:
                            mlflow.log_metric(key, float(value))
                        except (TypeError, ValueError):
                            pass  # skip non-numeric metrics silently

                if tags:
                    clean_tags = {k: str(v) for k, v in tags.items() if v is not None}
                    if clean_tags:
                        mlflow.set_tags(clean_tags)

                # Log artifact files that exist
                for artifact_path in artifacts:
                    artifact_path = Path(artifact_path)
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path))

                if extra_json:
                    mlflow.log_dict(extra_json, "run_summary.json")

                return run.info.run_id, tracking_uri
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("MLflow logging failed for tracking URI %s: %s", tracking_uri, exc)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("MLflow logging failed without a captured exception.")
