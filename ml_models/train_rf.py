"""Train or load a tile-level Random Forest risk model from NDVI-derived features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ml_integration import (
    ML_RESULTS_DIR,
    RF_MODEL_PATH,
    build_rf_dataset,
    discover_project_pairs,
    evaluate_rf_model,
    ensure_ml_dirs,
    train_or_load_rf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or reuse a Random Forest tile-risk model."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help=(
            "Optional CSV containing project_id, cloud_fraction, biome columns. "
            "NDVI features are still derived from existing clipped rasters."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ml_dirs()

    project_pairs = discover_project_pairs()
    dataset_rows = build_rf_dataset(project_pairs=project_pairs, metadata_csv=args.metadata_csv)
    result = train_or_load_rf(dataset_rows=dataset_rows, model_path=RF_MODEL_PATH)
    model = result["model"]
    evaluation = evaluate_rf_model(dataset_rows, model=model, model_path=RF_MODEL_PATH)
    prediction_summaries = []
    predicted_values = model.predict(
        np.stack([row["feature_vector"] for row in dataset_rows]).astype(float)
    )
    for row, pred in zip(dataset_rows, predicted_values, strict=False):
        prediction_summaries.append(
            {
                "project_id": row["project_id"],
                "risk_score": float(pred),
                "pseudo_target": float(row["pseudo_target"]),
                "source": "rf" if RF_MODEL_PATH.exists() else "pseudo_target",
            }
        )

    summary_path = ML_RESULTS_DIR / "train_rf_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trained": result["trained"],
                "model_path": str(RF_MODEL_PATH),
                "metrics_path": result.get("metrics_path") or evaluation["metrics_path"],
                "predictions_path": result.get("predictions_path") or evaluation["predictions_path"],
                "r2": evaluation["r2"],
                "mae": evaluation["mae"],
                "predictions": prediction_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
