"""Train or load a lightweight U-Net using existing project raster outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml_integration import (
    ML_RESULTS_DIR,
    UNET_MODEL_PATH,
    discover_project_pairs,
    ensure_ml_dirs,
    load_pair_from_paths,
    save_unet_predictions,
    train_or_load_unet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or reuse a U-Net for forest segmentation."
    )
    parser.add_argument("--before", type=Path, help="Path to start-year raster.")
    parser.add_argument("--after", type=Path, help="Path to end-year raster.")
    parser.add_argument(
        "--project-id",
        default="manual",
        help="Project identifier used for result folders when explicit rasters are supplied.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Small training budget for the fallback U-Net.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Patch size used to build segmentation samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ml_dirs()

    if args.before and args.after:
        project_pairs = [
            load_pair_from_paths(args.before, args.after, project_id=args.project_id)
        ]
    else:
        project_pairs = discover_project_pairs()

    result = train_or_load_unet(
        project_pairs=project_pairs,
        model_path=UNET_MODEL_PATH,
        epochs=args.epochs,
        patch_size=args.patch_size,
    )

    model = result["model"]
    prediction_summaries = save_unet_predictions(
        project_pairs,
        model=model,
        model_path=UNET_MODEL_PATH,
    )

    summary_path = ML_RESULTS_DIR / "train_unet_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trained": result["trained"],
                "model_path": str(UNET_MODEL_PATH),
                "metrics_path": result.get("metrics_path") or str(ML_RESULTS_DIR / "unet_metrics.json"),
                "predictions": prediction_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
