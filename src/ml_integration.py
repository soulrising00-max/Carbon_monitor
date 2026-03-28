"""
Modular ML/DL integration for NDVI-driven land-cover monitoring.

This module is intentionally standalone so existing pipeline code does not need
to change. Future callers can import these helpers from `src.pipeline.py` or
any other orchestration layer for optional inference.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from configs.settings import settings
from src.ndvi import compute_ndvi, hls_invalid_pixel_mask

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


MODELS_DIR = settings.REPO_ROOT / "models"
ML_RESULTS_DIR = settings.RESULTS_DIR / "ml"
UNET_MODEL_PATH = MODELS_DIR / "unet_v1.pt"
RF_MODEL_PATH = MODELS_DIR / "rf_tile_model.joblib"
DEFAULT_BIOMES = ["tropical", "subtropical", "temperate", "unknown"]


@dataclass(slots=True)
class RasterPair:
    """Container for paired before/after rasters and metadata."""

    project_id: str
    before: np.ndarray
    after: np.ndarray
    before_path: Path
    after_path: Path
    transform: Any
    crs: Any


def ensure_ml_dirs() -> tuple[Path, Path]:
    """Create model and ML result directories when missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR, ML_RESULTS_DIR


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError(
            "PyTorch is required for U-Net training/inference. "
            "Install torch to use this integration."
        )


def _require_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "scikit-learn and joblib are required for RF training/inference."
        ) from exc

    return joblib, RandomForestRegressor, mean_absolute_error, r2_score, train_test_split


def _read_raster(path: Path) -> tuple[np.ndarray, Any, Any]:
    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
    if data.shape[0] == 1:
        data = data[0]
    return data, transform, crs


def _sorted_clipped_rasters(project_dir: Path) -> list[Path]:
    rasters = sorted(project_dir.glob("clipped_*.tif"))
    return rasters


def discover_project_pairs(results_root: Path | None = None) -> list[RasterPair]:
    """Discover before/after raster pairs from existing project outputs."""
    root = results_root or settings.RESULTS_DIR
    pairs: list[RasterPair] = []

    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir() or project_dir.name == "ml":
            continue
        rasters = _sorted_clipped_rasters(project_dir)
        if len(rasters) < 2:
            continue
        before_path, after_path = rasters[0], rasters[-1]
        before, transform, crs = _read_raster(before_path)
        after, _, _ = _read_raster(after_path)
        pairs.append(
            RasterPair(
                project_id=project_dir.name,
                before=before,
                after=after,
                before_path=before_path,
                after_path=after_path,
                transform=transform,
                crs=crs,
            )
        )
    return pairs


def _extract_spectral_and_invalid(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim == 2:
        spectral = data[np.newaxis, ...].astype(np.float32)
        invalid = ~np.isfinite(spectral[0])
        return spectral, invalid

    spectral = data[:6].astype(np.float32)
    qa_band = data[6] if data.shape[0] > 6 else None
    invalid = hls_invalid_pixel_mask(spectral, qa_band)
    return spectral, invalid


def derive_ndvi(data: np.ndarray) -> np.ndarray:
    """Return NDVI from a raster stack, or the raster itself if already single-band."""
    if data.ndim == 2:
        ndvi = data.astype(np.float32)
        ndvi[~np.isfinite(ndvi)] = np.nan
        return ndvi

    spectral, invalid = _extract_spectral_and_invalid(data)
    if spectral.shape[0] < 4:
        raise ValueError("Multi-band inputs need at least 4 channels to derive NDVI.")
    return compute_ndvi(spectral, invalid).astype(np.float32)


def build_unet_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Build a compact feature cube for segmentation.

    Single-band inputs produce 3 channels: before, after, delta.
    Multi-band inputs produce before(6) + after(6) + NDVI before/after/delta.
    """
    if before.shape != after.shape:
        raise ValueError("Before/after rasters must have the same shape.")

    if before.ndim == 2:
        before_ndvi = derive_ndvi(before)
        after_ndvi = derive_ndvi(after)
        return np.stack(
            [before_ndvi, after_ndvi, after_ndvi - before_ndvi], axis=0
        ).astype(np.float32)

    before_spec, before_invalid = _extract_spectral_and_invalid(before)
    after_spec, after_invalid = _extract_spectral_and_invalid(after)
    before_ndvi = compute_ndvi(before_spec, before_invalid).astype(np.float32)
    after_ndvi = compute_ndvi(after_spec, after_invalid).astype(np.float32)
    ndvi_delta = (after_ndvi - before_ndvi).astype(np.float32)
    features = np.concatenate(
        [
            before_spec,
            after_spec,
            np.stack([before_ndvi, after_ndvi, ndvi_delta], axis=0),
        ],
        axis=0,
    )
    features[~np.isfinite(features)] = 0.0
    return features.astype(np.float32)


def build_pseudo_labels(
    before: np.ndarray,
    after: np.ndarray,
    forest_threshold: float = 0.35,
    delta_threshold: float = 0.15,
) -> np.ndarray:
    """Create pseudo forest-loss labels from NDVI change."""
    before_ndvi = derive_ndvi(before)
    after_ndvi = derive_ndvi(after)
    delta = after_ndvi - before_ndvi
    labels = (
        np.isfinite(before_ndvi)
        & np.isfinite(after_ndvi)
        & (before_ndvi >= forest_threshold)
        & (delta <= -abs(delta_threshold))
    )
    return labels.astype(np.float32)


def _compute_iou_f1(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    ref = ref_mask.astype(bool)
    intersection = float(np.logical_and(pred, ref).sum())
    union = float(np.logical_or(pred, ref).sum())
    pred_sum = float(pred.sum())
    ref_sum = float(ref.sum())
    iou = intersection / union if union else 1.0
    precision = intersection / pred_sum if pred_sum else 1.0
    recall = intersection / ref_sum if ref_sum else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 1.0
    )
    return {"iou": iou, "f1": f1}


def _patchify(
    features: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    samples_x: list[np.ndarray] = []
    samples_y: list[np.ndarray] = []
    _, height, width = features.shape

    if height < patch_size or width < patch_size:
        pad_h = max(0, patch_size - height)
        pad_w = max(0, patch_size - width)
        features = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)))
        labels = np.pad(labels, ((0, pad_h), (0, pad_w)))
        _, height, width = features.shape

    step = max(16, patch_size // 2)
    for row in range(0, height - patch_size + 1, step):
        for col in range(0, width - patch_size + 1, step):
            x_patch = features[:, row:row + patch_size, col:col + patch_size]
            y_patch = labels[row:row + patch_size, col:col + patch_size]
            samples_x.append(x_patch.astype(np.float32))
            samples_y.append(y_patch[np.newaxis, ...].astype(np.float32))

    if not samples_x:
        raise RuntimeError("Could not extract any training patches for U-Net.")
    return np.stack(samples_x), np.stack(samples_y)


if nn is not None:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.block(x)


    class SmallUNet(nn.Module):
        """A lightweight U-Net for pseudo-label segmentation."""

        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.enc1 = DoubleConv(in_channels, 16)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = DoubleConv(16, 32)
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(32, 64)
            self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(32, 16)
            self.head = nn.Conv2d(16, 1, kernel_size=1)

        def forward(self, x: Tensor) -> Tensor:
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool1(enc1))
            bottleneck = self.bottleneck(self.pool2(enc2))
            dec2 = self.up2(bottleneck)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            dec1 = self.up1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            return self.head(dec1)
else:  # pragma: no cover - only used when torch is unavailable
    class SmallUNet:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()


def _save_mask_outputs(
    mask: np.ndarray,
    project_id: str,
    stem: str,
    transform: Any = None,
    crs: Any = None,
) -> dict[str, str]:
    """Persist mask outputs as .npy and, when possible, GeoTIFF."""
    ensure_ml_dirs()
    out_dir = ML_RESULTS_DIR / project_id
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, mask.astype(np.uint8))

    outputs = {"npy": str(npy_path)}
    if transform is not None and crs is not None:
        tif_path = out_dir / f"{stem}.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype="uint8",
            transform=transform,
            crs=crs,
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        outputs["tif"] = str(tif_path)
    return outputs


def save_unet_predictions(
    project_pairs: list[RasterPair],
    model: Any | None = None,
    model_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Save predicted masks and IoU/F1 summaries for a set of project pairs."""
    summaries: list[dict[str, Any]] = []
    for pair in project_pairs:
        prediction = predict_unet_mask(
            pair.before,
            pair.after,
            model=model,
            model_path=model_path,
        )
        pseudo = build_pseudo_labels(pair.before, pair.after)
        metrics = _compute_iou_f1(prediction["mask"], pseudo)
        output_paths = _save_mask_outputs(
            prediction["mask"],
            pair.project_id,
            "unet_pred_mask",
            transform=pair.transform,
            crs=pair.crs,
        )
        summaries.append(
            {
                "project_id": pair.project_id,
                "iou": metrics["iou"],
                "f1": metrics["f1"],
                "mask_pixels": int(prediction["mask"].sum()),
                "source": prediction["source"],
                "outputs": output_paths,
            }
        )
    return summaries


def train_or_load_unet(
    project_pairs: list[RasterPair] | None = None,
    model_path: Path | None = None,
    epochs: int = 3,
    patch_size: int = 64,
    forest_threshold: float = 0.35,
    delta_threshold: float = 0.15,
) -> dict[str, Any]:
    """Train a lightweight U-Net on pseudo-labels or load an existing checkpoint."""
    _require_torch()
    ensure_ml_dirs()
    model_path = model_path or UNET_MODEL_PATH
    project_pairs = project_pairs or discover_project_pairs()
    if not project_pairs:
        raise RuntimeError(
            "No paired clipped rasters were found under results/. "
            "Run the pipeline first or pass explicit inputs."
        )

    example_features = build_unet_features(project_pairs[0].before, project_pairs[0].after)
    in_channels = int(example_features.shape[0])
    model = SmallUNet(in_channels)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return {"model": model, "trained": False, "model_path": str(model_path)}

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    for pair in project_pairs:
        features = build_unet_features(pair.before, pair.after)
        labels = build_pseudo_labels(
            pair.before,
            pair.after,
            forest_threshold=forest_threshold,
            delta_threshold=delta_threshold,
        )
        x_patches, y_patches = _patchify(features, labels, patch_size=patch_size)
        train_x.append(x_patches)
        train_y.append(y_patches)

    x_train = np.concatenate(train_x, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, len(loader)))

    torch.save(
        {"state_dict": model.state_dict(), "in_channels": in_channels},
        model_path,
    )
    model.eval()

    metrics_rows = save_unet_predictions(project_pairs, model=model, model_path=model_path)

    metrics_path = ML_RESULTS_DIR / "unet_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "epochs": epochs,
                "patch_size": patch_size,
                "mean_train_loss": float(np.mean(losses)) if losses else None,
                "projects": metrics_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "trained": True,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


def predict_unet_mask(
    before: np.ndarray,
    after: np.ndarray,
    model: Any | None = None,
    model_path: Path | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run U-Net inference, loading weights if needed."""
    _require_torch()
    ensure_ml_dirs()
    model_path = model_path or UNET_MODEL_PATH
    features = build_unet_features(before, after)
    in_channels = int(features.shape[0])

    if model is None:
        if not model_path.exists():
            mask = build_pseudo_labels(before, after).astype(np.uint8)
            return {"mask": mask, "source": "pseudo_label"}
        checkpoint = torch.load(model_path, map_location="cpu")
        model = SmallUNet(checkpoint.get("in_channels", in_channels))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    tensor = torch.from_numpy(features[np.newaxis, ...].astype(np.float32))
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs >= threshold).astype(np.uint8)
    return {"mask": mask, "score_map": probs, "source": "unet"}


def biome_one_hot(biome: str, biome_categories: list[str] | None = None) -> np.ndarray:
    categories = biome_categories or DEFAULT_BIOMES
    biome_norm = (biome or "unknown").strip().lower()
    if biome_norm not in categories:
        biome_norm = "unknown"
    return np.array([1.0 if b == biome_norm else 0.0 for b in categories], dtype=float)


def build_tile_risk_features(
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    cloud_fraction: float,
    biome: str,
    biome_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Create tile-level feature vectors for the RF model."""
    delta = ndvi_after - ndvi_before
    valid_mask = np.isfinite(ndvi_before) & np.isfinite(ndvi_after)
    if not np.any(valid_mask):
        raise ValueError("No valid NDVI pixels were available for RF features.")

    mean_before = float(np.nanmean(ndvi_before))
    mean_after = float(np.nanmean(ndvi_after))
    mean_delta = float(np.nanmean(delta))
    loss_fraction = float(
        np.mean((ndvi_before[valid_mask] >= 0.35) & (delta[valid_mask] <= -0.15))
    )
    feature_vector = np.concatenate(
        [
            np.array([mean_before, mean_after, mean_delta, float(cloud_fraction)], dtype=float),
            biome_one_hot(biome, biome_categories),
        ]
    )
    return {
        "mean_ndvi_before": mean_before,
        "mean_ndvi_after": mean_after,
        "delta_ndvi": mean_delta,
        "cloud_fraction": float(cloud_fraction),
        "biome": biome,
        "pseudo_target": loss_fraction,
        "feature_vector": feature_vector,
    }


def _load_feature_metadata(csv_path: Path | None) -> dict[str, dict[str, str]]:
    if csv_path is None or not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["project_id"]: row for row in reader if row.get("project_id")}


def build_rf_dataset(
    project_pairs: list[RasterPair] | None = None,
    metadata_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Build a dataset from existing clipped outputs and optional metadata."""
    project_pairs = project_pairs or discover_project_pairs()
    metadata = _load_feature_metadata(metadata_csv)
    rows: list[dict[str, Any]] = []

    for pair in project_pairs:
        before_ndvi = derive_ndvi(pair.before)
        after_ndvi = derive_ndvi(pair.after)
        meta = metadata.get(pair.project_id, {})
        cloud_fraction = float(meta.get("cloud_fraction", 0.0) or 0.0)
        biome = meta.get("biome", "unknown")
        feature_row = build_tile_risk_features(
            before_ndvi,
            after_ndvi,
            cloud_fraction=cloud_fraction,
            biome=biome,
        )
        feature_row["project_id"] = pair.project_id
        rows.append(feature_row)
    return rows


def train_or_load_rf(
    dataset_rows: list[dict[str, Any]] | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Train a Random Forest regressor or load a saved one."""
    ensure_ml_dirs()
    joblib, RandomForestRegressor, mean_absolute_error, r2_score, train_test_split = _require_sklearn()
    model_path = model_path or RF_MODEL_PATH
    dataset_rows = dataset_rows or build_rf_dataset()
    if not dataset_rows:
        raise RuntimeError(
            "No RF training rows were built from results/. "
            "Run the pipeline first or provide metadata CSV."
        )

    if model_path.exists():
        bundle = joblib.load(model_path)
        return {
            "model": bundle["model"],
            "biomes": bundle["biomes"],
            "trained": False,
            "model_path": str(model_path),
        }

    x = np.stack([row["feature_vector"] for row in dataset_rows]).astype(float)
    y = np.array([row["pseudo_target"] for row in dataset_rows], dtype=float)

    if len(dataset_rows) >= 4:
        x_train, x_test, y_train, y_test, rows_train, rows_test = train_test_split(
            x, y, dataset_rows, test_size=0.25, random_state=42
        )
    else:
        x_train = x_test = x
        y_train = y_test = y
        rows_train = rows_test = dataset_rows

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=42,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 1.0
    mae = float(mean_absolute_error(y_test, y_pred))

    bundle = {"model": model, "biomes": DEFAULT_BIOMES}
    joblib.dump(bundle, model_path)

    predictions_path = ML_RESULTS_DIR / "rf_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "project_id",
                "mean_ndvi_before",
                "mean_ndvi_after",
                "delta_ndvi",
                "cloud_fraction",
                "biome",
                "pseudo_target",
                "predicted_risk_score",
            ],
        )
        writer.writeheader()
        for row, pred in zip(rows_test, y_pred, strict=False):
            writer.writerow(
                {
                    "project_id": row["project_id"],
                    "mean_ndvi_before": row["mean_ndvi_before"],
                    "mean_ndvi_after": row["mean_ndvi_after"],
                    "delta_ndvi": row["delta_ndvi"],
                    "cloud_fraction": row["cloud_fraction"],
                    "biome": row["biome"],
                    "pseudo_target": row["pseudo_target"],
                    "predicted_risk_score": float(pred),
                }
            )

    metrics_path = ML_RESULTS_DIR / "rf_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "num_rows": len(dataset_rows),
                "r2": r2,
                "mae": mae,
                "prediction_log": str(predictions_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "biomes": DEFAULT_BIOMES,
        "trained": True,
        "r2": r2,
        "mae": mae,
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
    }


def evaluate_rf_model(
    dataset_rows: list[dict[str, Any]],
    model: Any | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate an RF model against pseudo-targets and log predictions."""
    ensure_ml_dirs()
    joblib, _, mean_absolute_error, r2_score, _ = _require_sklearn()
    model_path = model_path or RF_MODEL_PATH

    if model is None:
        if not model_path.exists():
            raise RuntimeError("RF model weights do not exist yet.")
        bundle = joblib.load(model_path)
        model = bundle["model"]

    x = np.stack([row["feature_vector"] for row in dataset_rows]).astype(float)
    y_true = np.array([row["pseudo_target"] for row in dataset_rows], dtype=float)
    y_pred = model.predict(x)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 1.0
    mae = float(mean_absolute_error(y_true, y_pred))

    predictions_path = ML_RESULTS_DIR / "rf_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "project_id",
                "mean_ndvi_before",
                "mean_ndvi_after",
                "delta_ndvi",
                "cloud_fraction",
                "biome",
                "pseudo_target",
                "predicted_risk_score",
            ],
        )
        writer.writeheader()
        for row, pred in zip(dataset_rows, y_pred, strict=False):
            writer.writerow(
                {
                    "project_id": row["project_id"],
                    "mean_ndvi_before": row["mean_ndvi_before"],
                    "mean_ndvi_after": row["mean_ndvi_after"],
                    "delta_ndvi": row["delta_ndvi"],
                    "cloud_fraction": row["cloud_fraction"],
                    "biome": row["biome"],
                    "pseudo_target": row["pseudo_target"],
                    "predicted_risk_score": float(pred),
                }
            )

    metrics_path = ML_RESULTS_DIR / "rf_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "num_rows": len(dataset_rows),
                "r2": r2,
                "mae": mae,
                "prediction_log": str(predictions_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "r2": r2,
        "mae": mae,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
    }


def predict_tile_risk(
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    cloud_fraction: float,
    biome: str,
    model: Any | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Run RF inference for tile-level forest loss risk."""
    ensure_ml_dirs()
    joblib, _, _, _, _ = _require_sklearn()
    model_path = model_path or RF_MODEL_PATH

    if model is None:
        if not model_path.exists():
            features = build_tile_risk_features(
                ndvi_before,
                ndvi_after,
                cloud_fraction=cloud_fraction,
                biome=biome,
            )
            return {
                "risk_score": features["pseudo_target"],
                "source": "pseudo_target",
                "features": features,
            }
        bundle = joblib.load(model_path)
        model = bundle["model"]

    features = build_tile_risk_features(
        ndvi_before,
        ndvi_after,
        cloud_fraction=cloud_fraction,
        biome=biome,
    )
    risk_score = float(model.predict(features["feature_vector"].reshape(1, -1))[0])
    return {"risk_score": risk_score, "source": "rf", "features": features}


def load_pair_from_paths(
    before_path: Path,
    after_path: Path,
    project_id: str = "manual",
) -> RasterPair:
    """Load a single before/after pair from explicit file paths."""
    before, transform, crs = _read_raster(before_path)
    after, _, _ = _read_raster(after_path)
    return RasterPair(
        project_id=project_id,
        before=before,
        after=after,
        before_path=before_path,
        after_path=after_path,
        transform=transform,
        crs=crs,
    )
