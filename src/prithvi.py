"""
Forest segmentation inference wrapper, evaluation metrics, and patch reconstruction.

Replaces Prithvi-100M with a lightweight U-Net (ForestUNet) that:
  - Runs comfortably on CPU (~0.5 s per patch)
  - Loads trained weights from ml_models/unet_forest.pth when available
  - Falls back to random weights with a warning when no .pth file is found
  - Exposes identical function signatures so pipeline.py needs zero changes

Upgrade path:
  1. Train on Kaggle (see notebooks/training.ipynb)
  2. Download unet_forest.pth
  3. Drop into ml_models/  — loader picks it up automatically
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default weight path — resolved relative to repo root
_DEFAULT_WEIGHTS = Path(__file__).parent.parent / "ml_models" / "unet_forest.pth"


# ---------------------------------------------------------------------------
# Public API  (signatures unchanged from original prithvi.py)
# ---------------------------------------------------------------------------

def load_prithvi_model(device: str = "cpu"):
    """
    Build ForestUNet and load weights if available.

    Tries to load from ml_models/unet_forest.pth.
    If the file is missing, uses random weights and logs a warning —
    the pipeline will still run; IoU will be low and NDVI fallback will trigger.

    Args:
        device: torch device string, e.g. "cpu" or "cuda"

    Returns:
        (model, config) tuple
            model:  ForestUNet instance in eval mode
            config: dict with model metadata

    Raises:
        RuntimeError: if model construction itself fails (should never happen)
    """
    try:
        import torch
        from models.unet import ForestUNet

        model = ForestUNet(in_channels=6, base_features=64)

        weights_path = _DEFAULT_WEIGHTS
        if weights_path.exists():
            state = torch.load(weights_path, map_location=device)
            # Support both raw state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            logger.info("Loaded U-Net weights from %s", weights_path)
        else:
            logger.warning(
                "No weights found at %s — using random weights. "
                "IoU will be low; pipeline will fall back to NDVI masks. "
                "Train on Kaggle and drop unet_forest.pth into ml_models/ to fix this.",
                weights_path,
            )

        model.to(device)
        model.eval()

        config = {
            "model_type": "ForestUNet",
            "in_channels": 6,
            "base_features": 64,
            "patch_size": 128,
            "weights_loaded": weights_path.exists(),
            "device": device,
        }

        return model, config

    except Exception as e:
        raise RuntimeError(f"Prithvi load failed: {e}") from e


def run_prithvi_inference(patch: np.ndarray, model, config) -> np.ndarray:
    """
    Run forest segmentation on a single patch.

    Args:
        patch:  shape (6, 128, 128), float32, bands normalized to [0, 1]
        model:  ForestUNet instance returned by load_prithvi_model()
        config: config dict returned by load_prithvi_model()

    Returns:
        binary forest mask of shape (128, 128), dtype bool

    Raises:
        RuntimeError: on OOM or any runtime error
    """
    try:
        import torch
        from models.unet import logits_to_mask

        device = config.get("device", "cpu")

        # (6, H, W) → (1, 6, H, W)
        tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)          # (1, 1, 128, 128)

        mask = logits_to_mask(logits)       # (1, 128, 128) bool
        return mask.squeeze(0).cpu().numpy().astype(bool)  # (128, 128)

    except Exception as e:
        raise RuntimeError(f"Prithvi inference failed: {e}") from e


def evaluate_against_hansen(
    predicted_mask: np.ndarray, hansen_mask: np.ndarray
) -> dict:
    """
    Compute IoU, precision, recall, and F1 vs Hansen ground truth.

    Args:
        predicted_mask: boolean array (any shape)
        hansen_mask:    boolean array, same shape (ground truth)

    Returns:
        {"iou": float, "precision": float, "recall": float, "f1": float}
    """
    predicted = predicted_mask.astype(bool)
    hansen = hansen_mask.astype(bool)

    tp = int(np.sum(predicted & hansen))
    fp = int(np.sum(predicted & ~hansen))
    fn = int(np.sum(~predicted & hansen))

    iou       = tp / (tp + fp + fn)          if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)               if (tp + fp)      > 0 else 0.0
    recall    = tp / (tp + fn)               if (tp + fn)      > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) \
                if (precision + recall) > 0 else 0.0

    return {
        "iou":       float(iou),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def reconstruct_from_patches(
    patches: list, full_height: int, full_width: int
) -> np.ndarray:
    """
    Reassemble patch masks into a full-size boolean array.

    Args:
        patches:      list of dicts with keys "mask" (np.ndarray), "row" (int), "col" (int)
        full_height:  target array height in pixels
        full_width:   target array width in pixels

    Returns:
        boolean array of shape (full_height, full_width)
    """
    output = np.zeros((full_height, full_width), dtype=bool)
    for patch in patches:
        mask = patch["mask"]
        row  = patch["row"]
        col  = patch["col"]
        ph, pw  = mask.shape
        row_end = min(row + ph, full_height)
        col_end = min(col + pw, full_width)
        output[row:row_end, col:col_end] = mask[: row_end - row, : col_end - col]
    return output
