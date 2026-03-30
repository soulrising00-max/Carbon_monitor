"""
Forest segmentation inference wrapper, evaluation metrics, and patch reconstruction.

Replaces Prithvi-100M with a lightweight U-Net (ForestUNet) that:
  - Runs comfortably on CPU (~0.5 s per patch)
  - Loads trained weights from ml_models/unet_forest.pth when available
  - Falls back to random weights with a warning when no .pth file is found
  - Exposes identical function signatures so pipeline.py needs zero changes

Model input: 12 bands — before (6) + after (6) stacked along the channel axis.
  patch["before"]: (6, 128, 128)  — normalised spectral bands from year_before
  patch["after"]:  (6, 128, 128)  — normalised spectral bands from year_after
  stacked:         (12, 128, 128) — fed to ForestUNet

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
# Checkpoint key remapping
# ---------------------------------------------------------------------------

def _remap_state_dict(state: dict) -> dict:
    """
    Translate checkpoint keys saved by the Kaggle training notebook's inline
    ForestUNet (short attribute names) to the keys expected by models/unet.py
    (long attribute names).

    Notebook attribute names  →  unet.py attribute names:
        inc.b.*         → inc.block.*
        d1.b.1.b.*      → down1.block.1.block.*
        d2.b.1.b.*      → down2.block.1.block.*
        d3.b.1.b.*      → down3.block.1.block.*
        d4.b.1.b.*      → down4.block.1.block.*
        u1.c.b.*        → up1.conv.block.*
        u2.c.b.*        → up2.conv.block.*
        u3.c.b.*        → up3.conv.block.*
        u4.c.b.*        → up4.conv.block.*
        out.*           → out_conv.*

    If the keys already match (i.e. the checkpoint was saved from unet.py
    directly), the state dict is returned unchanged.
    """
    # Quick check: if the expected keys are already present, skip remapping
    first_key = next(iter(state), "")
    if first_key.startswith("inc.block") or first_key.startswith("down"):
        return state

    # Ordered replacement rules — more specific patterns must come first
    # so that e.g. "d1.b.1.b." is matched before a hypothetical "d1.b."
    _RULES = [
        # Encoder blocks (Down wraps DoubleConv, so two levels of .block)
        ("d1.b.1.b.", "down1.block.1.block."),
        ("d2.b.1.b.", "down2.block.1.block."),
        ("d3.b.1.b.", "down3.block.1.block."),
        ("d4.b.1.b.", "down4.block.1.block."),
        # Decoder blocks (Up uses .conv which contains DoubleConv)
        ("u1.c.b.", "up1.conv.block."),
        ("u2.c.b.", "up2.conv.block."),
        ("u3.c.b.", "up3.conv.block."),
        ("u4.c.b.", "up4.conv.block."),
        # Input conv
        ("inc.b.", "inc.block."),
        # Output head
        ("out.", "out_conv."),
    ]

    remapped = {}
    for old_key, tensor in state.items():
        new_key = old_key
        for old_prefix, new_prefix in _RULES:
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix):]
                break
        remapped[new_key] = tensor

    return remapped


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

        model = ForestUNet(in_channels=12, base_features=64)

        weights_path = _DEFAULT_WEIGHTS
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=device)

            # Support both raw state_dict and checkpoint dicts
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint

            # Remap short notebook keys → long unet.py keys if needed
            state = _remap_state_dict(state)

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
            "in_channels": 12,
            "base_features": 64,
            "patch_size": 128,
            "weights_loaded": weights_path.exists(),
            "device": device,
        }

        return model, config

    except Exception as e:
        raise RuntimeError(f"Prithvi load failed: {e}") from e


def run_prithvi_inference(patch: dict, model, config) -> np.ndarray:
    """
    Run forest segmentation on a single before/after patch pair.

    Args:
        patch:  dict with keys:
                  "before" — np.ndarray (6, 128, 128), float32, bands normalised to [0, 1]
                  "after"  — np.ndarray (6, 128, 128), float32, bands normalised to [0, 1]
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

        # Stack before + after along channel axis: (6,H,W) + (6,H,W) → (12,H,W)
        stacked = np.concatenate([patch["before"], patch["after"]], axis=0)  # (12, 128, 128)

        # (12, H, W) → (1, 12, H, W)
        tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)

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
