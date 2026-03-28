"""
Prithvi-100M inference wrapper, evaluation metrics, and patch reconstruction.
"""

import numpy as np


def load_prithvi_model(device: str = "cpu"):
    """
    Load Prithvi-100M from HuggingFace: 'ibm-nasa-geospatial/Prithvi-100M'.

    Returns:
        (model, config) tuple

    Raises:
        RuntimeError: if loading fails for any reason
    """
    try:
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            "ibm-nasa-geospatial/Prithvi-100M", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "ibm-nasa-geospatial/Prithvi-100M", trust_remote_code=True
        )
        model.to(device)
        model.eval()
        return model, config
    except Exception as e:
        raise RuntimeError(f"Prithvi load failed: {e}")


def run_prithvi_inference(patch: np.ndarray, model, config) -> np.ndarray:
    """
    Run zero-shot segmentation on a single patch.

    Args:
        patch: shape (6, 128, 128)
        model: loaded Prithvi model
        config: model config

    Returns:
        binary forest mask of shape (128, 128)

    Raises:
        RuntimeError: on OOM or any runtime error
    """
    try:
        import torch

        tensor = torch.from_numpy(patch).float().unsqueeze(0)  # (1, 6, 128, 128)
        with torch.no_grad():
            output = model(tensor)

        # Extract segmentation output
        if hasattr(output, "logits"):
            logits = output.logits.squeeze(0)
        else:
            logits = output[0].squeeze(0)

        if logits.ndim == 3:
            mask = (logits.argmax(dim=0) > 0).numpy().astype(bool)
        else:
            mask = (logits > 0).numpy().astype(bool)

        return mask.astype(bool)
    except Exception as e:
        raise RuntimeError(f"Prithvi inference failed: {e}")


def evaluate_against_hansen(
    predicted_mask: np.ndarray, hansen_mask: np.ndarray
) -> dict:
    """
    Compute IoU, precision, recall, and F1 vs Hansen ground truth.

    Args:
        predicted_mask: boolean array
        hansen_mask:    boolean array (ground truth)

    Returns:
        {"iou": float, "precision": float, "recall": float, "f1": float}
    """
    predicted = predicted_mask.astype(bool)
    hansen = hansen_mask.astype(bool)

    tp = int(np.sum(predicted & hansen))
    fp = int(np.sum(predicted & ~hansen))
    fn = int(np.sum(~predicted & hansen))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def reconstruct_from_patches(
    patches: list, full_height: int, full_width: int
) -> np.ndarray:
    """
    Reassemble patch masks into a full-size boolean array.

    Args:
        patches:      list of dicts with keys "mask" (np.ndarray), "row" (int), "col" (int)
        full_height:  target array height
        full_width:   target array width

    Returns:
        boolean array of shape (full_height, full_width)
    """
    output = np.zeros((full_height, full_width), dtype=bool)
    for patch in patches:
        mask = patch["mask"]
        row = patch["row"]
        col = patch["col"]
        ph, pw = mask.shape
        row_end = min(row + ph, full_height)
        col_end = min(col + pw, full_width)
        output[row:row_end, col:col_end] = mask[: row_end - row, : col_end - col]
    return output
