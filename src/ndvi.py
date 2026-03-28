"""
NDVI-based forest masking and loss detection.
Band order: [Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5]
"""

import numpy as np


def hls_invalid_pixel_mask(data: np.ndarray, qa_band: np.ndarray | None = None) -> np.ndarray:
    """Return True where pixels are invalid due to nodata, non-finite values, or QA flags."""
    spectral = data[:6]
    invalid = ~np.all(np.isfinite(spectral), axis=0)
    invalid |= np.all(spectral == 0, axis=0)

    if qa_band is not None:
        qa = qa_band.astype(np.uint8)
        cloud_bit = (qa >> 1) & 1
        shadow_bit = (qa >> 2) & 1
        snow_bit = (qa >> 5) & 1
        invalid |= (cloud_bit | shadow_bit | snow_bit).astype(bool)

    return invalid


def compute_ndvi(data: np.ndarray, invalid_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Compute NDVI from a 6-band array.

    Args:
        data: shape (6, H, W) — band order [Blue, Green, Red, NIR, SWIR1, SWIR2]

    Returns:
        NDVI array of shape (H, W). Invalid pixels are set to NaN.
    """
    nir = data[3]
    red = data[2]
    nir = nir.astype(float)
    red = red.astype(float)
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)
    if invalid_mask is not None:
        ndvi = ndvi.astype(float, copy=False)
        ndvi[invalid_mask] = np.nan
    return ndvi


def forest_mask_from_ndvi(ndvi: np.ndarray, threshold: float) -> np.ndarray:
    """Returns boolean mask: True where ndvi > threshold (forest pixels)."""
    return ndvi > threshold


def compute_forest_loss_mask(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float,
    before_invalid_mask: np.ndarray | None = None,
    after_invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute forest loss mask between two 6-band arrays.
    Returns boolean array: True where forest before but not after.
    """
    ndvi_before = compute_ndvi(before, before_invalid_mask)
    ndvi_after = compute_ndvi(after, after_invalid_mask)
    forest_before = forest_mask_from_ndvi(ndvi_before, threshold)
    forest_after = forest_mask_from_ndvi(ndvi_after, threshold)
    combined_invalid = np.zeros_like(forest_before, dtype=bool)
    if before_invalid_mask is not None:
        combined_invalid |= before_invalid_mask
    if after_invalid_mask is not None:
        combined_invalid |= after_invalid_mask
    combined_invalid |= ~np.isfinite(ndvi_before) | ~np.isfinite(ndvi_after)
    return (forest_before & ~forest_after) & ~combined_invalid


def validate_ndvi_for_scoring(
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    min_valid_fraction: float = 0.10,
    min_valid_pixels: int = 1000,
) -> tuple[bool, str]:
    """Return whether NDVI rasters are usable for downstream scoring."""
    if ndvi_before.shape != ndvi_after.shape:
        return False, "NDVI before/after arrays must have the same shape for scoring."

    before_valid = int(np.isfinite(ndvi_before).sum())
    after_valid = int(np.isfinite(ndvi_after).sum())
    overlap_mask = np.isfinite(ndvi_before) & np.isfinite(ndvi_after)
    overlap_valid = int(overlap_mask.sum())
    total_pixels = int(ndvi_before.size)
    overlap_fraction = (overlap_valid / total_pixels) if total_pixels else 0.0

    if before_valid == 0:
        return False, "NDVI validation failed: no valid pixels remained in the start-year raster."
    if after_valid == 0:
        return False, "NDVI validation failed: no valid pixels remained in the end-year raster."
    if overlap_valid == 0:
        return False, "NDVI validation failed: no overlapping valid pixels remained between years."
    if overlap_valid < min_valid_pixels:
        return (
            False,
            "NDVI validation failed: only "
            f"{overlap_valid} overlapping valid pixels remained; need at least "
            f"{min_valid_pixels}.",
        )
    if overlap_fraction < min_valid_fraction:
        return (
            False,
            "NDVI validation failed: only "
            f"{overlap_fraction:.2%} of raster pixels remained valid across both years; "
            f"need at least {min_valid_fraction:.2%}.",
        )

    return True, ""


def ndvi_stats(ndvi: np.ndarray) -> dict:
    """Returns basic statistics for an NDVI array."""
    valid = ndvi[np.isfinite(ndvi)]
    if valid.size == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
    }
