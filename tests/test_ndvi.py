"""
Tests for src/ndvi.py and evaluation utilities in src/prithvi.py.
All tests use synthetic numpy arrays — no model downloads or network required.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_6band(H=64, W=64, nir_val=0.5, red_val=0.2):
    data = np.zeros((6, H, W), dtype=np.float32)
    data[3] = nir_val  # NIR
    data[2] = red_val  # Red
    return data


# ---------------------------------------------------------------------------
# compute_ndvi
# ---------------------------------------------------------------------------


def test_compute_ndvi_output_shape():
    from src.ndvi import compute_ndvi

    data = make_6band(H=32, W=48)
    ndvi = compute_ndvi(data)
    assert ndvi.shape == (32, 48)


def test_compute_ndvi_pure_nir():
    """Band 3 = 1.0, Band 2 = 0.0 → NDVI ≈ 1.0"""
    from src.ndvi import compute_ndvi

    data = make_6band(nir_val=1.0, red_val=0.0)
    ndvi = compute_ndvi(data)
    assert np.allclose(ndvi, 1.0, atol=1e-5)


def test_compute_ndvi_equal_nir_red():
    """Equal NIR and Red → NDVI ≈ 0.0"""
    from src.ndvi import compute_ndvi

    data = make_6band(nir_val=0.5, red_val=0.5)
    ndvi = compute_ndvi(data)
    assert np.allclose(ndvi, 0.0, atol=1e-4)


def test_compute_ndvi_masks_invalid_pixels():
    from src.ndvi import compute_ndvi

    data = make_6band(H=2, W=2, nir_val=1.0, red_val=0.0)
    invalid_mask = np.array([[False, True], [False, False]])

    ndvi = compute_ndvi(data, invalid_mask=invalid_mask)

    assert np.isnan(ndvi[0, 1])
    assert np.isfinite(ndvi[0, 0])


# ---------------------------------------------------------------------------
# forest_mask_from_ndvi
# ---------------------------------------------------------------------------


def test_forest_mask_from_ndvi():
    from src.ndvi import compute_ndvi, forest_mask_from_ndvi

    data = np.zeros((6, 4, 4), dtype=np.float32)
    # Top half: high NIR → forest
    data[3, :2, :] = 1.0
    data[2, :2, :] = 0.0
    # Bottom half: equal → not forest
    data[3, 2:, :] = 0.3
    data[2, 2:, :] = 0.3

    ndvi = compute_ndvi(data)
    mask = forest_mask_from_ndvi(ndvi, threshold=0.5)

    assert mask[:2, :].all(), "Top half should be forest"
    assert not mask[2:, :].any(), "Bottom half should not be forest"


# ---------------------------------------------------------------------------
# compute_forest_loss_mask
# ---------------------------------------------------------------------------


def test_compute_forest_loss_mask_detects_loss():
    from src.ndvi import compute_forest_loss_mask

    before = np.zeros((6, 4, 4), dtype=np.float32)
    before[3] = 1.0
    before[2] = 0.0

    after = np.zeros((6, 4, 4), dtype=np.float32)
    after[3] = 0.1
    after[2] = 0.5

    loss = compute_forest_loss_mask(before, after, threshold=0.5)
    assert loss.all(), "All pixels should show loss"


def test_compute_forest_loss_mask_no_loss():
    from src.ndvi import compute_forest_loss_mask

    data = make_6band(nir_val=1.0, red_val=0.0)
    loss = compute_forest_loss_mask(data, data, threshold=0.5)
    assert not loss.any()


def test_compute_forest_loss_mask_excludes_invalid_pixels():
    from src.ndvi import compute_forest_loss_mask

    before = np.zeros((6, 2, 2), dtype=np.float32)
    before[3] = 1.0
    before[2] = 0.0

    after = np.zeros((6, 2, 2), dtype=np.float32)
    after[3] = 0.1
    after[2] = 0.5

    after_invalid_mask = np.array([[True, False], [False, False]])
    loss = compute_forest_loss_mask(
        before,
        after,
        threshold=0.5,
        after_invalid_mask=after_invalid_mask,
    )

    assert not loss[0, 0]
    assert loss[0, 1]


def test_hls_invalid_pixel_mask_uses_qa_and_nodata():
    from src.ndvi import hls_invalid_pixel_mask

    data = make_6band(H=2, W=2, nir_val=1.0, red_val=0.0)
    data[:, 1, 1] = 0.0
    qa = np.zeros((2, 2), dtype=np.uint8)
    qa[0, 1] = 0b10

    invalid = hls_invalid_pixel_mask(data, qa)

    assert invalid[0, 1]
    assert invalid[1, 1]
    assert not invalid[0, 0]


def test_ndvi_stats_ignores_nan_pixels():
    from src.ndvi import ndvi_stats

    ndvi = np.array([[0.5, np.nan], [0.1, 0.3]])
    stats = ndvi_stats(ndvi)

    assert stats["mean"] == pytest.approx(0.3)


def test_validate_ndvi_for_scoring_rejects_no_valid_pixels():
    from src.ndvi import validate_ndvi_for_scoring

    ndvi_before = np.full((2, 2), np.nan)
    ndvi_after = np.full((2, 2), np.nan)

    is_valid, message = validate_ndvi_for_scoring(ndvi_before, ndvi_after)

    assert not is_valid
    assert "no valid pixels" in message.lower()


def test_validate_ndvi_for_scoring_accepts_overlapping_valid_pixels():
    from src.ndvi import validate_ndvi_for_scoring

    ndvi_before = np.full((40, 40), 0.5)
    ndvi_after = np.full((40, 40), 0.4)

    is_valid, message = validate_ndvi_for_scoring(ndvi_before, ndvi_after)

    assert is_valid
    assert message == ""


def test_validate_ndvi_for_scoring_rejects_too_few_pixels():
    from src.ndvi import validate_ndvi_for_scoring

    ndvi_before = np.full((20, 20), 0.5)
    ndvi_after = np.full((20, 20), 0.4)

    is_valid, message = validate_ndvi_for_scoring(ndvi_before, ndvi_after)

    assert not is_valid
    assert "at least 1000" in message


def test_validate_ndvi_for_scoring_rejects_low_fraction():
    from src.ndvi import validate_ndvi_for_scoring

    ndvi_before = np.full((100, 100), np.nan)
    ndvi_after = np.full((100, 100), np.nan)
    ndvi_before[:5, :100] = 0.5
    ndvi_after[:5, :100] = 0.4

    is_valid, message = validate_ndvi_for_scoring(
        ndvi_before,
        ndvi_after,
        min_valid_fraction=0.10,
        min_valid_pixels=100,
    )

    assert not is_valid
    assert "10.00%" in message


# ---------------------------------------------------------------------------
# evaluate_against_hansen
# ---------------------------------------------------------------------------


def test_evaluate_against_hansen_iou():
    """TP=3, FP=1, FN=1 → IoU = 3/5 = 0.6"""
    from src.prithvi import evaluate_against_hansen

    predicted = np.array([True, True, True, True, False], dtype=bool)
    hansen = np.array([True, True, True, False, True], dtype=bool)
    result = evaluate_against_hansen(predicted, hansen)
    assert abs(result["iou"] - 0.6) < 1e-5
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_evaluate_against_hansen_all_zero_predicted():
    from src.prithvi import evaluate_against_hansen

    predicted = np.zeros((10,), dtype=bool)
    hansen = np.ones((10,), dtype=bool)
    result = evaluate_against_hansen(predicted, hansen)
    assert result["iou"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_evaluate_against_hansen_perfect_match():
    from src.prithvi import evaluate_against_hansen

    mask = np.array([True, False, True, True], dtype=bool)
    result = evaluate_against_hansen(mask, mask)
    assert abs(result["iou"] - 1.0) < 1e-5
    assert abs(result["f1"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# reconstruct_from_patches
# ---------------------------------------------------------------------------


def test_reconstruct_from_patches_shape_and_placement():
    """4 patches of 128×128 reassemble into a 256×256 array with correct quadrant values."""
    from src.prithvi import reconstruct_from_patches

    patch_size = 128
    full_h, full_w = 256, 256

    # Quadrants: (0,0)=False, (0,128)=True, (128,0)=False, (128,128)=True
    patches = [
        {"mask": np.zeros((patch_size, patch_size), dtype=bool), "row": 0, "col": 0},
        {"mask": np.ones((patch_size, patch_size), dtype=bool), "row": 0, "col": 128},
        {"mask": np.zeros((patch_size, patch_size), dtype=bool), "row": 128, "col": 0},
        {"mask": np.ones((patch_size, patch_size), dtype=bool), "row": 128, "col": 128},
    ]

    result = reconstruct_from_patches(patches, full_h, full_w)
    assert result.shape == (full_h, full_w)
    assert result.dtype == bool
    assert not result[:128, :128].any()
    assert result[:128, 128:].all()
    assert not result[128:, :128].any()
    assert result[128:, 128:].all()


def test_reconstruct_from_patches_all_true():
    from src.prithvi import reconstruct_from_patches

    patches = [
        {"mask": np.ones((128, 128), dtype=bool), "row": 0, "col": 0},
        {"mask": np.ones((128, 128), dtype=bool), "row": 0, "col": 128},
        {"mask": np.ones((128, 128), dtype=bool), "row": 128, "col": 0},
        {"mask": np.ones((128, 128), dtype=bool), "row": 128, "col": 128},
    ]
    result = reconstruct_from_patches(patches, 256, 256)
    assert result.all()


# ---------------------------------------------------------------------------
# Prithvi integration stub
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_prithvi_load_and_inference():
    pytest.skip("requires Prithvi model download")
