"""
Unit tests for src/cloud_masking and src/preprocessing.
All tests use synthetic numpy arrays — no real rasters or network needed.
"""

import pytest
import numpy as np
import affine
from pyproj import CRS

from src.cloud_masking import unusable_fraction
from src.preprocessing import (
    utm_crs_from_centroid,
    assert_pixel_alignment,
    normalize_bands,
    generate_patches,
)


# ---------------------------------------------------------------------------
# utm_crs_from_centroid
# ---------------------------------------------------------------------------

class TestUtmCrsFromCentroid:
    def test_zone_43_north(self):
        """Karnataka, India centroid should be UTM zone 43N."""
        crs = utm_crs_from_centroid(78.5, 13.5)
        assert isinstance(crs, CRS)
        wkt = crs.to_wkt().upper()
        assert "43" in wkt
        # Northern hemisphere — "south" should NOT appear
        assert "SOUTH" not in wkt

    def test_zone_43_south(self):
        """Same longitude but southern latitude → UTM zone 43S."""
        crs = utm_crs_from_centroid(78.5, -13.5)
        assert isinstance(crs, CRS)
        wkt = crs.to_wkt().upper()
        assert "43" in wkt
        # pyproj encodes southern hemisphere via 10,000,000 m false northing,
        # not a literal "SOUTH" string in the WKT.
        assert "10000000" in wkt

    def test_returns_crs_object(self):
        crs = utm_crs_from_centroid(0.0, 0.0)
        assert isinstance(crs, CRS)


# ---------------------------------------------------------------------------
# assert_pixel_alignment
# ---------------------------------------------------------------------------

class TestAssertPixelAlignment:
    def _transform(self, origin_x=0.0, origin_y=0.0):
        return affine.Affine(30.0, 0.0, origin_x, 0.0, -30.0, origin_y)

    def test_passes_when_equal(self):
        t = self._transform()
        assert_pixel_alignment(t, (6, 100, 100), t, (6, 100, 100))  # no exception

    def test_raises_on_transform_mismatch(self):
        t1 = self._transform(0.0, 0.0)
        t2 = self._transform(30.0, 0.0)  # origin shifted
        with pytest.raises(ValueError, match="transform"):
            assert_pixel_alignment(t1, (6, 100, 100), t2, (6, 100, 100))

    def test_raises_on_shape_mismatch(self):
        t = self._transform()
        with pytest.raises(ValueError, match="shape"):
            assert_pixel_alignment(t, (6, 100, 100), t, (6, 101, 100))

    def test_passes_with_2d_shape(self):
        """Should work with (H, W) shapes (no band dimension)."""
        t = self._transform()
        assert_pixel_alignment(t, (100, 100), t, (100, 100))


# ---------------------------------------------------------------------------
# normalize_bands
# ---------------------------------------------------------------------------

class TestNormalizeBands:
    def test_raises_on_wrong_band_count(self):
        data = np.random.rand(7, 64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="6"):
            normalize_bands(data)

    def test_output_range(self):
        data = np.random.rand(6, 64, 64).astype(np.float32) * 10_000
        out = normalize_bands(data)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_output_shape(self):
        data = np.random.rand(6, 32, 32).astype(np.float32)
        out = normalize_bands(data)
        assert out.shape == (6, 32, 32)

    def test_constant_band_does_not_crash(self):
        data = np.ones((6, 32, 32), dtype=np.float32)
        out = normalize_bands(data)
        assert out.shape == (6, 32, 32)
        # All-same band → all zeros
        assert (out == 0.0).all()

    def test_five_bands_raises(self):
        data = np.random.rand(5, 64, 64).astype(np.float32)
        with pytest.raises(ValueError):
            normalize_bands(data)

    def test_invalid_pixels_are_excluded_from_min_max(self):
        data = np.array(
            [
                [[100.0, 1.0], [2.0, 3.0]],
                [[100.0, 1.0], [2.0, 3.0]],
                [[100.0, 1.0], [2.0, 3.0]],
                [[100.0, 1.0], [2.0, 3.0]],
                [[100.0, 1.0], [2.0, 3.0]],
                [[100.0, 1.0], [2.0, 3.0]],
            ],
            dtype=np.float32,
        )
        invalid_mask = np.zeros((2, 2), dtype=bool)
        invalid_mask[0, 0] = True

        out = normalize_bands(data, invalid_mask=invalid_mask)

        assert out[:, 0, 1].tolist() == pytest.approx([0.0] * 6)
        assert out[:, 1, 1].tolist() == pytest.approx([1.0] * 6)


# ---------------------------------------------------------------------------
# generate_patches
# ---------------------------------------------------------------------------

class TestGeneratePatches:
    def _make_arrays(self, H=256, W=256):
        before = np.random.rand(6, H, W).astype(np.float32)
        after = np.random.rand(6, H, W).astype(np.float32)
        return before, after

    def test_exact_divisible_count(self):
        """256×256 with patch_size=128 → 4 patches."""
        before, after = self._make_arrays(256, 256)
        patches = generate_patches(before, after, patch_size=128)
        assert len(patches) == 4

    def test_patch_keys(self):
        before, after = self._make_arrays(128, 128)
        patches = generate_patches(before, after, patch_size=128)
        assert len(patches) == 1
        p = patches[0]
        assert set(p.keys()) == {"before", "after", "row", "col"}

    def test_patch_shape(self):
        before, after = self._make_arrays(256, 256)
        patches = generate_patches(before, after, patch_size=128)
        for p in patches:
            assert p["before"].shape == (6, 128, 128)
            assert p["after"].shape == (6, 128, 128)

    def test_row_col_values(self):
        before, after = self._make_arrays(256, 256)
        patches = generate_patches(before, after, patch_size=128)
        row_cols = {(p["row"], p["col"]) for p in patches}
        assert row_cols == {(0, 0), (0, 128), (128, 0), (128, 128)}

    def test_non_divisible_size_has_more_patches(self):
        """257×257 with patch_size=128 → 3×3 = 9 patches (edge ones zero-padded)."""
        before, after = self._make_arrays(257, 257)
        patches = generate_patches(before, after, patch_size=128)
        assert len(patches) == 9

    def test_edge_patch_zero_padding(self):
        """For a 130×130 input with patch 128, the second col patch should be padded."""
        H, W = 130, 130
        before = np.ones((6, H, W), dtype=np.float32)
        after = np.ones((6, H, W), dtype=np.float32)
        patches = generate_patches(before, after, patch_size=128)
        # Patch at (0, 128) covers only 2 columns; rest should be zero
        edge_patch = next(p for p in patches if p["col"] == 128 and p["row"] == 0)
        # Columns 2..127 must be zero (padding)
        assert (edge_patch["before"][:, :, 2:] == 0).all()


# ---------------------------------------------------------------------------
# unusable_fraction (from cloud_masking)
# ---------------------------------------------------------------------------

class TestUnusableFraction:
    def test_all_true(self):
        mask = np.ones((100, 100), dtype=bool)
        assert unusable_fraction(mask) == pytest.approx(1.0)

    def test_all_false(self):
        mask = np.zeros((100, 100), dtype=bool)
        assert unusable_fraction(mask) == pytest.approx(0.0)

    def test_half(self):
        mask = np.zeros((100,), dtype=bool)
        mask[:50] = True
        assert unusable_fraction(mask) == pytest.approx(0.5)

    def test_empty_mask(self):
        mask = np.array([], dtype=bool)
        assert unusable_fraction(mask) == pytest.approx(0.0)
