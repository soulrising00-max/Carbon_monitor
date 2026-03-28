"""
Tests for src/lpdaac.py

Unit tests run without network or credentials.
Integration tests require internet (CMR is public — no login needed for search).
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.lpdaac import search_scenes, select_top_scenes, validate_download


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scene_list(n: int) -> list[dict]:
    return [
        {
            "granule_id": f"HLS.L30.T43PHR.2020{i:03d}T050430",
            "cloud_cover": float(i * 5),
            "download_urls": [f"https://example.com/band_{i}.tif"],
            "actual_tile_id": "T43PHR",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# select_top_scenes — unit tests
# ---------------------------------------------------------------------------


def test_select_top_scenes_returns_n():
    assert len(select_top_scenes(_make_scene_list(5), 3)) == 3


def test_select_top_scenes_fewer_than_n():
    assert len(select_top_scenes(_make_scene_list(2), 3)) == 2


def test_select_top_scenes_empty():
    assert select_top_scenes([], 3) == []


def test_select_top_scenes_preserves_order():
    result = select_top_scenes(_make_scene_list(5), 2)
    assert result[0]["cloud_cover"] == 0.0
    assert result[1]["cloud_cover"] == 5.0


def test_select_top_scenes_exact_n():
    assert len(select_top_scenes(_make_scene_list(3), 3)) == 3


# ---------------------------------------------------------------------------
# validate_download — unit tests
# ---------------------------------------------------------------------------


def test_validate_download_nonexistent_path():
    assert validate_download(Path("/nonexistent/path/xyz")) is False


def test_validate_download_none():
    assert validate_download(None) is False


def test_validate_download_empty_dir(tmp_path):
    assert validate_download(tmp_path) is False


def test_validate_download_missing_expected_l30_bands(tmp_path):
    granule = tmp_path / "HLS.L30.T43PHR.2020020T050426.v2.0"
    granule.mkdir()
    for band in ["B02", "B03", "B04", "B11", "Fmask"]:
        (granule / f"{granule.name}.{band}.tif").write_bytes(b"fake")
    assert validate_download(granule) is False


def test_search_scenes_filters_to_requested_tile():
    entries = [
        {
            "producer_granule_id": "HLS.L30.T43PHR.2020020T050426.v2.0",
            "cloud_cover": 5.0,
        },
        {
            "producer_granule_id": "HLS.L30.T44PKA.2020020T050426.v2.0",
            "cloud_cover": 1.0,
        },
    ]

    with patch("src.lpdaac._cmr_search_bbox", side_effect=[entries, []]):
        results = search_scenes("T43PHR", 2020, cloud_max=1.0, bbox=(0, 0, 1, 1))

    assert [r["actual_tile_id"] for r in results] == ["T43PHR"]
    assert [r["granule_id"] for r in results] == [
        "HLS.L30.T43PHR.2020020T050426.v2.0"
    ]


def test_search_scenes_bbox_mode_keeps_adjacent_tiles():
    entries = [
        {
            "producer_granule_id": "HLS.L30.T43PHR.2020020T050426.v2.0",
            "cloud_cover": 5.0,
        },
        {
            "producer_granule_id": "HLS.L30.T44PKA.2020020T050426.v2.0",
            "cloud_cover": 1.0,
        },
    ]

    with patch("src.lpdaac._cmr_search_bbox", side_effect=[entries, []]):
        results = search_scenes(
            "bbox_fallback",
            2020,
            cloud_max=1.0,
            bbox=(0, 0, 1, 1),
            restrict_to_tile=False,
        )

    assert [r["actual_tile_id"] for r in results] == ["T44PKA", "T43PHR"]


def test_select_top_scenes_after_relaxed_filter():
    strict = []
    relaxed = _make_scene_list(3)
    assert select_top_scenes(strict, 3) == []
    assert len(select_top_scenes(relaxed, 3)) == 3


# ---------------------------------------------------------------------------
# Integration tests — require internet; CMR is public (no credentials needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_scenes_real_tile():
    """
    Uses bounding box search for Karnataka area (lon 78-79, lat 13-14).
    T43PHR is a confirmed real HLS tile in that area.
    """
    bbox = (78.0, 13.0, 79.0, 14.0)
    results = search_scenes("T43PHR", 2020, cloud_max=1.0, bbox=bbox)
    assert isinstance(results, list)
    assert len(results) >= 1, (
        f"Expected >=1 HLS scene for bbox {bbox} in 2020. Got 0. "
        "Check internet connection."
    )
    first = results[0]
    assert "granule_id" in first
    assert "cloud_cover" in first
    assert "download_urls" in first
    assert "actual_tile_id" in first
    # The actual tile IDs should all be real HLS tile IDs (start with T)
    assert first["actual_tile_id"].startswith("T")


@pytest.mark.integration
def test_search_scenes_sorted_by_cloud():
    bbox = (78.0, 13.0, 79.0, 14.0)
    results = search_scenes("T43PHR", 2020, cloud_max=1.0, bbox=bbox)
    if len(results) >= 2:
        covers = [r["cloud_cover"] for r in results]
        assert covers == sorted(covers)


@pytest.mark.integration
def test_search_scenes_cloud_filter():
    bbox = (78.0, 13.0, 79.0, 14.0)
    all_results = search_scenes("T43PHR", 2020, cloud_max=1.0, bbox=bbox)
    tight_results = search_scenes("T43PHR", 2020, cloud_max=0.10, bbox=bbox)
    all_ids = {r["granule_id"] for r in all_results}
    for r in tight_results:
        assert r["granule_id"] in all_ids
        assert r["cloud_cover"] <= 10.0


@pytest.mark.integration
def test_search_scenes_amazon():
    """Verify bounding box search also works for Amazon area."""
    bbox = (-77.0, 3.0, -76.0, 4.0)
    results = search_scenes("T18NTJ", 2020, cloud_max=1.0, bbox=bbox)
    assert isinstance(results, list)
    # Amazon is often cloudy so we just check structure, not count
    for r in results:
        assert "granule_id" in r
        assert r["cloud_cover"] <= 100.0
