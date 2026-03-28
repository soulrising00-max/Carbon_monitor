from unittest.mock import patch

from src.pipeline import _resolve_scenes_with_fallback


def test_resolve_scenes_with_fallback_uses_previous_year_after_same_year_attempts():
    responses = [
        [],
        [],
        [{"granule_id": "g1", "cloud_cover": 12.0, "download_urls": [], "actual_tile_id": "T20MPU"}],
    ]

    with patch("src.pipeline.search_scenes", side_effect=responses) as mock_search:
        scenes, search_year, cloud_max = _resolve_scenes_with_fallback(
            "T20MPU",
            2023,
            (-1.0, -1.0, 1.0, 1.0),
            True,
        )

    assert len(scenes) == 1
    assert search_year == 2022
    assert cloud_max == 0.8
    assert mock_search.call_count == 3


def test_resolve_scenes_with_fallback_stops_at_same_year_relaxed_cloud():
    responses = [
        [],
        [{"granule_id": "g1", "cloud_cover": 25.0, "download_urls": [], "actual_tile_id": "T20MPU"}],
    ]

    with patch("src.pipeline.search_scenes", side_effect=responses) as mock_search:
        scenes, search_year, cloud_max = _resolve_scenes_with_fallback(
            "T20MPU",
            2023,
            (-1.0, -1.0, 1.0, 1.0),
            True,
        )

    assert len(scenes) == 1
    assert search_year == 2023
    assert cloud_max == 0.8
    assert mock_search.call_count == 2
