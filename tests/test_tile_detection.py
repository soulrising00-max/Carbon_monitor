"""
Tests for src/tile_detection.py — load_tile_grid, find_covering_tiles, biome_params.
"""

import pytest
from shapely.geometry import box

from src.tile_detection import load_tile_grid, find_covering_tiles, biome_params
from src.validation import validate_analyze_request


# ── load_tile_grid ─────────────────────────────────────────────────────────────

class TestLoadTileGrid:
    def test_returns_nonempty_list(self, tile_grid):
        assert isinstance(tile_grid, list)
        assert len(tile_grid) > 0

    def test_each_entry_has_correct_keys(self, tile_grid):
        for entry in tile_grid:
            assert "tile_id" in entry
            assert "bbox" in entry

    def test_bbox_is_four_tuple(self, tile_grid):
        for entry in tile_grid:
            bbox = entry["bbox"]
            assert len(bbox) == 4
            min_lon, min_lat, max_lon, max_lat = bbox
            assert min_lon < max_lon
            assert min_lat < max_lat

    def test_raises_file_not_found_for_bad_path(self):
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            load_tile_grid(Path("/nonexistent/path/mgrs_tile_grid.geojson"))


# ── find_covering_tiles ────────────────────────────────────────────────────────

class TestFindCoveringTiles:
    def test_test_polygon_is_covered(self, test_polygon_geojson, tile_grid):
        geoms, err = validate_analyze_request(test_polygon_geojson, 2020, 2023)
        assert err == "", f"Validation failed: {err}"
        polygon = geoms[0]
        tiles = find_covering_tiles(polygon, tile_grid)
        assert tiles, "Expected at least one covering MGRS tile"

    def test_polar_polygon_returns_empty(self, tile_grid):
        # Standard UTM MGRS excludes the UPS/polar region above 84N.
        polar_poly = box(0.0, 85.0, 0.1, 85.1)
        tiles = find_covering_tiles(polar_poly, tile_grid)
        assert tiles == [], f"Expected no tiles but got {tiles}"

    def test_returns_list_type(self, test_polygon_geojson, tile_grid):
        geoms, _ = validate_analyze_request(test_polygon_geojson, 2020, 2023)
        result = find_covering_tiles(geoms[0], tile_grid)
        assert isinstance(result, list)


# ── biome_params ──────────────────────────────────────────────────────────────

class TestBiomeParams:
    def test_tropical_lat(self):
        result = biome_params(13.5)
        assert result["biome"] == "tropical"

    def test_subtropical_lat(self):
        result = biome_params(30.0)
        assert result["biome"] == "subtropical"

    def test_temperate_lat(self):
        result = biome_params(50.0)
        assert result["biome"] == "temperate"

    def test_tropical_boundary_lat_23(self):
        result = biome_params(23.0)
        assert result["biome"] == "tropical"

    def test_subtropical_boundary_lat_40(self):
        result = biome_params(40.0)
        assert result["biome"] == "subtropical"

    def test_temperate_above_40(self):
        result = biome_params(41.0)
        assert result["biome"] == "temperate"

    def test_southern_hemisphere_tropical(self):
        result = biome_params(-13.5)
        assert result["biome"] == "tropical"

    def test_returns_correct_keys(self):
        result = biome_params(13.5)
        assert "biome" in result
        assert "ndvi_threshold" in result
        assert "sequestration_rate" in result

    def test_ndvi_threshold_is_float(self):
        result = biome_params(13.5)
        assert isinstance(result["ndvi_threshold"], float)

    def test_sequestration_rate_is_float(self):
        result = biome_params(13.5)
        assert isinstance(result["sequestration_rate"], float)
