"""
Tests for src/validation.py — validate_analyze_request().
"""

import pytest
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from shapely.geometry.multipolygon import MultiPolygon as ShapelyMultiPolygon

from src.validation import validate_analyze_request


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_polygon_geojson(coords=None):
    """Return a minimal valid Polygon FeatureCollection."""
    if coords is None:
        coords = [[[78.4, 13.4], [78.6, 13.4], [78.6, 13.6], [78.4, 13.6], [78.4, 13.4]]]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {},
            }
        ],
    }


VALID_GEOJSON = _make_polygon_geojson()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestValidGeoJSON:
    def test_valid_polygon_returns_shapely_geoms(self):
        geoms, err = validate_analyze_request(VALID_GEOJSON, 2020, 2023)
        assert err == ""
        assert len(geoms) > 0
        assert isinstance(geoms[0], ShapelyPolygon)

    def test_valid_multipolygon_passes(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [[[78.4, 13.4], [78.6, 13.4], [78.6, 13.6], [78.4, 13.6], [78.4, 13.4]]],
                            [[[78.7, 13.7], [78.9, 13.7], [78.9, 13.9], [78.7, 13.9], [78.7, 13.7]]],
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        geoms, err = validate_analyze_request(geojson, 2020, 2023)
        assert err == ""
        assert len(geoms) > 0
        assert isinstance(geoms[0], ShapelyMultiPolygon)


class TestMissingKeys:
    def test_missing_features_key(self):
        bad = {"type": "FeatureCollection"}
        geoms, err = validate_analyze_request(bad, 2020, 2023)
        assert geoms == []
        assert "features" in err.lower()

    def test_missing_type_key(self):
        bad = {"features": []}
        geoms, err = validate_analyze_request(bad, 2020, 2023)
        assert geoms == []
        assert "type" in err.lower()


class TestGeometryType:
    def test_point_geometry_raises_error(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [78.5, 13.5]},
                    "properties": {},
                }
            ],
        }
        geoms, err = validate_analyze_request(geojson, 2020, 2023)
        assert geoms == []
        assert err != ""

    def test_linestring_geometry_raises_error(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[78.4, 13.4], [78.6, 13.6]],
                    },
                    "properties": {},
                }
            ],
        }
        geoms, err = validate_analyze_request(geojson, 2020, 2023)
        assert geoms == []
        assert err != ""


class TestCoordinateRanges:
    def test_out_of_range_longitude(self):
        coords = [[[200.0, 13.4], [200.5, 13.4], [200.5, 13.6], [200.0, 13.6], [200.0, 13.4]]]
        geoms, err = validate_analyze_request(_make_polygon_geojson(coords), 2020, 2023)
        assert geoms == []
        assert "longitude" in err.lower()

    def test_out_of_range_latitude(self):
        coords = [[[78.4, 95.0], [78.6, 95.0], [78.6, 96.0], [78.4, 96.0], [78.4, 95.0]]]
        geoms, err = validate_analyze_request(_make_polygon_geojson(coords), 2020, 2023)
        assert geoms == []
        assert "latitude" in err.lower()


class TestYearValidation:
    def test_start_year_equal_to_end_year(self):
        geoms, err = validate_analyze_request(VALID_GEOJSON, 2020, 2020)
        assert geoms == []
        assert "start_year" in err.lower() or "end_year" in err.lower()

    def test_start_year_greater_than_end_year(self):
        geoms, err = validate_analyze_request(VALID_GEOJSON, 2023, 2020)
        assert geoms == []
        assert err != ""

    def test_start_year_before_2013(self):
        geoms, err = validate_analyze_request(VALID_GEOJSON, 2010, 2023)
        assert geoms == []
        assert "2013" in err

    def test_end_year_after_start_must_both_be_at_least_2013(self):
        geoms, err = validate_analyze_request(VALID_GEOJSON, 2013, 2014)
        assert err == ""
        assert len(geoms) > 0
