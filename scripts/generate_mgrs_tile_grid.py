"""
Generate a global UTM-based MGRS 100 km tile grid as GeoJSON.

This covers standard UTM MGRS latitude bands C..X (-80 to 84 degrees),
excluding the polar UPS regions. Output polygons are clipped to the
corresponding UTM zone and latitude band in WGS84 lon/lat.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "mgrs_tile_grid.geojson"

LAT_BANDS = [
    ("C", -80, -72),
    ("D", -72, -64),
    ("E", -64, -56),
    ("F", -56, -48),
    ("G", -48, -40),
    ("H", -40, -32),
    ("J", -32, -24),
    ("K", -24, -16),
    ("L", -16, -8),
    ("M", -8, 0),
    ("N", 0, 8),
    ("P", 8, 16),
    ("Q", 16, 24),
    ("R", 24, 32),
    ("S", 32, 40),
    ("T", 40, 48),
    ("U", 48, 56),
    ("V", 56, 64),
    ("W", 64, 72),
    ("X", 72, 84),
]

ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
COLUMN_SETS = (
    "ABCDEFGH",
    "JKLMNPQR",
    "STUVWXYZ",
)


def _utm_crs(zone: int, lat_band: str) -> CRS:
    epsg = 32600 + zone if lat_band >= "N" else 32700 + zone
    return CRS.from_epsg(epsg)


def _zone_lon_bounds(zone: int) -> tuple[float, float]:
    lon_min = -180 + (zone - 1) * 6
    return lon_min, lon_min + 6


def _column_letter(zone: int, easting: int) -> str:
    sequence = COLUMN_SETS[(zone - 1) % 3]
    index = (easting // 100000) - 1
    return sequence[index]


def _row_letter(zone: int, northing: int) -> str:
    offset = 0 if zone % 2 == 1 else 5
    index = ((northing // 100000) + offset) % len(ROW_LETTERS)
    return ROW_LETTERS[index]


def _tile_id(zone: int, lat_band: str, easting: int, northing: int) -> str:
    return f"T{zone:02d}{lat_band}{_column_letter(zone, easting)}{_row_letter(zone, northing)}"


def _iter_zone_band_tiles(zone: int, lat_band: str, min_lat: float, max_lat: float):
    lon_min, lon_max = _zone_lon_bounds(zone)
    zone_band_wgs84 = box(lon_min, min_lat, lon_max, max_lat)

    utm = _utm_crs(zone, lat_band)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    zone_band_utm = transform(to_utm, zone_band_wgs84)
    minx, miny, maxx, maxy = zone_band_utm.bounds

    easting_start = max(100000, int(minx // 100000) * 100000)
    easting_end = min(900000, int(maxx // 100000) * 100000 + 100000)
    northing_start = max(0, int(miny // 100000) * 100000)
    northing_end = min(10000000, int(maxy // 100000) * 100000 + 100000)

    for easting in range(easting_start, easting_end, 100000):
        if not 100000 <= easting <= 800000:
            continue
        for northing in range(northing_start, northing_end, 100000):
            utm_square = box(easting, northing, easting + 100000, northing + 100000)
            if not utm_square.intersects(zone_band_utm):
                continue

            wgs84_square = transform(to_wgs84, utm_square)
            clipped = wgs84_square.intersection(zone_band_wgs84)
            if clipped.is_empty or clipped.area <= 0:
                continue

            yield {
                "type": "Feature",
                "properties": {"tile_id": _tile_id(zone, lat_band, easting, northing)},
                "geometry": mapping(clipped),
            }


def generate_mgrs_grid() -> dict:
    features = []
    for zone in range(1, 61):
        for lat_band, min_lat, max_lat in LAT_BANDS:
            features.extend(_iter_zone_band_tiles(zone, lat_band, min_lat, max_lat))

    return {
        "type": "FeatureCollection",
        "name": "mgrs_tile_grid",
        "features": features,
    }


def main() -> None:
    grid = generate_mgrs_grid()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(grid), encoding="utf-8")
    print(f"Wrote {len(grid['features'])} features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
