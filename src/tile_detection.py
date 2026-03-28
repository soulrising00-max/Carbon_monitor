"""
MGRS tile grid loading and polygon-to-tile intersection detection.
Also provides biome classification based on centroid latitude.
"""

import json
from pathlib import Path

from shapely.geometry import box

from configs.settings import settings


def load_tile_grid(path: Path) -> list[dict]:
    """
    Load MGRS tile grid from GeoJSON.

    Returns:
        List of dicts: [{"tile_id": str, "bbox": (min_lon, min_lat, max_lon, max_lat)}, ...]

    Raises:
        FileNotFoundError: if path does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MGRS tile grid not found at: {path}. "
            "Make sure data/mgrs_tile_grid.geojson exists."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tiles = []
    for feature in data.get("features", []):
        tile_id = feature["properties"]["tile_id"]
        coords = feature["geometry"]["coordinates"][0]  # outer ring of polygon
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        tiles.append({"tile_id": tile_id, "bbox": bbox})

    return tiles


def find_covering_tiles(polygon, tile_grid: list[dict]) -> list[str]:
    """
    Find all MGRS tile IDs whose bounding boxes intersect the given polygon.

    Args:
        polygon: Shapely polygon geometry
        tile_grid: list of dicts from load_tile_grid()

    Returns:
        List of tile ID strings that intersect the polygon
    """
    covering = []
    for tile in tile_grid:
        tile_box = box(*tile["bbox"])
        if tile_box.intersects(polygon):
            covering.append(tile["tile_id"])
    return covering


def biome_params(centroid_lat: float) -> dict:
    """
    Return biome name, NDVI threshold, and sequestration rate for a given latitude.

    Rules:
        abs_lat <= 23  → tropical
        23 < abs_lat <= 40 → subtropical
        abs_lat > 40   → temperate

    Returns:
        {"biome": str, "ndvi_threshold": float, "sequestration_rate": float}
    """
    abs_lat = abs(centroid_lat)

    if abs_lat <= 23:
        biome = "tropical"
    elif abs_lat <= 40:
        biome = "subtropical"
    else:
        biome = "temperate"

    return {
        "biome": biome,
        "ndvi_threshold": settings.NDVI_THRESHOLDS[biome],
        "sequestration_rate": settings.SEQUESTRATION_RATES[biome],
    }
