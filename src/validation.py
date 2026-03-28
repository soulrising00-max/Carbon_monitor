"""
Input validation for the Carbon Project Land Cover Monitor analyze request.
Validates GeoJSON geometry, coordinate ranges, and year constraints.
"""

from shapely.geometry import shape
# from configs.settings import settings


def validate_analyze_request(
    geojson: dict, start_year: int, end_year: int
) -> tuple[list, str]:
    """
    Validate an analyze request.

    Returns:
        (shapely_geometries_list, error_message)
        On success: ([shapely polygon objects], "")
        On failure: ([], "human-readable error")
    """

    # 1. Check top-level GeoJSON structure
    if not isinstance(geojson, dict):
        return [], "GeoJSON must be a dict"
    if "type" not in geojson:
        return [], "GeoJSON missing required key: 'type'"
    if "features" not in geojson:
        return [], "GeoJSON missing required key: 'features'"
    if not isinstance(geojson["features"], list):
        return [], "'features' must be a list"

    # 2. Each feature must have geometry with type and coordinates
    for i, feature in enumerate(geojson["features"]):
        if not isinstance(feature, dict):
            return [], f"Feature {i} is not a dict"
        geometry = feature.get("geometry")
        if geometry is None:
            return [], f"Feature {i} missing 'geometry'"
        if "type" not in geometry:
            return [], f"Feature {i} geometry missing 'type'"
        if "coordinates" not in geometry:
            return [], f"Feature {i} geometry missing 'coordinates'"

    # 3. At least one Polygon or MultiPolygon
    allowed_types = {"Polygon", "MultiPolygon"}
    polygon_features = [
        f for f in geojson["features"]
        if isinstance(f.get("geometry"), dict)
        and f["geometry"].get("type") in allowed_types
    ]
    if not polygon_features:
        return [], "GeoJSON must contain at least one Polygon or MultiPolygon geometry"

    # 4. Coordinates must not be zero-length
    for i, feature in enumerate(polygon_features):
        coords = feature["geometry"]["coordinates"]
        if not coords or len(coords) == 0:
            return [], f"Feature {i} has empty coordinates array"

    # 5 & 6. Validate coordinate ranges and collect all coords
    def _iter_coords(coords, geom_type):
        """Flatten coordinate pairs regardless of nesting depth."""
        if geom_type == "Polygon":
            for ring in coords:
                for lon, lat, *_ in ring:
                    yield lon, lat
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    for lon, lat, *_ in ring:
                        yield lon, lat

    for i, feature in enumerate(polygon_features):
        geom = feature["geometry"]
        for lon, lat in _iter_coords(geom["coordinates"], geom["type"]):
            if not (-180 <= lon <= 180):
                return [], f"Feature {i} has out-of-range longitude: {lon}"
            if not (-90 <= lat <= 90):
                return [], f"Feature {i} has out-of-range latitude: {lat}"

    # 7. Year validation
    if start_year >= end_year:
        return [], f"start_year ({start_year}) must be less than end_year ({end_year})"
    if start_year < 2013:
        return [], f"start_year ({start_year}) must be >= 2013 (HLS data availability)"
    if end_year < 2013:
        return [], f"end_year ({end_year}) must be >= 2013 (HLS data availability)"

    # Convert to Shapely geometries
    geometries = []
    for feature in polygon_features:
        try:
            geom = shape(feature["geometry"])
            geometries.append(geom)
        except Exception as e:
            return [], f"Failed to parse geometry: {e}"

    return geometries, ""
