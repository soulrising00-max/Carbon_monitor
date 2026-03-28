"""
Cloud masking utilities for HLS (Harmonized Landsat Sentinel-2) imagery.
Uses the Fmask QA band with bit-level parsing; falls back to brightness threshold.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from pyproj import Transformer

from shapely.ops import transform as shapely_transform

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping
except ImportError:
    rasterio = None
    rio_mask = None
    mapping = None


def compute_cloud_mask(
    qa_band_path: Path,
    polygon_geom,
    band_paths: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute a boolean cloud/shadow/snow mask from the HLS Fmask QA band.

    Parameters
    ----------
    qa_band_path : Path
        Path to the Fmask GeoTIFF band file.
    polygon_geom : shapely geometry
        Polygon used to clip the QA raster before masking.
    band_paths : dict, optional
        {'B02': Path, 'B03': Path, 'B04': Path} — used only when QA read fails
        to fall back to a brightness-threshold mask.

    Returns
    -------
    np.ndarray (bool)
        True = unusable pixel (cloud / cloud shadow / snow-ice).
    """
    if rasterio is None or rio_mask is None or mapping is None:
        raise ImportError(
            "cloud masking requires rasterio and shapely to be installed."
        )

    try:
        with rasterio.open(qa_band_path) as src:
            geom = polygon_geom
            if src.crs:
                project = Transformer.from_crs(
                    "EPSG:4326", src.crs, always_xy=True
                ).transform
                geom = shapely_transform(project, polygon_geom)
            geom_json = [mapping(geom)]
            clipped, _ = rio_mask(src, geom_json, crop=True, filled=True, nodata=0)
        qa = clipped[0].astype(np.uint8)

        # HLS Fmask bit definitions
        # bit 1 = cloud, bit 2 = cloud shadow, bit 5 = snow/ice
        cloud_bit = (qa >> 1) & 1
        shadow_bit = (qa >> 2) & 1
        snow_bit = (qa >> 5) & 1
        bad_mask = (cloud_bit | shadow_bit | snow_bit).astype(bool)
        return bad_mask

    except Exception:
        # Fallback: brightness threshold on visible bands
        if band_paths is None:
            raise RuntimeError(
                "QA band read failed and no fallback band_paths provided."
            )
        arrays = []
        for key in ("B02", "B03", "B04"):
            with rasterio.open(band_paths[key]) as src:
                geom = polygon_geom
                if src.crs:
                    project = Transformer.from_crs(
                        "EPSG:4326", src.crs, always_xy=True
                    ).transform
                    geom = shapely_transform(project, polygon_geom)
                clipped, _ = rio_mask(
                    src,
                    [mapping(geom)],
                    crop=True,
                    filled=True,
                    nodata=0,
                )
                arrays.append(clipped[0].astype(float))
        brightness = (arrays[0] + arrays[1] + arrays[2]) / 3.0
        return brightness > 0.3


def unusable_fraction(cloud_mask: np.ndarray) -> float:
    """
    Return the fraction of True (unusable) pixels in the mask.

    Parameters
    ----------
    cloud_mask : np.ndarray (bool)

    Returns
    -------
    float in [0.0, 1.0]
    """
    if cloud_mask.size == 0:
        return 0.0
    return float(np.sum(cloud_mask)) / float(cloud_mask.size)
