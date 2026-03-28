"""
Preprocessing utilities: UTM CRS derivation, mosaicking, clipping,
pixel-alignment checks, band normalisation, and patch generation.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform
# import affine

try:
    import rasterio
    from rasterio import MemoryFile
    from rasterio.merge import merge as rio_merge
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from shapely.geometry import mapping
except ImportError:
    rasterio = None
    MemoryFile = None
    rio_merge = None
    rio_mask = None
    reproject = None
    Resampling = None
    calculate_default_transform = None
    mapping = None


def _require_geospatial_dependencies() -> None:
    """Raise a clear error if raster/geospatial extras are unavailable."""
    if (
        rasterio is None
        or MemoryFile is None
        or rio_merge is None
        or rio_mask is None
        or reproject is None
        or Resampling is None
        or calculate_default_transform is None
        or mapping is None
    ):
        raise ImportError(
            "Geospatial preprocessing requires rasterio and shapely to be installed."
        )


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def utm_crs_from_centroid(lon: float, lat: float) -> CRS:
    """
    Derive the UTM CRS for an arbitrary lon/lat centroid.

    Never hardcodes a CRS — always computed from coordinates.
    """
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": hemisphere == "south"})


# ---------------------------------------------------------------------------
# Mosaicking
# ---------------------------------------------------------------------------

def mosaic_tiles(
    tile_paths: list,
    target_crs: CRS,
) -> tuple:
    """
    Reproject (if needed) and merge multiple single-scene tile GeoTIFFs.

    Parameters
    ----------
    tile_paths : list[Path]
        Paths to the raster files to merge. Must have len > 1.
    target_crs : CRS
        Target CRS for reprojection (typically UTM from centroid).

    Returns
    -------
    (data_array, transform, crs) where data_array has shape (bands, H, W).
    """
    if len(tile_paths) <= 1:
        raise ValueError("mosaic_tiles requires at least 2 tile paths.")
    _require_geospatial_dependencies()
    assert rasterio is not None
    assert MemoryFile is not None
    assert calculate_default_transform is not None
    assert reproject is not None
    assert Resampling is not None
    assert rio_merge is not None

    datasets = []
    memory_files = []  # keep references alive

    for path in tile_paths:
        src = rasterio.open(path)
        if CRS(src.crs) != target_crs:
            # Reproject into target CRS at 30 m
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds, resolution=30
            )
            if width is None or height is None:
                raise ValueError("Reprojection returned invalid output dimensions.")
            width = int(width)
            height = int(height)
            band_dtype = np.dtype(src.dtypes[0])
            profile = src.profile.copy()
            profile.update(
                crs=target_crs,
                transform=transform,
                width=width,
                height=height,
            )
            buf = np.zeros((src.count, height, width), dtype=band_dtype)
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=buf,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
            mf = MemoryFile()
            with mf.open(**profile) as dst:
                dst.write(buf)
            memory_files.append(mf)
            datasets.append(mf.open())
            src.close()
        else:
            datasets.append(src)

    merged_data, merged_transform = rio_merge(datasets, method="first")
    merged_crs = target_crs

    for ds in datasets:
        ds.close()
    for mf in memory_files:
        mf.close()

    return merged_data, merged_transform, merged_crs


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip_to_polygon(
    data: np.ndarray,
    transform,
    crs,
    polygon_geom,
    save_path: Optional[Path] = None,
) -> tuple:
    """
    Clip a data array to a polygon boundary.

    Parameters
    ----------
    data : np.ndarray shape (bands, H, W)
    transform : affine.Affine
    crs : CRS
    polygon_geom : shapely geometry
    save_path : Path, optional
        If given, saves the clipped raster as a GeoTIFF here.

    Returns
    -------
    (clipped_data, clipped_transform)
    """
    _require_geospatial_dependencies()
    assert rasterio is not None
    assert MemoryFile is not None
    assert rio_mask is not None
    assert mapping is not None

    profile = {
        "driver": "GTiff",
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "width": data.shape[2],
        "height": data.shape[1],
    }

    with MemoryFile() as mf:
        with mf.open(**profile) as dst:
            dst.write(data)
        with mf.open() as src:
            geom = polygon_geom
            if src.crs:
                project = Transformer.from_crs(
                    "EPSG:4326", src.crs, always_xy=True
                ).transform
                geom = shapely_transform(project, polygon_geom)
            geom_json = [mapping(geom)]
            clipped, clipped_transform = rio_mask(
                src, geom_json, crop=True, filled=True, nodata=0
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        clip_profile = profile.copy()
        clip_profile.update(
            transform=clipped_transform,
            width=clipped.shape[2],
            height=clipped.shape[1],
        )
        with rasterio.open(save_path, "w", **clip_profile) as dst:
            dst.write(clipped)

    return clipped, clipped_transform


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def assert_pixel_alignment(
    before_transform,
    before_shape: tuple,
    after_transform,
    after_shape: tuple,
):
    """
    Hard stop if before/after rasters are not pixel-aligned.

    Parameters
    ----------
    before_transform, after_transform : affine.Affine
    before_shape, after_shape : (H, W) or (bands, H, W)

    Raises
    ------
    ValueError if transforms or spatial shapes differ.
    """
    # Normalise to (H, W)
    def _hw(shape):
        return shape[-2], shape[-1]

    if before_transform != after_transform:
        raise ValueError(
            f"Pixel alignment failure: transforms differ.\n"
            f"  before: {before_transform}\n"
            f"  after:  {after_transform}"
        )
    if _hw(before_shape) != _hw(after_shape):
        raise ValueError(
            f"Pixel alignment failure: shapes differ.\n"
            f"  before: {_hw(before_shape)}\n"
            f"  after:  {_hw(after_shape)}"
        )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_bands(
    data: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-band min-max normalise to [0, 1].

    Parameters
    ----------
    data : np.ndarray shape (6, H, W)  — exactly 6 spectral bands required.

    invalid_mask : np.ndarray shape (H, W), optional
        True where pixels should be excluded from min/max scaling.

    Returns
    -------
    np.ndarray float32 shape (6, H, W)

    Raises
    ------
    ValueError if band count != 6.
    """
    if data.shape[0] != 6:
        raise ValueError(
            f"normalize_bands expects exactly 6 bands, got {data.shape[0]}."
        )
    out = np.empty_like(data, dtype=np.float32)
    for i in range(6):
        band = data[i].astype(np.float32)
        if invalid_mask is not None:
            valid = band[~invalid_mask]
        else:
            valid = band.reshape(-1)

        if valid.size == 0:
            out[i] = np.zeros_like(band)
            continue

        bmin, bmax = valid.min(), valid.max()
        if bmax == bmin:
            out[i] = np.zeros_like(band)
        else:
            out[i] = (band - bmin) / (bmax - bmin)
    return out


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def generate_patches(
    before: np.ndarray,
    after: np.ndarray,
    patch_size: int,
) -> list:
    """
    Slice before/after arrays into spatially aligned square patches.

    Parameters
    ----------
    before, after : np.ndarray shape (6, H, W)
    patch_size : int

    Returns
    -------
    list of dicts: [{"before": ndarray, "after": ndarray, "row": int, "col": int}, ...]
    Each patch is (6, patch_size, patch_size); edge patches are zero-padded.
    """
    _, H, W = before.shape
    patches = []

    row = 0
    while row < H:
        col = 0
        while col < W:
            b_patch = np.zeros((before.shape[0], patch_size, patch_size), dtype=before.dtype)
            a_patch = np.zeros((after.shape[0], patch_size, patch_size), dtype=after.dtype)

            r_end = min(row + patch_size, H)
            c_end = min(col + patch_size, W)
            ph = r_end - row
            pw = c_end - col

            b_patch[:, :ph, :pw] = before[:, row:r_end, col:c_end]
            a_patch[:, :ph, :pw] = after[:, row:r_end, col:c_end]

            patches.append({
                "before": b_patch,
                "after": a_patch,
                "row": row,
                "col": col,
            })
            col += patch_size
        row += patch_size

    return patches
