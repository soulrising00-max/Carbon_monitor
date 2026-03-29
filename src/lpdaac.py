"""
LPDAAC scene discovery and HLS tile download module.
Handles CMR search, scene selection, download with retry, and cache validation.

CMR search strategy: bounding_box parameter (producer_granule_id wildcards
are unreliable across CMR deployments). The tile_id argument is used only
for cache path organisation and result labelling.
"""

import time
import logging
import netrc
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.auth import HTTPBasicAuth
import rasterio

from configs.settings import settings
from src.tile_detection import load_tile_grid

logger = logging.getLogger(__name__)

_TOKEN_CACHE: dict = {}

# Band mapping for Landsat 8/9 (L30) vs Sentinel-2 (S30).
# settings.HLS_BANDS uses Sentinel-2 names; L30 files use different filenames.
_L30_BAND_MAP: dict[str, str] = {"B8A": "B05", "B11": "B06", "B12": "B07"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _EarthdataSession(requests.Session):
    """
    requests.Session that re-attaches Basic auth on every redirect back to
    urs.earthdata.nasa.gov — mirrors NASA's official Python example exactly.

    Without this, requests drops the Authorization header on cross-domain
    redirects, causing the auth server to return an HTML login page.
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self._edl_auth = (username, password)
        self.auth = HTTPBasicAuth(username, password)

    def rebuild_auth(self, prepared_request, response):
        """Keep auth header when redirecting through the EDL auth host."""
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            original = urlparse(response.request.url)
            redirect = urlparse(url)
            # Drop auth only when neither endpoint is the EDL auth host
            if (
                original.hostname != redirect.hostname
                and redirect.hostname != self.AUTH_HOST
                and original.hostname != self.AUTH_HOST
            ):
                del headers["Authorization"]


def _get_download_session() -> _EarthdataSession:
    """Return a cached session that handles Earthdata auth redirects."""
    if "session" in _TOKEN_CACHE:
        return _TOKEN_CACHE["session"]

    username = settings.EARTHDATA_USERNAME.strip()
    password = settings.EARTHDATA_PASSWORD.strip()

    # Fallback to ~/.netrc if .env credentials are empty
    if not username or not password:
        try:
            rc = netrc.netrc()
            creds = rc.authenticators("urs.earthdata.nasa.gov")
            if creds:
                username, _, password = creds
        except Exception:
            pass

    if not username or not password:
        raise RuntimeError(
            "No Earthdata credentials found. Set EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD in your .env file, or create a ~/.netrc entry "
            "for urs.earthdata.nasa.gov"
        )

    session = _EarthdataSession(username, password)
    _TOKEN_CACHE["session"] = session
    logger.info("Earthdata session created for user '%s'", username)
    return session


def _sensor_bands(granule_id: str) -> list[str]:
    """
    Return the correct band filenames for this granule.
    L30 (Landsat 8/9) uses different band names than S30 (Sentinel-2).
    """
    if ".L30." in granule_id:
        return [_L30_BAND_MAP.get(b, b) for b in settings.HLS_BANDS]
    return list(settings.HLS_BANDS)


def _expected_band_filenames(granule_id: str) -> set[str]:
    """Return the exact GeoTIFF filenames expected for a granule download."""
    base_id = granule_id[:-5] if granule_id.endswith(".v2.0") else granule_id
    return {f"{base_id}.v2.0.{band}.tif" for band in _sensor_bands(granule_id)}


def _bbox_from_tile(
    tile_grid: list[dict], tile_id: str
) -> tuple[float, float, float, float] | None:
    """Return (min_lon, min_lat, max_lon, max_lat) for a tile_id, or None."""
    for tile in tile_grid:
        if tile["tile_id"] == tile_id:
            return tile["bbox"]
    return None


def _cmr_search_bbox(short_name: str, bbox: tuple, year: int) -> list[dict]:
    """
    Query CMR using a bounding box for a single product short_name.
    bbox = (min_lon, min_lat, max_lon, max_lat)
    Returns raw entry list; empty list on any error.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name={short_name}"
        f"&temporal[]={year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
        f"&bounding_box={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&page_size=100"
        f"&sort_key[]=cloud_cover"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json().get("feed", {}).get("entry", [])
    except Exception as exc:
        logger.warning(
            "CMR search failed for %s / %s / %d: %s", short_name, bbox, year, exc
        )
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_scenes(
    tile_id: str,
    year: int,
    cloud_max: float,
    bbox: tuple[float, float, float, float] | None = None,
    restrict_to_tile: bool = True,
) -> list[dict]:
    """
    Search CMR for HLS granules covering *tile_id* in *year* with cloud
    cover <= cloud_max (0-1 fraction).

    bbox: (min_lon, min_lat, max_lon, max_lat). If None, looks up tile_id
    in the loaded tile grid automatically.

    Returns a list sorted ascending by cloud cover:
        [{"granule_id": str, "cloud_cover": float, "download_urls": list[str],
          "actual_tile_id": str, "sensor": str, "band_filenames": list[str]}, ...]
    Returns an empty list if nothing found — never raises.
    """
    if bbox is None:
        grid = load_tile_grid(settings.MGRS_GRID_PATH)
        bbox = _bbox_from_tile(grid, tile_id)
        if bbox is None:
            logger.warning("tile_id %s not found in tile grid — cannot search", tile_id)
            return []

    entries_l30 = _cmr_search_bbox("HLSL30", bbox, year)
    entries_s30 = _cmr_search_bbox("HLSS30", bbox, year)

    raw_entries = entries_l30 + entries_s30
    print("Scenes found before filtering:", len(raw_entries))

    seen: set[str] = set()
    filtered_scenes: list[dict] = []

    for entry in raw_entries:
        gid = entry.get("producer_granule_id") or entry.get("id", "")
        if not gid or gid in seen:
            continue
        seen.add(gid)

        raw_cc = entry.get("cloud_cover")
        try:
            cloud_pct = float(raw_cc) if raw_cc is not None else 100.0
        except (TypeError, ValueError):
            cloud_pct = 100.0

        if cloud_pct > cloud_max * 100:
            continue

        # Extract the actual HLS tile ID from the granule ID
        # Format: HLS.L30.T43PHR.2020004T050430.v2.0
        parts = gid.split(".")
        actual_tile = parts[2] if len(parts) >= 3 else tile_id
        if restrict_to_tile and actual_tile != tile_id:
            # Bounding-box queries can spill into adjacent tiles. The pipeline
            # expects scenes for the specific MGRS tile it is iterating over.
            continue

        is_l30 = ".L30." in gid
        product = "HLSL30.020" if is_l30 else "HLSS30.020"
        base_url = (
            f"https://data.lpdaac.earthdatacloud.nasa.gov"
            f"/lp-prod-protected/{product}/{gid}.v2.0"
        )

        # Use sensor-correct band names for URL construction
        bands = _sensor_bands(gid)
        urls = [f"{base_url}/{gid}.v2.0.{band}.tif" for band in bands]

        filtered_scenes.append(
            {
                "granule_id": gid,
                "cloud_cover": cloud_pct,
                "download_urls": urls,
                "actual_tile_id": actual_tile,
                "sensor": "L30" if is_l30 else "S30",
                "band_filenames": bands,
            }
        )

    print("Scenes after filtering:", len(filtered_scenes))
    filtered_scenes.sort(key=lambda x: x["cloud_cover"])
    return filtered_scenes


def select_top_scenes(scene_list: list[dict], n: int) -> list[dict]:
    """
    Return the top *n* scenes (already sorted ascending by cloud cover).
    If fewer than *n* available, return all. Never raises.
    """
    return scene_list[:n]


def download_scene(
    granule_id: str,
    download_urls: list[str],
    tile_id: str,
    year: int,
    cache_dir: Path,
) -> Path:
    """
    Download the 7 required HLS bands for *granule_id* into the local cache.
    Cache layout: cache_dir / tile_id / str(year) / granule_id /
    Skips download if all band files are already present.
    Retries up to 3 times with 5-second backoff.
    Raises RuntimeError after 3 failed attempts or on a 404 (missing band).
    """
    cache_path = cache_dir / tile_id / str(year) / granule_id
    cache_path.mkdir(parents=True, exist_ok=True)

    # Use sensor-correct band names to filter the URL list.
    # download_urls already contain correct filenames (built in search_scenes),
    # but we re-derive bands here so download_scene works when called standalone.
    bands = _sensor_bands(granule_id)
    band_urls = [
        url for url in download_urls if any(url.endswith(f".{b}.tif") for b in bands)
    ]

    # Cache-hit check: all expected filenames already on disk?
    needed_names = _expected_band_filenames(granule_id)
    if cache_path.exists():
        existing_tifs = [
            f for f in cache_path.iterdir() if f.is_file() and f.suffix == ".tif"
        ]
        existing_names = {f.name for f in existing_tifs}
        if needed_names and needed_names.issubset(existing_names):
            logger.info("Cache hit for %s — skipping download.", granule_id)
            return cache_path

        if existing_tifs:
            # Earlier runs may have left partial or wrong-band files behind.
            # Start this granule fresh so reruns don't keep mixing cache states.
            logger.info(
                "Clearing incomplete cache for %s (found %d tif files, need %d).",
                granule_id,
                len(existing_tifs),
                len(needed_names),
            )
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)

    # Use the session (handles EDL redirects correctly).
    # Previously this used bare requests.get(url, headers=headers, ...) where
    # `headers` was never defined in this scope — a guaranteed NameError.
    session = _get_download_session()

    for url in band_urls:
        filename = url.split("/")[-1].split("?")[0]
        dest = cache_path / filename
        if dest.exists():
            continue

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                with session.get(url, stream=True, timeout=120) as r:
                    if r.status_code == 404:
                        raise RuntimeError(
                            f"Granule {granule_id} is missing band file {filename} "
                            f"(404) — granule is incomplete, skipping."
                        )
                    r.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                logger.info("Downloaded %s (attempt %d)", filename, attempt)
                last_exc = None
                break
            except RuntimeError:
                raise  # propagate 404 immediately — no retry
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d failed for %s: %s", attempt, filename, exc)
                if attempt < 3:
                    time.sleep(5)

        if last_exc is not None:
            raise RuntimeError(
                f"Failed to download {filename} for granule {granule_id} "
                f"after 3 attempts: {last_exc}"
            )

    return cache_path


def validate_download(cache_path: Path) -> bool:
    """
    Return True only if cache_path exists, contains exactly 7 .tif files,
    and every file opens cleanly with rasterio.
    """
    if not cache_path or not cache_path.exists():
        return False

    files = [f for f in cache_path.iterdir() if f.is_file() and f.suffix == ".tif"]
    if len(files) < 7:
        logger.warning(
            "validate_download: only %d .tif files in %s", len(files), cache_path
        )
        return False

    granule_id = cache_path.name
    expected_names = _expected_band_filenames(granule_id)
    present_names = {f.name for f in files}
    if not expected_names.issubset(present_names):
        missing = sorted(expected_names - present_names)
        logger.warning(
            "validate_download: missing expected files for %s: %s",
            cache_path,
            missing,
        )
        return False

    for f in files:
        try:
            with rasterio.open(f):
                pass
        except Exception as exc:
            logger.warning("validate_download: rasterio failed on %s: %s", f, exc)
            return False

    return True
