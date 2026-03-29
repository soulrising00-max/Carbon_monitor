"""
download_training_data.py
=========================
Standalone script to download curated HLS training scenes for the
ForestUNet model. Designed for multi-biome coverage so the model
generalises globally, not just over Karnataka.

USAGE
-----
    python download_training_data.py

    # Dry-run: see what would be downloaded without touching disk
    python download_training_data.py --dry-run

    # Override output directory
    python download_training_data.py --cache-dir /path/to/cache

    # Skip specific tiles if they're already good
    python download_training_data.py --skip T43PHR T18NXF

WHAT IT DOES
------------
1. For each of 6 curated tiles (2 per biome), searches CMR for the
   single best scene in 2018 AND 2022 (4-year gap = strong deforestation
   signal for training labels).
2. Applies a 20% cloud-cover filter at scene level (strict enough for
   quality, lenient enough that tropical tiles are not skipped).
3. Downloads only the 7 required band files per scene (6 spectral + Fmask).
4. Validates every download with rasterio before marking it complete.
5. Writes a manifest.json at the end listing every tile, its biome,
   its before/after scene paths, and scene-level cloud cover.
   The Kaggle training notebook reads this manifest to loop over tiles
   instead of being hardcoded to a single tile.

OUTPUT STRUCTURE
----------------
    <cache_dir>/
        T18NXF/
            2018/<granule_id>/   ← 7 .tif files
            2022/<granule_id>/   ← 7 .tif files
        T36MZE/
            2018/...
            2022/...
        ...
        training_manifest.json

REQUIREMENTS
------------
- EARTHDATA_USERNAME and EARTHDATA_PASSWORD in your .env file
  (or in ~/.netrc for urs.earthdata.nasa.gov)
- pip packages: requests, rasterio, python-dotenv
- Disk space: ~6–12 GB total (1 scene per tile per year × 6 tiles × ~700 MB each)

NOTES ON TILE SELECTION
-----------------------
Tile IDs and bounding boxes were validated against live CMR responses using
diagnose_tiles.py. All 6 tiles confirmed to have clean scenes (≤30% cloud)
for both 2018 and 2022.

Tropical  (East Africa — Amazon dropped due to persistent cloud cover):
  T36MZB  Kenya/Tanzania  — 2% cloud confirmed, strong woodland-loss signal
  T37MBS  Tanzania        — 5% cloud confirmed, different forest type

Subtropical:
  T43PHR  Karnataka, India        — 0% cloud confirmed, agricultural expansion
  T44QLH  Andhra Pradesh, India   — 0-14% cloud confirmed, different land use

Temperate:
  T55HGD  New South Wales, AU  — 2% S30 cloud confirmed, eucalypt forest change
  T32UPU  Bavaria, Germany     — 4% S30 cloud confirmed, spruce dieback 2018-2022
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the repo root is on sys.path so we can import
# src.lpdaac and configs.settings regardless of where the script is run from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env before importing settings (settings reads from .env on import)
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional; credentials can come from env vars directly

from configs.settings import settings          # noqa: E402  (after sys.path fix)
from src.lpdaac import (                       # noqa: E402
    search_scenes,
    select_top_scenes,
    download_scene,
    validate_download,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_training_data")

# ---------------------------------------------------------------------------
# Curated tile catalogue
# ---------------------------------------------------------------------------
# Each entry: (tile_id, biome, approx_bbox, hansen_tile, description)
# bbox = (min_lon, min_lat, max_lon, max_lat) — used directly for CMR search
# so we don't depend on mgrs_tile_grid.geojson having every tile.
# hansen_tile is the GFC tile name the Kaggle notebook needs to download labels.

TRAINING_TILES = [
    # --- TROPICAL ---
    # Both tiles confirmed via diagnose_tiles.py: S30 scenes at 1-5% cloud.
    # Amazon dropped — persistent 74-100% cloud in 2018 and 2022.
    {
        "tile_id":     "T36MZB",
        "biome":       "tropical",
        # Bbox shifted west/south from original T36MZE bbox to centre on T36MZB.
        # Confirmed: S30 returns T36MZB at 2% cloud for this bbox.
        "bbox":        (35.5, -4.5, 36.5, -3.5),
        "hansen_tile": "00N_040E",
        "description": "Kenya/Tanzania — confirmed 2% cloud, strong woodland loss",
    },
    {
        "tile_id":     "T37MBS",
        "biome":       "tropical",
        # Bbox centred one UTM zone east of T36MZB — confirmed via diagnostic.
        # S30 returns T37MBS at 5% cloud.
        "bbox":        (36.5, -4.5, 37.5, -3.5),
        "hansen_tile": "00N_040E",
        "description": "Tanzania — confirmed 5% cloud, different forest type from T36MZB",
    },

    # --- SUBTROPICAL ---
    # T43PHR confirmed working (0% cloud Landsat). Bbox slightly widened to
    # ensure T43PHR appears first in CMR results, not adjacent tiles.
    {
        "tile_id":     "T43PHR",
        "biome":       "subtropical",
        "bbox":        (78.2, 14.1, 78.9, 14.9),
        "hansen_tile": "20N_080E",
        "description": "Karnataka, India — confirmed 0% cloud, agricultural expansion",
    },
    {
        "tile_id":     "T44QLH",
        "biome":       "subtropical",
        # Diagnostic returned T44QLH at 1-14% cloud for this bbox.
        # Original T44QKD (79-80 lon, 20-21 lat) was returning wrong tiles.
        "bbox":        (79.5, 19.5, 80.5, 20.5),
        "hansen_tile": "20N_080E",
        "description": "Andhra Pradesh, India — confirmed 1-14% cloud, different land use",
    },

    # --- TEMPERATE ---
    # T55HFD confirmed at 0-8% cloud (excellent). T30UVC kept but cloudy —
    # script will warn if no scenes pass filter and suggest T32UNU (Germany).
    {
        "tile_id":     "T55HGD",
        "biome":       "temperate",
        # T55HFC does not exist in the HLS catalogue (wildcard search returned
        # zero results). T55HGD appeared at 2% S30 cloud in the first diagnostic
        # run for this region. Using the original bbox from that run.
        "bbox":        (149.0, -34.0, 150.0, -33.0),
        "hansen_tile": "30S_150E",
        "description": "New South Wales, AU — T55HGD confirmed 2% cloud S30, eucalypt change",
    },
    {
        "tile_id":     "T32UPU",
        "biome":       "temperate",
        # Diagnostic: S30 T32UPU at 4% cloud for Bavaria bbox.
        # T32UNU was the intended tile but bbox keeps landing on T32UPU instead.
        # T32UPU covers the same Bavarian Forest / spruce dieback region.
        "bbox":        (11.8, 48.2, 12.8, 49.2),
        "hansen_tile": "50N_010E",
        "description": "Bavaria, Germany — T32UPU confirmed 4% cloud, spruce dieback signal",
    },
]

# Year pair: 4-year gap gives strong deforestation signal for Hansen labels.
YEAR_BEFORE = 2018
YEAR_AFTER  = 2022

# Scene-level cloud filter: strict enough for quality, lenient enough that
# tropical tiles are not skipped. The Kaggle notebook applies a second
# patch-level filter (30%) to remove remaining cloudy pixels.
CLOUD_MAX_FRACTION = 0.20   # 20%

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_best_scene(tile: dict, year: int, dry_run: bool) -> dict | None:
    """
    Search CMR for the single least-cloudy scene for tile/year.
    Returns the scene dict, or None if nothing acceptable found.
    """
    tile_id = tile["tile_id"]
    bbox    = tile["bbox"]

    logger.info("  Searching CMR: %s  year=%d  cloud_max=%.0f%%",
                tile_id, year, CLOUD_MAX_FRACTION * 100)

    scenes = search_scenes(
        tile_id=tile_id,
        year=year,
        cloud_max=CLOUD_MAX_FRACTION,
        bbox=bbox,
        restrict_to_tile=True,
    )

    if not scenes:
        logger.warning("  ✗ No scenes found for %s / %d within %.0f%% cloud cover.",
                       tile_id, year, CLOUD_MAX_FRACTION * 100)
        return None

    # search_scenes already sorts ascending by cloud cover; take the best one.
    best = select_top_scenes(scenes, n=1)[0]
    logger.info("  ✓ Best scene: %s  cloud=%.1f%%",
                best["granule_id"], best["cloud_cover"])

    if dry_run:
        logger.info("  [dry-run] Would download %d URLs.", len(best["download_urls"]))

    return best


def _download_and_validate(
    scene: dict,
    tile_id: str,
    year: int,
    cache_dir: Path,
    dry_run: bool,
) -> Path | None:
    """
    Download scene to cache and validate. Returns cache_path or None on failure.
    """
    granule_id    = scene["granule_id"]
    download_urls = scene["download_urls"]

    if dry_run:
        cache_path = cache_dir / tile_id / str(year) / granule_id
        logger.info("  [dry-run] Cache path would be: %s", cache_path)
        return cache_path

    logger.info("  Downloading %s ...", granule_id)
    try:
        cache_path = download_scene(
            granule_id=granule_id,
            download_urls=download_urls,
            tile_id=tile_id,
            year=year,
            cache_dir=cache_dir,
        )
    except RuntimeError as exc:
        logger.error("  ✗ Download failed for %s: %s", granule_id, exc)
        return None

    logger.info("  Validating %s ...", cache_path)
    if not validate_download(cache_path):
        logger.error("  ✗ Validation failed for %s", cache_path)
        return None

    logger.info("  ✓ Download complete and validated: %s", cache_path)
    return cache_path


def run(cache_dir: Path, skip_tiles: list[str], dry_run: bool) -> None:
    """
    Main download loop. Iterates over all training tiles and years.
    Writes training_manifest.json at the end.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "year_before": YEAR_BEFORE,
        "year_after":  YEAR_AFTER,
        "cloud_max_fraction": CLOUD_MAX_FRACTION,
        "tiles": [],
    }

    total_tiles   = len(TRAINING_TILES)
    success_tiles = 0
    failed_tiles  = []

    for idx, tile in enumerate(TRAINING_TILES, start=1):
        tile_id = tile["tile_id"]
        biome   = tile["biome"]

        logger.info("")
        logger.info("=" * 60)
        logger.info("Tile %d/%d  |  %s  |  biome=%s", idx, total_tiles, tile_id, biome)
        logger.info("  %s", tile["description"])
        logger.info("=" * 60)

        if tile_id in skip_tiles:
            logger.info("  Skipping (--skip flag set).")
            continue

        tile_result = {
            "tile_id":        tile_id,
            "biome":          tile["biome"],
            "hansen_tile":    tile["hansen_tile"],
            "description":    tile["description"],
            "bbox":           tile["bbox"],
            "before": None,
            "after":  None,
            "status": "pending",
        }

        # --- BEFORE year ---
        logger.info("Before year (%d):", YEAR_BEFORE)
        before_scene = _find_best_scene(tile, YEAR_BEFORE, dry_run)
        if before_scene:
            before_path = _download_and_validate(
                before_scene, tile_id, YEAR_BEFORE, cache_dir, dry_run
            )
            if before_path:
                tile_result["before"] = {
                    "granule_id":    before_scene["granule_id"],
                    "cloud_cover":   before_scene["cloud_cover"],
                    "sensor":        before_scene.get("sensor", "unknown"),
                    "cache_path":    str(before_path),
                }

        # Small pause between years to be a polite API citizen
        time.sleep(2)

        # --- AFTER year ---
        logger.info("After year (%d):", YEAR_AFTER)
        after_scene = _find_best_scene(tile, YEAR_AFTER, dry_run)
        if after_scene:
            after_path = _download_and_validate(
                after_scene, tile_id, YEAR_AFTER, cache_dir, dry_run
            )
            if after_path:
                tile_result["after"] = {
                    "granule_id":    after_scene["granule_id"],
                    "cloud_cover":   after_scene["cloud_cover"],
                    "sensor":        after_scene.get("sensor", "unknown"),
                    "cache_path":    str(after_path),
                }

        # Mark tile complete only if BOTH years succeeded
        if tile_result["before"] and tile_result["after"]:
            tile_result["status"] = "complete"
            success_tiles += 1
            logger.info("✓ Tile %s complete (both years).", tile_id)
        elif tile_result["before"] or tile_result["after"]:
            tile_result["status"] = "partial"
            failed_tiles.append(tile_id)
            logger.warning("⚠ Tile %s partial — only one year succeeded.", tile_id)
        else:
            tile_result["status"] = "failed"
            failed_tiles.append(tile_id)
            logger.error("✗ Tile %s failed — no scenes for either year.", tile_id)

        manifest["tiles"].append(tile_result)

        # Pause between tiles: polite to CMR and LPDAAC servers
        if idx < total_tiles:
            time.sleep(3)

    # --- Write manifest ---
    manifest_path = cache_dir / "training_manifest.json"
    if not dry_run:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info("")
        logger.info("Manifest written: %s", manifest_path)
    else:
        logger.info("")
        logger.info("[dry-run] Manifest would be written to: %s", manifest_path)
        logger.info("[dry-run] Manifest preview:")
        print(json.dumps(manifest, indent=2))

    # --- Summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total tiles attempted : %d", total_tiles - len(skip_tiles))
    logger.info("  Complete (both years) : %d", success_tiles)
    logger.info("  Failed / partial      : %d  %s",
                len(failed_tiles),
                failed_tiles if failed_tiles else "")
    logger.info("=" * 60)

    if failed_tiles:
        logger.warning(
            "\nSome tiles failed. Common causes:\n"
            "  1. Cloud cover > %.0f%% for all available scenes in that year\n"
            "     → Try relaxing CLOUD_MAX_FRACTION to 0.30 in this script\n"
            "  2. Earthdata credentials missing or expired\n"
            "     → Check EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env\n"
            "  3. Tile has no HLS coverage for 2018 (HLS.S30 starts mid-2015,\n"
            "     HLS.L30 starts April 2013 — both should cover 2018)\n"
            "  4. Network timeout — just re-run the script; cache-hit logic\n"
            "     skips anything already downloaded correctly\n",
            CLOUD_MAX_FRACTION * 100,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download curated HLS training scenes for ForestUNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.CACHE_DIR,
        help="Root directory for downloaded scenes. "
             f"Default: {settings.CACHE_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search CMR and print what would be downloaded — no files written.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="TILE_ID",
        help="Tile IDs to skip (e.g. --skip T43PHR T18NXF).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    logger.info("Cache directory : %s", args.cache_dir)
    logger.info("Year pair       : %d → %d", YEAR_BEFORE, YEAR_AFTER)
    logger.info("Cloud max       : %.0f%%", CLOUD_MAX_FRACTION * 100)
    logger.info("Tiles to skip   : %s", args.skip or "none")
    logger.info("Tiles planned   : %s",
                [t["tile_id"] for t in TRAINING_TILES if t["tile_id"] not in args.skip])

    run(
        cache_dir=args.cache_dir,
        skip_tiles=args.skip,
        dry_run=args.dry_run,
    )
