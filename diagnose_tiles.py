"""
diagnose_tiles.py
=================
Run this BEFORE the main download script to understand why tiles are failing.
It shows the raw CMR granule IDs without any filtering, so you can see
exactly what tile IDs CMR is returning for each bounding box.

Usage:
    python diagnose_tiles.py

No downloads. No credentials needed. Just CMR queries.
"""

import sys
import json
import time
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Same tile list from the download script
# ---------------------------------------------------------------------------
TILES_TO_CHECK = [
    {"tile_id": "T36MZB", "biome": "tropical",    "bbox": ( 35.5, -4.5,  36.5, -3.5)},
    {"tile_id": "T37MBS", "biome": "tropical",    "bbox": ( 36.5, -4.5,  37.5, -3.5)},
    {"tile_id": "T43PHR", "biome": "subtropical", "bbox": ( 78.2, 14.1,  78.9, 14.9)},
    {"tile_id": "T44QLH", "biome": "subtropical", "bbox": ( 79.5, 19.5,  80.5, 20.5)},
    {"tile_id": "T55HGD", "biome": "temperate",   "bbox": (149.0,-34.0, 150.0,-33.0)},
    {"tile_id": "T32UPU", "biome": "temperate",   "bbox": ( 11.8, 48.2,  12.8, 49.2)},
]

YEAR = 2022   # just check one year for diagnosis
CLOUD_DISPLAY_MAX = 50   # show all scenes up to 50% cloud for diagnosis


def raw_cmr_search(short_name: str, bbox: tuple, year: int, page_size: int = 10) -> list:
    min_lon, min_lat, max_lon, max_lat = bbox
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name={short_name}"
        f"&temporal[]={year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
        f"&bounding_box={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&page_size={page_size}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json().get("feed", {}).get("entry", [])
    except Exception as exc:
        print(f"  CMR error: {exc}")
        return []


def diagnose():
    print(f"\nDiagnosing CMR tile coverage for year {YEAR}")
    print("=" * 70)

    for tile in TILES_TO_CHECK:
        tile_id = tile["tile_id"]
        bbox    = tile["bbox"]
        biome   = tile["biome"]

        print(f"\n{tile_id}  ({biome})  bbox={bbox}")
        print("-" * 70)

        for short_name in ["HLSL30", "HLSS30"]:
            entries = raw_cmr_search(short_name, bbox, YEAR, page_size=5)
            if not entries:
                print(f"  {short_name}: no results")
                continue

            print(f"  {short_name}: {len(entries)} results (showing first 5)")
            for e in entries:
                gid       = e.get("producer_granule_id") or e.get("id", "unknown")
                cc        = e.get("cloud_cover", "N/A")
                # Extract tile from granule ID: HLS.S30.T18NXF.2022xxx → T18NXF
                parts     = gid.split(".")
                actual_tile = parts[2] if len(parts) >= 3 else "?"
                match     = "✓ MATCH" if actual_tile == tile_id else f"✗ MISMATCH (got {actual_tile})"
                print(f"    {gid}  cloud={cc}%  {match}")

        time.sleep(1)   # be polite to CMR

    print("\n" + "=" * 70)
    print("WHAT TO LOOK FOR:")
    print("  ✓ MATCH   = granule tile ID matches what we searched for → filtering is correct")
    print("  ✗ MISMATCH = bbox is returning the wrong tile → bbox needs adjusting")
    print("  no results = HLS has no coverage for this region+year (unlikely for 2022)")
    print("\nIf you see MISMATCH: the actual tile IDs in the output are what")
    print("you should use instead of the ones in TRAINING_TILES.")


if __name__ == "__main__":
    diagnose()
