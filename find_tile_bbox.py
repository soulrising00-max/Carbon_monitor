"""
find_tile_bbox.py
=================
Searches CMR directly by granule ID wildcard to find exactly what spatial
extent CMR associates with a specific MGRS tile — bypasses bbox guessing.

Usage:
    python find_tile_bbox.py T55HFC
    python find_tile_bbox.py T55HFC 2022
"""

import sys
import json
import requests


def find_tile(tile_id: str, year: int = 2022):
    print(f"\nSearching CMR for tile {tile_id} in {year} using granule ID wildcard...")
    print("=" * 60)

    for short_name in ["HLSS30", "HLSL30"]:
        url = (
            f"https://cmr.earthdata.nasa.gov/search/granules.json"
            f"?short_name={short_name}"
            f"&temporal[]={year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
            f"&producer_granule_id=*{tile_id}*"
            f"&page_size=5"
            f"&sort_key=cloud_cover"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            entries = resp.json().get("feed", {}).get("entry", [])
        except Exception as e:
            print(f"  {short_name}: CMR error: {e}")
            continue

        if not entries:
            print(f"  {short_name}: no results")
            continue

        print(f"\n  {short_name}: {len(entries)} results")
        for e in entries:
            gid = e.get("producer_granule_id") or e.get("id", "?")
            cc  = e.get("cloud_cover", "N/A")

            # Extract spatial extent from CMR response
            boxes   = e.get("boxes", [])
            points  = e.get("points", [])
            poly    = e.get("polygons", [])

            print(f"\n    Granule : {gid}")
            print(f"    Cloud   : {cc}%")
            if boxes:
                # CMR box format: "S W N E"
                print(f"    CMR box : {boxes}  (format: S W N E)")
                for b in boxes:
                    parts = b.strip().split()
                    if len(parts) == 4:
                        s, w, n, e_ = parts
                        print(f"    → bbox for script: ({w}, {s}, {e_}, {n})")
                        print(f"      i.e. min_lon={w}, min_lat={s}, max_lon={e_}, max_lat={n}")
            elif points:
                print(f"    Points  : {points}")
            elif poly:
                print(f"    Polygon : {str(poly)[:200]}")
            else:
                print(f"    No spatial info in response")

    print("\n" + "=" * 60)
    print("Copy the bbox line above into TRAINING_TILES for this tile.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No args — check both tiles that needed verification
        for t in ["T55HGD", "T32UPU"]:
            find_tile(t, 2022)
    else:
        tile = sys.argv[1]
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2022
        find_tile(tile, year)
