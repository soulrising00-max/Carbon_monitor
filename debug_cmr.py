"""
Debug script — run from repo root:
    python debug_cmr.py
"""

import requests

tile_id = "T43PGP"
year = 2020

for short_name in ("HLSL30", "HLSS30"):
    # Build URL manually (no params dict — avoids encoding issues)
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name={short_name}"
        f"&temporal[]={year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
        f"&producer_granule_id=*{tile_id}*"
        f"&page_size=5"
    )
    print(f"\n=== {short_name} ===")
    print(f"URL: {url}")

    resp = requests.get(url, timeout=30)
    print(f"HTTP status: {resp.status_code}")

    data = resp.json()
    entries = data.get("feed", {}).get("entry", [])
    print(f"Entries returned: {len(entries)}")

    if entries:
        e = entries[0]
        print(f"  granule id  : {e.get('producer_granule_id') or e.get('id')}")
        print(f"  cloud_cover : {e.get('cloud_cover')}")
        print(f"  all keys    : {list(e.keys())}")
        links = e.get("links", [])
        print(f"  link count  : {len(links)}")
        for lnk in links[:5]:
            print(f"    {lnk.get('rel','?'):50s}  {lnk.get('href','')[:80]}")
    else:
        # Show raw response to spot errors
        print("  raw response (first 800 chars):")
        print(resp.text[:800])

    # Also try without the wildcard — maybe CMR is strict
    url2 = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name={short_name}"
        f"&temporal[]={year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
        f"&page_size=2"
    )
    print(f"\n  [sanity] first 2 granules of {short_name} regardless of tile:")
    resp2 = requests.get(url2, timeout=30)
    entries2 = resp2.json().get("feed", {}).get("entry", [])
    print(f"  count: {len(entries2)}")
    for e2 in entries2:
        print(f"    {e2.get('producer_granule_id') or e2.get('id')}")
# Paste this at the bottom of debug_cmr.py and run again

print("\n\n=== TILE ID INVESTIGATION ===")

# 1. What does a real Karnataka-area granule look like?
#    Try a nearby tile that definitely exists — T44NKM is also in our grid
for test_tile in ("T43PGP", "T44NKM", "43PGP", "PGP"):
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name=HLSL30"
        f"&temporal[]=2020-01-01T00:00:00Z,2020-06-30T23:59:59Z"
        f"&producer_granule_id=*{test_tile}*"
        f"&page_size=2"
    )
    resp = requests.get(url, timeout=30)
    entries = resp.json().get("feed", {}).get("entry", [])
    print(f"Filter '*{test_tile}*' → {len(entries)} results", end="")
    if entries:
        print(f"  → {entries[0].get('producer_granule_id') or entries[0].get('id')}")
    else:
        print()

# 2. Try using the 'granule_ur' field instead of producer_granule_id
print()
for test_tile in ("T43PGP", "T44NKM"):
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name=HLSL30"
        f"&temporal[]=2020-01-01T00:00:00Z,2020-06-30T23:59:59Z"
        f"&granule_ur=*{test_tile}*"
        f"&page_size=2"
    )
    resp = requests.get(url, timeout=30)
    entries = resp.json().get("feed", {}).get("entry", [])
    print(f"granule_ur filter '*{test_tile}*' → {len(entries)} results", end="")
    if entries:
        print(f"  → {entries[0].get('producer_granule_id') or entries[0].get('id')}")
    else:
        print()

# 3. Try bounding box search instead — guaranteed to work regardless of ID format
#    T43PGP covers lon 78-79, lat 13-14
print()
url = (
    f"https://cmr.earthdata.nasa.gov/search/granules.json"
    f"?short_name=HLSL30"
    f"&temporal[]=2020-01-01T00:00:00Z,2020-06-30T23:59:59Z"
    f"&bounding_box=78,13,79,14"
    f"&page_size=3"
)
resp = requests.get(url, timeout=30)
entries = resp.json().get("feed", {}).get("entry", [])
print(f"Bounding box 78,13,79,14 → {len(entries)} results")
for e in entries:
    print(
        f"  {e.get('producer_granule_id') or e.get('id')}  cloud={e.get('cloud_cover')}"
    )
print("\n\n=== REAL TILE IDS FOR ALL BBOXES ===")

bboxes = [
    ("T43PGP_area", "78,13,79,14"),
    ("T43PFP_area", "78,12,79,13"),
    ("T44NKM_area", "79,8,80,9"),
    ("T43QGU_area", "78,17,79,18"),
    ("T44QKD_area", "79,20,80,21"),
    ("T18NXF_area", "-77,3,-76,4"),
    ("T19LGK_area", "-76,-5,-75,-4"),
    ("T55HBU_area", "149,-34,150,-33"),
    ("T36MZE_area", "36,-4,37,-3"),
    ("T36NZF_area", "36,-3,37,-2"),
]

for label, bbox in bboxes:
    url = (
        f"https://cmr.earthdata.nasa.gov/search/granules.json"
        f"?short_name=HLSL30"
        f"&temporal[]=2020-06-01T00:00:00Z,2020-06-30T23:59:59Z"
        f"&bounding_box={bbox}"
        f"&page_size=5"
    )
    resp = requests.get(url, timeout=30)
    entries = resp.json().get("feed", {}).get("entry", [])
    tile_ids = set()
    for e in entries:
        gid = e.get("producer_granule_id") or e.get("id", "")
        # Extract tile portion: HLS.L30.T43PHR.2020... → T43PHR
        parts = gid.split(".")
        if len(parts) >= 3:
            tile_ids.add(parts[2])
    print(f"{label:20s} bbox={bbox:20s} → {sorted(tile_ids)}")
