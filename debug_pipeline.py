from configs.settings import settings
from src.tile_detection import load_tile_grid, find_covering_tiles
from src.lpdaac import search_scenes, select_top_scenes
import json

# Load the test polygon
with open("tests/fixtures/test_polygon.geojson") as f:
    gj = json.load(f)

from shapely.geometry import shape

geom = shape(gj["features"][0]["geometry"])

tile_grid = load_tile_grid(settings.MGRS_GRID_PATH)
tile_ids = find_covering_tiles(geom, tile_grid)
print("Covering tiles:", tile_ids)

for tile in tile_grid:
    if tile["tile_id"] not in tile_ids:
        continue
    print(f"\nSearching tile {tile['tile_id']} bbox {tile['bbox']} year 2020...")
    scenes = search_scenes(
        tile["tile_id"], 2020, settings.CLOUD_COVER_THRESHOLD, bbox=tile["bbox"]
    )
    print(f"  Found {len(scenes)} scenes")
    if scenes:
        print(f"  First scene: {scenes[0]['granule_id']}")
        print(
            f"  URLs ({len(scenes[0]['download_urls'])}): {scenes[0]['download_urls'][:2]}"
        )


url = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T44PKA.2020020T050426.v2.0/HLS.L30.T44PKA.2020020T050426.v2.0.B05.tif"
for band in settings.HLS_BANDS:
    matched = f".{band}." in url or f"_{band}_" in url or url.endswith(f".{band}.tif")
    print(f"{band}: {matched}")
top = select_top_scenes(scenes, 1)
print("\nBand URLs for top scene:")
urls = top[0]["download_urls"]
all_bands = list(settings.HLS_BANDS) + ["B05"]
band_urls = [u for u in urls if any(b in u.split("/")[-1] for b in all_bands)]
print(f"  Matched {len(band_urls)} of {len(urls)} URLs")
for u in band_urls:
    print(" ", u.split("/")[-1])

from src.lpdaac import download_scene, validate_download

print("\nSearching for first downloadable scene...")
# Use the scenes from the last tile search
all_scenes = search_scenes(
    "T43PHR", 2020, settings.CLOUD_COVER_THRESHOLD, bbox=(78.0, 13.0, 79.0, 14.0)
)
print(f"Total scenes to try: {len(all_scenes)}")

for i, scene in enumerate(all_scenes[:10]):  # try first 10
    print(f"\nTrying scene {i+1}: {scene['granule_id']}")
    try:
        cache_path = download_scene(
            scene["granule_id"],
            scene["download_urls"],
            scene["actual_tile_id"],
            2020,
            settings.CACHE_DIR,
        )
        valid = validate_download(cache_path)
        print(f"  Downloaded to: {cache_path}")
        print(f"  Files: {[f.name for f in cache_path.iterdir()]}")
        print(f"  Valid: {valid}")
        if valid:
            print("  ✓ SUCCESS — found a complete scene, stopping search.")
            break
    except RuntimeError as e:
        print(f"  Skipping: {e}")

import requests
from configs.settings import settings

username = settings.EARTHDATA_USERNAME
password = settings.EARTHDATA_PASSWORD
print(f"Username: '{username}'")
print(f"Password set: {bool(password)}")

url = f"https://urs.earthdata.nasa.gov/api/users/{username}/tokens"
resp = requests.get(url, auth=(username, password), timeout=30)
print(f"Token endpoint status: {resp.status_code}")
print(f"Response text: {resp.text[:300]}")

import base64, requests

username = settings.EARTHDATA_USERNAME
password = settings.EARTHDATA_PASSWORD
creds = base64.b64encode(f"{username}:{password}".encode()).decode()
url = f"https://urs.earthdata.nasa.gov/api/users/{username}/tokens"
resp = requests.get(url, headers={"Authorization": f"Basic {creds}"}, timeout=30)
print(f"Token status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"Body: {resp.text[:200]}")

from src.lpdaac import _get_earthdata_token

try:
    token = _get_earthdata_token(
        settings.EARTHDATA_USERNAME, settings.EARTHDATA_PASSWORD
    )
    print(f"Token obtained: {token[:20]}...")
except RuntimeError as e:
    print(f"Token failed:\n{e}")
from src.lpdaac import _get_earthdata_token

try:
    tok = _get_earthdata_token()
    print("Token OK:", tok[:20], "...")
except RuntimeError as e:
    print("Token FAILED:", e)
