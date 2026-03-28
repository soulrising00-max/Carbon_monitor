import base64, requests
from configs.settings import settings

u = settings.EARTHDATA_USERNAME
p = settings.EARTHDATA_PASSWORD

print(f"Username: '{u}'")
print(f"Password length: {len(p)}, first char: '{p[0]}', last char: '{p[-1]}'")
print(f"Any whitespace in username: {u != u.strip()}")
print(f"Any whitespace in password: {p != p.strip()}")

# Test 1: GET with requests auth= (most reliable)
resp = requests.get(
    f"https://urs.earthdata.nasa.gov/api/users/{u}/tokens",
    auth=(u, p),
    timeout=30,
)
print(f"\nGET /tokens auth= status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type', 'none')}")
print(f"Body[:300]: {resp.text[:300]}")

# Test 2: POST with requests auth=
resp2 = requests.post(
    "https://urs.earthdata.nasa.gov/api/users/tokens",
    auth=(u, p),
    timeout=30,
)
print(f"\nPOST /tokens auth= status: {resp2.status_code}")
print(f"Content-Type: {resp2.headers.get('Content-Type', 'none')}")
print(f"Body[:300]: {resp2.text[:300]}")
