"""
API tests using httpx AsyncClient + pytest-asyncio.
"""

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def test_polygon_geojson():
    path = Path(__file__).parent / "fixtures" / "test_polygon.geojson"
    return json.loads(path.read_text())


@pytest.fixture
def valid_payload(test_polygon_geojson):
    return {
        "geojson": test_polygon_geojson,
        "start_year": 2020,
        "end_year": 2023,
        "annual_offset_tco2": 12500.0,
    }


@pytest.fixture
def invalid_payload():
    """GeoJSON missing 'features' key."""
    return {
        "geojson": {"type": "FeatureCollection"},  # missing features
        "start_year": 2020,
        "end_year": 2023,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_returns_queued(valid_payload):
    async with make_client() as client:
        resp = await client.post("/projects/test-001/analyze", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert "poll_url" in body


@pytest.mark.asyncio
async def test_analyze_accepts_optional_annual_offset(test_polygon_geojson):
    payload = {
        "geojson": test_polygon_geojson,
        "start_year": 2020,
        "end_year": 2023,
        "annual_offset_tco2": 5000.0,
    }
    async with make_client() as client:
        resp = await client.post("/projects/test-offset/analyze", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_get_results_unknown_project():
    async with make_client() as client:
        resp = await client.get("/projects/nonexistent-xyz/results")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_results_immediately_after_post(valid_payload):
    """Status key must exist; value may be 'running' or 'complete'."""
    async with make_client() as client:
        await client.post("/projects/test-002/analyze", json=valid_payload)
        resp = await client.get("/projects/test-002/results")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert body["status"] in ("running", "complete", "failed")


@pytest.mark.asyncio
async def test_invalid_geojson_sets_failed(invalid_payload):
    """Background task should set status to 'failed' for invalid GeoJSON."""
    async with make_client() as client:
        await client.post("/projects/test-003/analyze", json=invalid_payload)
        # Give the background task a moment to run
        await asyncio.sleep(1.0)
        resp = await client.get("/projects/test-003/results")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"


@pytest.mark.asyncio
async def test_poll_url_matches_project_id(valid_payload):
    async with make_client() as client:
        resp = await client.post("/projects/test-004/analyze", json=valid_payload)
    body = resp.json()
    assert body["poll_url"] == "/projects/test-004/results"


@pytest.mark.asyncio
async def test_multiple_projects_independent(valid_payload):
    """Two project IDs should not interfere with each other."""
    async with make_client() as client:
        r1 = await client.post("/projects/proj-A/analyze", json=valid_payload)
        r2 = await client.post("/projects/proj-B/analyze", json=valid_payload)
    assert r1.json()["status"] == "queued"
    assert r2.json()["status"] == "queued"
    assert r1.json()["poll_url"] != r2.json()["poll_url"]
