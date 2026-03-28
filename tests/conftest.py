"""
Shared pytest fixtures for the Carbon Monitor test suite.
"""

import pytest
import json
from pathlib import Path


# Register custom markers so pytest doesn't warn about unknown markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring network/credentials"
    )


@pytest.fixture
def test_polygon_geojson():
    path = Path(__file__).parent / "fixtures" / "test_polygon.geojson"
    return json.loads(path.read_text())


@pytest.fixture
def tile_grid():
    from src.tile_detection import load_tile_grid
    from configs.settings import settings

    return load_tile_grid(settings.MGRS_GRID_PATH)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring network/credentials"
    )


@pytest.fixture
def test_polygon_geojson():
    path = Path(__file__).parent / "fixtures" / "test_polygon.geojson"
    return json.loads(path.read_text())


@pytest.fixture
def tile_grid():
    from src.tile_detection import load_tile_grid
    from configs.settings import settings

    return load_tile_grid(settings.MGRS_GRID_PATH)
