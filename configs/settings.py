"""
Central config. All tunable values live here.
Import with: from configs.settings import settings
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    # Paths
    REPO_ROOT: Path = Path(__file__).parent.parent
    CACHE_DIR: Path = REPO_ROOT / "cache"
    RESULTS_DIR: Path = REPO_ROOT / "results"
    MGRS_GRID_PATH: Path = REPO_ROOT / "generated" / "mgrs_tile_grid.geojson"
    RUNS_DB_PATH: Path = REPO_ROOT / "generated" / "carbon_monitor.db"

    # LPDAAC / Earthdata
    EARTHDATA_USERNAME: str = ""
    EARTHDATA_PASSWORD: str = ""

    # HLS download
    CLOUD_COVER_THRESHOLD: float = 0.30   # max acceptable scene cloud cover
    SCENES_PER_YEAR: int = 3              # top-N least cloudy scenes for compositing
    HLS_BANDS: list[str] = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    # Band order for model input: [Blue, Green, Red, NIR, SWIR1, SWIR2]

    # Biome NDVI thresholds (keyed by biome name)
    NDVI_THRESHOLDS: dict = {
        "tropical":    0.50,
        "subtropical": 0.35,
        "temperate":   0.25,
    }

    # Biome sequestration rates tCO2/ha/yr (IPCC Tier 1 defaults)
    SEQUESTRATION_RATES: dict = {
        "tropical":    12.0,
        "subtropical":  4.0,
        "temperate":    6.0,
    }

    # Risk scoring
    RISK_THRESHOLD: float = 0.05          # above this = HIGH risk
    PATCH_SIZE: int = 128
    MIN_VALID_NDVI_FRACTION: float = 0.10
    MIN_VALID_NDVI_PIXELS: int = 1000

    # MLflow
    MLFLOW_EXPERIMENT_NAME: str = "carbon-monitor"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
