"""
Risk scoring, Verra offset loading, and forest loss visualisation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.settings import settings


# ---------------------------------------------------------------------------
# Pixel → hectare conversion
# ---------------------------------------------------------------------------

def forest_loss_hectares(loss_mask: np.ndarray) -> float:
    """Return total forest-loss area in hectares.

    Each True pixel represents a 30 m × 30 m cell = 900 m² = 0.09 ha.
    """
    return float(np.count_nonzero(loss_mask) * 0.09)


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def compute_risk_score(
    forest_loss_ha: float,
    sequestration_rate: float,
    claimed_annual_offset: float | None,
    start_year: int,
    end_year: int,
) -> dict:
    """Compute a risk score for a carbon project.

    Returns a dict with keys:
        risk_score      float | None
        risk_flag       "HIGH" | "LOW" | "DATA_MISSING"
        annual_loss_ha  float
        num_years       int
    """
    num_years = end_year - start_year

    # Guard: missing or invalid offset
    if claimed_annual_offset is None or claimed_annual_offset <= 0:
        return {
            "risk_score": None,
            "risk_flag": "DATA_MISSING",
            "annual_loss_ha": forest_loss_ha / num_years if num_years > 0 else 0.0,
            "num_years": num_years,
        }

    # Guard: no loss
    if forest_loss_ha == 0:
        return {
            "risk_score": 0.0,
            "risk_flag": "LOW",
            "annual_loss_ha": 0.0,
            "num_years": num_years,
        }

    annual_loss_ha = forest_loss_ha / num_years
    risk_score = (annual_loss_ha * sequestration_rate) / claimed_annual_offset
    risk_flag = "HIGH" if risk_score > settings.RISK_THRESHOLD else "LOW"

    return {
        "risk_score": float(risk_score),
        "risk_flag": risk_flag,
        "annual_loss_ha": float(annual_loss_ha),
        "num_years": num_years,
    }


# ---------------------------------------------------------------------------
# Verra offset loader
# ---------------------------------------------------------------------------

def load_verra_offset(project_id: str, verra_csv_path: Path) -> float | None:
    """Return the annual offset (tCO₂) for *project_id* from a Verra CSV.

    Returns None on any error (file missing, project not found, parse failure).
    Expected CSV columns: project_id, annual_offset_tco2
    """
    try:
        import pandas as pd

        if not Path(verra_csv_path).exists():
            return None

        df = pd.read_csv(verra_csv_path)
        row = df[df["project_id"] == project_id]
        if row.empty:
            return None
        return float(row["annual_offset_tco2"].iloc[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def generate_forest_loss_png(
    loss_mask: np.ndarray,
    clipped_raster: np.ndarray,
    save_path: Path,
) -> None:
    """Save a PNG showing the RGB composite with forest-loss pixels in red.

    clipped_raster band order: [Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5]
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Build an RGB image (bands 2, 1, 0) normalised to [0, 1]
    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr, dtype=float)
        return (arr - lo) / (hi - lo)

    r = _norm(clipped_raster[2].astype(float))
    g = _norm(clipped_raster[1].astype(float))
    b = _norm(clipped_raster[0].astype(float))
    rgb = np.stack([r, g, b], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)

    # Overlay loss pixels in red
    red_overlay = np.zeros((*loss_mask.shape, 4), dtype=float)
    red_overlay[loss_mask] = [1.0, 0.0, 0.0, 0.6]
    ax.imshow(red_overlay)

    ax.set_title("Forest Loss Map")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
