"""
Tests for src/risk_scoring.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.risk_scoring import (
    compute_risk_score,
    forest_loss_hectares,
    load_verra_offset,
)


# ---------------------------------------------------------------------------
# forest_loss_hectares
# ---------------------------------------------------------------------------

def test_forest_loss_hectares_100_pixels():
    mask = np.ones(100, dtype=bool)
    assert forest_loss_hectares(mask) == pytest.approx(9.0)


def test_forest_loss_hectares_zero():
    mask = np.zeros((50, 50), dtype=bool)
    assert forest_loss_hectares(mask) == 0.0


def test_forest_loss_hectares_partial():
    mask = np.zeros(200, dtype=bool)
    mask[:50] = True  # 50 True pixels → 50 * 0.09 = 4.5 ha
    assert forest_loss_hectares(mask) == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# compute_risk_score — edge cases
# ---------------------------------------------------------------------------

def test_risk_score_offset_zero_returns_data_missing():
    result = compute_risk_score(100.0, 12.0, 0, 2020, 2023)
    assert result["risk_flag"] == "DATA_MISSING"
    assert result["risk_score"] is None


def test_risk_score_offset_negative_returns_data_missing():
    result = compute_risk_score(100.0, 12.0, -1.0, 2020, 2023)
    assert result["risk_flag"] == "DATA_MISSING"
    assert result["risk_score"] is None


def test_risk_score_offset_none_returns_data_missing():
    result = compute_risk_score(100.0, 12.0, None, 2020, 2023)
    assert result["risk_flag"] == "DATA_MISSING"
    assert result["risk_score"] is None


def test_risk_score_zero_loss_returns_low():
    result = compute_risk_score(0.0, 12.0, 5000.0, 2020, 2023)
    assert result["risk_score"] == pytest.approx(0.0)
    assert result["risk_flag"] == "LOW"


# ---------------------------------------------------------------------------
# compute_risk_score — formula verification
# ---------------------------------------------------------------------------
#
# loss=100 ha, rate=12 tCO2/ha/yr, offset=10000 tCO2/yr, years=3
#   annual_loss_ha = 100 / 3 ≈ 33.333
#   risk_score     = (33.333 * 12) / 10000 = 400 / 10000 = 0.04
#   0.04 < 0.05 (RISK_THRESHOLD) → "LOW"
#

def test_risk_score_formula_known_inputs():
    result = compute_risk_score(100.0, 12.0, 10000.0, 2020, 2023)
    expected_score = (100.0 / 3) * 12.0 / 10000.0
    assert result["risk_score"] == pytest.approx(expected_score, rel=1e-6)
    assert result["num_years"] == 3
    assert result["annual_loss_ha"] == pytest.approx(100.0 / 3, rel=1e-6)
    # 0.04 < 0.05 → LOW
    assert result["risk_flag"] == "LOW"


def test_risk_score_high_flag():
    # Craft inputs that push score above 0.05
    # annual_loss_ha = 500 / 1 = 500
    # risk_score = (500 * 12) / 10000 = 0.6 → HIGH
    result = compute_risk_score(500.0, 12.0, 10000.0, 2020, 2021)
    assert result["risk_flag"] == "HIGH"
    assert result["risk_score"] > 0.05


def test_risk_score_valid_returns_expected_keys():
    result = compute_risk_score(50.0, 4.0, 2000.0, 2018, 2022)
    for key in ("risk_score", "risk_flag", "annual_loss_ha", "num_years"):
        assert key in result


# ---------------------------------------------------------------------------
# load_verra_offset
# ---------------------------------------------------------------------------

def test_load_verra_offset_nonexistent_path_returns_none():
    result = load_verra_offset("VCS-9999", Path("/nonexistent/path/verra.csv"))
    assert result is None


def test_load_verra_offset_missing_project_returns_none(tmp_path):
    csv_file = tmp_path / "verra.csv"
    csv_file.write_text("project_id,annual_offset_tco2\nVCS-0001,5000\n")
    result = load_verra_offset("VCS-9999", csv_file)
    assert result is None


def test_load_verra_offset_found(tmp_path):
    csv_file = tmp_path / "verra.csv"
    csv_file.write_text("project_id,annual_offset_tco2\nVCS-1234,12500.5\n")
    result = load_verra_offset("VCS-1234", csv_file)
    assert result == pytest.approx(12500.5)
