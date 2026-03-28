import numpy as np

from src.ml_integration import (
    build_pseudo_labels,
    build_tile_risk_features,
    build_unet_features,
)


def test_build_unet_features_from_ndvi_rasters():
    before = np.array([[0.7, 0.6], [0.4, 0.2]], dtype=np.float32)
    after = np.array([[0.3, 0.5], [0.1, 0.2]], dtype=np.float32)

    features = build_unet_features(before, after)

    assert features.shape == (3, 2, 2)
    assert np.isclose(features[2, 0, 0], -0.4)


def test_build_pseudo_labels_flags_forest_loss():
    before = np.array([[0.7, 0.4], [0.2, 0.8]], dtype=np.float32)
    after = np.array([[0.3, 0.35], [0.1, 0.7]], dtype=np.float32)

    labels = build_pseudo_labels(before, after, forest_threshold=0.35, delta_threshold=0.15)

    assert labels.dtype == np.float32
    assert labels[0, 0] == 1.0
    assert labels[1, 1] == 0.0


def test_build_tile_risk_features_returns_expected_summary():
    before = np.array([[0.7, 0.6], [0.5, 0.4]], dtype=np.float32)
    after = np.array([[0.3, 0.4], [0.5, 0.2]], dtype=np.float32)

    features = build_tile_risk_features(
        before,
        after,
        cloud_fraction=0.2,
        biome="temperate",
    )

    assert features["biome"] == "temperate"
    assert features["cloud_fraction"] == 0.2
    assert features["feature_vector"].shape[0] == 8
    assert 0.0 <= features["pseudo_target"] <= 1.0
