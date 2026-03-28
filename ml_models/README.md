Basic usage:

```python
from src.ml_integration import predict_tile_risk, predict_unet_mask

# Optional U-Net segmentation from pipeline tensors
unet_result = predict_unet_mask(before_spec, after_spec)
segmentation_mask = unet_result["mask"]

# Optional tile-level RF risk from pipeline NDVI outputs
rf_result = predict_tile_risk(
    ndvi_before,
    ndvi_after,
    cloud_fraction=unusable_pct,
    biome=biome,
)
risk_score = rf_result["risk_score"]
```

Training commands:

```bash
python -m ml_models.train_unet
python -m ml_models.train_rf --metadata-csv results/project_metadata.csv
```
