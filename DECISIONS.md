# DECISIONS.md

## Stage N — <date>
- Changed: <what>
- Reason: <why>
- Affects: <which downstream stages>

## Stage 1
- Repo: https://github.com/soulrised00-max/Carbon_monitor.git
- Env: venv, Python 3.10, Windows 11, VS Code, CPU only, no errors

## Stage 2
- Installed: shapely
- Changed: configs/settings.py Config class → ConfigDict
- Reason: Pydantic v2 deprecation warning
- Affects: nothing downstream

## Stage 3
- Installed: rasterio
- Changed: search_scenes() uses bounding_box instead of producer_granule_id wildcard
- Reason: CMR wildcard returns 0 results for HLS; bounding_box confirmed working
- Affects: Stage 7 pipeline.py — pass bbox explicitly from tile["bbox"]

- Changed: search_scenes() signature → search_scenes(tile_id, year, cloud_max, bbox=None)
- Reason: allows direct bbox pass; tile_id is now a cache label only
- Affects: Stage 7 pipeline.py

- Changed: scene dicts include "actual_tile_id" field
- Reason: bbox search returns multiple HLS tiles; actual tile ID differs from logical tile_id
- Affects: Stage 7 pipeline.py — use actual_tile_id for mosaicking and cache

- Changed: data/mgrs_tile_grid.geojson — all 10 tile IDs replaced with real CMR-confirmed IDs
- Reason: Stage 1 IDs not valid HLS tiles
- Mapping:
    Karnataka (78-79, 13-14)   → T43PHR, T44PKA, T43PHQ
    Karnataka (78-79, 12-13)   → T43PHP, T43PHQ
    South India (79-80, 8-9)   → T44NLP
    Central India (78-79,17-18)→ T43QHA
    North India (79-80, 20-21) → T44QKH
    Amazon (-77,-76, 3-4)      → T18NTJ
    Amazon (-76,-75, -5,-4)    → T18MUA
    Australia (149-150,-34,-33)→ T55HFC
    East Africa (36-37, -4,-3) → T36MZA
    East Africa (36-37, -3,-2) → T36MZB
- Affects: test_tile_detection.py — assert T43PHR (not T43PGP) for Karnataka polygon

## Stage 4
- Changed: test_preprocessing.py — southern hemisphere UTM assertion
- Reason: pyproj encodes southern zones via false northing (10,000,000 m), not "SOUTH" in WKT
- Fix: assert "10000000" in crs.to_wkt()
- Affects: nothing downstream (test assertion bug only)

## Stage 5
- No logic changes
- Prithvi: zero-shot only, model not downloaded in unit tests
- evaluate_against_hansen and reconstruct_from_patches in src/prithvi.py per spec

## Stage 6 — <date>
- Implemented: src/risk_scoring.py, src/mlflow_tracking.py, tests/test_risk_scoring.py
- forest_loss_hectares: 30m pixel = 900m² = 0.09 ha per True pixel
- compute_risk_score edge cases: claimed_annual_offset=0 or None → DATA_MISSING; forest_loss_ha=0 → risk_score=0.0, LOW
- Risk threshold read from settings.RISK_THRESHOLD (0.05); > threshold = HIGH, <= = LOW
- load_verra_offset: expects CSV with columns [project_id, annual_offset_tco2]; returns None on any error, never raises
- generate_forest_loss_png: RGB composite from band indices [2,1,0]; loss overlay at alpha=0.4 red
- mlflow_tracking.log_run: skips None metrics silently; skips artifact paths that don't exist on disk; logs all params as str()
- MLflow experiment name from settings.MLFLOW_EXPERIMENT_NAME ("carbon-monitor")
- Known formula result: loss=100ha, rate=12, offset=10000, years=3 → annual_loss=33.33, score=0.04 → LOW
- Affects: Stage 7 (pipeline.py calls compute_risk_score, load_verra_offset, generate_forest_loss_png, log_run)

## Stage 7 — <date>
- pipeline.py: mosaic_tiles called only when len(paths) > 1; single-tile year reads first .tif directly
- pipeline.py: Prithvi OOM fallback gracefully per-patch; avg IoU threshold 0.6 governs method selection
- api/routes.py: results_store is module-level dict — resets on server restart (by design, Stage 7 only)
- api/main.py: RESULTS_DIR created at startup before StaticFiles mount to avoid 500 on first run
- tests/test_api.py: asyncio_mode = auto required in pytest.ini
- Affects: Stage 8 (dashboard reads /static/ URLs; ensure RESULTS_DIR exists before starting)
## Stage ML-1 — Model swap: Prithvi-100M → ForestUNet

### src/prithvi.py
- Changed: replaced Prithvi-100M HuggingFace loader with ForestUNet (lightweight U-Net, ~7M params)
- Reason: Prithvi-100M requires CUDA and trust_remote_code; not runnable on CPU laptop
- All four public function signatures are IDENTICAL — pipeline.py untouched:
    load_prithvi_model(device)           → (model, config)
    run_prithvi_inference(patch, m, c)   → (128, 128) bool mask
    evaluate_against_hansen(pred, true)  → {iou, precision, recall, f1}
    reconstruct_from_patches(patches, H, W) → (H, W) bool array
- Affects: nothing downstream

### models/unet.py  (NEW FILE)
- ForestUNet class: 6-band input → 1-channel logit output, ~7M parameters
- DoubleConv / Down / Up building blocks
- logits_to_mask() convenience function
- Trained separately on Kaggle (see notebooks/training.ipynb)

### ml_models/unet_forest.pth  (PLACEHOLDER — you add this)
- Default weight path: ml_models/unet_forest.pth
- If missing: pipeline runs with random weights, IoU will be low,
  NDVI fallback triggers automatically (existing pipeline behaviour)
- After Kaggle training: drop .pth here, no code changes needed

### src/ndvi.py
- Added: confusion_matrix_stats(predicted_mask, true_mask) → dict
- Returns: TP, FP, FN, TN, accuracy, precision, recall, f1, iou
- Use for full-raster evaluation after patch reconstruction
- evaluate_against_hansen() in prithvi.py unchanged (patch-level only)

### tests/test_ndvi.py
- Added 3 tests for confusion_matrix_stats:
    test_confusion_matrix_known_values
    test_confusion_matrix_all_zero_predicted
    test_confusion_matrix_perfect_prediction
- Updated integration stub: test_unet_load_and_inference (skip marker)
- All 15 existing tests unchanged

### notebooks/training.ipynb  (NEW FILE)
- Kaggle-ready training notebook
- Sections: install → config → data prep (Option A: Kaggle dataset / Option B: raw GeoTIFFs)
  → Dataset/DataLoader → ForestUNet → BCE+Dice loss → training loop with early stopping
  → confusion matrix evaluation → visual predictions → download instructions
- Exports: unet_forest.pth (state_dict wrapped in checkpoint dict)

### Upgrade path
- Phase 1 (now):    Random weights → pipeline works end-to-end, NDVI fallback active
- Phase 2 (Kaggle): Train on Hansen+Sentinel-2, download unet_forest.pth
- Phase 3 (later):  Swap DoubleConv encoder for ResNet-18 backbone in models/unet.py only
