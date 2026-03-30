# Carbon Project Land Cover Monitor

Carbon Monitor is a local geospatial analysis app for screening forest-carbon projects for possible land-cover loss between two years. It combines NASA HLS imagery, NDVI-based change detection, an optional lightweight U-Net segmentation model, project-level risk scoring, and a small web UI for reviewing results.

The repository currently ships as a CPU-friendly prototype:

- FastAPI backend for job submission and result polling
- Streamlit dashboard for uploading a GeoJSON project boundary and reviewing outputs
- HLS scene search/download pipeline with caching and fallback logic
- NDVI-first forest loss detection with optional model inference
- SQLite-backed run history and MLflow experiment logging

## What the project does

For a project polygon and two monitoring years, the pipeline:

1. Validates the GeoJSON and time range.
2. Finds intersecting MGRS tiles from the local tile grid.
3. Searches NASA HLS scenes for each year, with cloud-threshold and prior-year fallback when needed.
4. Downloads and validates the selected scenes.
5. Mosaics tiles, clips imagery to the project boundary, and checks alignment across years.
6. Computes NDVI before and after the monitoring window.
7. Detects forest loss from NDVI decline.
8. Optionally runs a U-Net model on patch pairs and uses it only when it performs well enough against the NDVI reference.
9. Estimates forest loss area, derives a simple reversal-risk score, saves map outputs, and logs the run to MLflow.

## Current architecture

- `api/`: FastAPI app and HTTP routes
- `dashboard/`: Streamlit dashboard
- `src/pipeline.py`: main orchestration logic
- `src/lpdaac.py`: HLS search and download helpers
- `src/ndvi.py`: NDVI computation and forest-loss masks
- `src/prithvi.py`: model wrapper; now backed by a lightweight U-Net instead of the original Prithvi-100M dependency
- `src/risk_scoring.py`: hectare conversion, offset lookup, and risk score calculation
- `src/run_store.py`: SQLite persistence for submitted runs
- `configs/settings.py`: central configuration
- `models/unet.py`: ForestUNet definition
- `ml_models/`: trained model artifacts and example plots
- `generated/mgrs_tile_grid.geojson`: local tile lookup data used during scene discovery

## Analysis flow

The main entry point is the FastAPI route:

- `POST /projects/{project_id}/analyze`

It accepts:

- `geojson`: project polygon or feature collection
- `start_year`
- `end_year`
- `annual_offset_tco2` (optional)

The API queues a background task and returns a `run_id` plus a poll URL:

- `GET /projects/{project_id}/results`

While running, the result payload includes progress metadata such as stage name, step number, and scene-download counters. When complete, it returns metrics and artifact URLs including:

- `forest_loss_ha`
- `forest_loss_pct`
- `risk_score`
- `risk_flag`
- `ndvi_before_mean`
- `ndvi_after_mean`
- `iou_score`
- `f1_score`
- `forest_loss_map_url`
- `ndvi_overlay_url`
- `mlflow_run_id`

## Detection and scoring logic

### Biome defaults

Biome is inferred from polygon centroid latitude:

- `tropical`: `abs(lat) <= 23`
- `subtropical`: `23 < abs(lat) <= 40`
- `temperate`: `abs(lat) > 40`

Each biome selects a default NDVI threshold and sequestration rate from [`configs/settings.py`](/c:/python/ICT_project/carbon-monitor/configs/settings.py).

### Forest loss

Forest loss is estimated from NDVI decline between the start and end year. The pipeline uses NDVI as the baseline method everywhere, then attempts model-based segmentation on 128x128 patches. If the model's average IoU against the NDVI reference is below `0.6`, the pipeline falls back to NDVI masks for the final result.

### Model status

Despite the module name `src/prithvi.py`, the current implementation no longer downloads or runs Prithvi-100M. It now wraps a local U-Net (`ForestUNet`) that:

- loads weights from `ml_models/unet_forest.pth` when present
- runs on CPU
- falls back to random weights when no checkpoint exists
- usually triggers the NDVI fallback until trained weights are supplied

Training support for that model lives in [`notebooks/training.ipynb`](/c:/python/ICT_project/carbon-monitor/notebooks/training.ipynb).

### Risk score

Risk scoring is implemented in [`src/risk_scoring.py`](/c:/python/ICT_project/carbon-monitor/src/risk_scoring.py):

- each detected loss pixel is treated as `30m x 30m = 900 m^2 = 0.09 ha`
- annual loss is `forest_loss_ha / (end_year - start_year)`
- risk score is `(annual_loss_ha * sequestration_rate) / claimed_annual_offset`
- score `> 0.05` is flagged `HIGH`, otherwise `LOW`
- if annual offset data is missing, the flag becomes `DATA_MISSING`

If `annual_offset_tco2` is not provided in the request, the pipeline tries to load it from `data/verra_offsets.csv`.

## Inputs and outputs

### Required inputs

- a project polygon in GeoJSON format
- `start_year` and `end_year`
- NASA Earthdata credentials in `.env` for HLS downloads

### Optional inputs

- `annual_offset_tco2` in the API request
- `ml_models/unet_forest.pth` for trained segmentation weights
- `data/verra_offsets.csv` with columns `project_id,annual_offset_tco2`

### Generated outputs

For each project run, the app writes files under `results/<project_id>/`:

- `clipped_<year>.tif`
- `forest_loss.png`
- `ndvi_overlay.png`

It also stores:

- run history in `generated/carbon_monitor.db`
- MLflow metadata in `mlflow.db`

## Setup

### Environment

The repo was developed around Python 3.10 on Windows. Dependencies are listed in [`requirements.txt`](/c:/python/ICT_project/carbon-monitor/requirements.txt) and [`environment.yml`](/c:/python/ICT_project/carbon-monitor/environment.yml).

Install with pip:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### `.env`

Create a `.env` file in the repo root using [`.env.example`](/c:/python/ICT_project/carbon-monitor/.env.example) as a template:

```env
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
```

Other tunables such as cloud threshold, scenes per year, patch size, and API host/port are defined in [`configs/settings.py`](/c:/python/ICT_project/carbon-monitor/configs/settings.py).

## Running the app

Start the backend:

```powershell
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger docs at `http://localhost:8000/docs`.

Start the dashboard in a second terminal:

```powershell
python -m streamlit run dashboard/app.py
```

Open the dashboard at `http://localhost:8501`.

Start MLflow UI if you want experiment tracking:

```powershell
python -m mlflow ui --backend-store-uri "sqlite:///C:/python/ICT_project/carbon-monitor/mlflow.db"
```

Open MLflow at `http://localhost:5000`.

The same commands are also noted in [`run_commands.txt`](/c:/python/ICT_project/carbon-monitor/run_commands.txt).

## Example usage

1. Start FastAPI and Streamlit.
2. Open the dashboard.
3. Enter a project ID.
4. Upload a GeoJSON boundary.
5. Choose `start_year` and `end_year`.
6. Optionally enter `annual_offset_tco2`.
7. Click `Run Analysis`.
8. Watch progress while the backend searches scenes, downloads imagery, and builds outputs.

## Important implementation notes

- The API stores live in-progress results in memory, so active status information is lost if the FastAPI server restarts.
- Completed/queued run metadata is also persisted to SQLite, which lets the latest run be recovered at a basic level.
- The results directory is created at API startup before static file mounting, preventing first-run static-file errors.
- Scene search uses bounding-box based queries and can fall back to earlier years and looser cloud thresholds when a requested year has no usable scenes.
- The pipeline can auto-swap latitude/longitude ordering if the uploaded polygon appears reversed.
- The system is designed to keep running even when the ML model is unavailable or underperforming by falling back to NDVI outputs.

## Known limitations

- No trained `unet_forest.pth` checkpoint is included by default, so many runs will use NDVI as the final segmentation source.
- HLS downloads depend on valid Earthdata credentials and network access.
- Risk scoring is intentionally simple and should be treated as a screening signal, not a final crediting decision.
- The local tile grid is finite and scene coverage depends on the configured search/fallback rules.
- The dashboard is optimized for local use and polls the backend rather than using a distributed job queue.

## Project status

This repository is best understood as a functional prototype for local analysis and demonstration. The end-to-end flow is implemented, the UI is usable, and the fallback behavior is deliberate, but model training data, production hardening, and broader validation are still future work.
