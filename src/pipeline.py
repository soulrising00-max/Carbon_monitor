"""
Pipeline orchestration — runs as a FastAPI background task.
Writes results into results_store[project_id] as it progresses.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING
from numpy.typing import NDArray

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.ops import transform, unary_union

from configs.settings import settings
from src.validation import validate_analyze_request
from src.tile_detection import load_tile_grid, find_covering_tiles, biome_params
from src.lpdaac import (
    search_scenes,
    select_top_scenes,
    download_scene,
    validate_download,
)
from src.cloud_masking import compute_cloud_mask, unusable_fraction
from src.preprocessing import (
    utm_crs_from_centroid,
    mosaic_tiles,
    clip_to_polygon,
    assert_pixel_alignment,
    normalize_bands,
    generate_patches,
)
from src.ndvi import (
    compute_ndvi,
    compute_forest_loss_mask,
    hls_invalid_pixel_mask,
    ndvi_stats,
    validate_ndvi_for_scoring,
)
from src.prithvi import (
    load_prithvi_model,
    run_prithvi_inference,
    evaluate_against_hansen,
    reconstruct_from_patches,
)
from src.risk_scoring import (
    forest_loss_hectares,
    compute_risk_score,
    load_verra_offset,
    generate_forest_loss_png,
)
from src.mlflow_tracking import log_run
from src.run_store import update_run

if TYPE_CHECKING:
    from api.schemas import AnalyzeRequest

# L30 (Landsat) band name mapping — mirrors lpdaac._L30_BAND_MAP
_L30_BAND_MAP: dict[str, str] = {"B8A": "B05", "B11": "B06", "B12": "B07"}


def _swap_xy_geometry(geom):
    """Return a geometry with longitude/latitude axes swapped."""
    return transform(lambda x, y, *args: (y, x), geom)


def _ndvi_overlay_png(before: np.ndarray, after: np.ndarray, save_path: Path) -> None:
    """Save side-by-side NDVI before/after as PNG."""
    import matplotlib.pyplot as plt

    before_invalid_mask = hls_invalid_pixel_mask(before)
    after_invalid_mask = hls_invalid_pixel_mask(after)
    ndvi_before = compute_ndvi(before, before_invalid_mask)
    ndvi_after = compute_ndvi(after, after_invalid_mask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ndvi_before, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title("NDVI — Before")
    axes[0].axis("off")
    axes[1].imshow(ndvi_after, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI — After")
    axes[1].axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _read_cache_to_array(
    cache_path: Path,
) -> tuple[np.ndarray, Affine, CRS]:
    """
    Stack all 7 HLS bands from a cache directory into a (7, H, W) array.

    Band order: [B02, B03, B04, B8A/B05, B11/B06, B12/B07, Fmask]
    Handles both L30 (Landsat) and S30 (Sentinel-2) band filenames.

    FIX (pipeline.py bug 1): was inline in run_pipeline with a duplicate
    `import numpy as np` and `import rasterio` inside the loop. Extracted
    to a helper so imports are clean and the logic is reusable.
    """
    band_order = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    band_arrays: list[NDArray[np.generic]] = []
    transform: Affine | None = None
    crs: CRS | None = None

    for band in band_order:
        # Try the canonical (S30 / settings) name first
        matches = [
            f for f in cache_path.glob("*.tif")
            if f.name.endswith(f".{band}.tif")
        ]
        if not matches:
            # Try the L30 equivalent
            alt = _L30_BAND_MAP.get(band)
            if alt:
                matches = [
                    f for f in cache_path.glob("*.tif")
                    if f.name.endswith(f".{alt}.tif")
                ]
        if not matches:
            raise RuntimeError(
                f"Band {band} (and its L30 equivalent) not found in {cache_path}. "
                f"Files present: {[f.name for f in cache_path.glob('*.tif')]}"
            )

        with rasterio.open(matches[0]) as src:
            band_arrays.append(src.read(1))
            if transform is None:
                transform = src.transform
                crs = src.crs

    if transform is None or crs is None:
        raise RuntimeError(
            f"Could not determine raster transform/CRS from cached bands in {cache_path}"
        )

    return np.stack(band_arrays, axis=0), transform, crs


def _find_band_file(cache_path: Path, band: str) -> Path | None:
    """Return the first matching band file in a cache directory."""
    matches = [f for f in cache_path.glob("*.tif") if f.name.endswith(f".{band}.tif")]
    if matches:
        return matches[0]

    alt = _L30_BAND_MAP.get(band)
    if alt:
        alt_matches = [
            f for f in cache_path.glob("*.tif") if f.name.endswith(f".{alt}.tif")
        ]
        if alt_matches:
            return alt_matches[0]
    return None


def _resolve_scenes_with_fallback(
    tile_id: str,
    requested_year: int,
    bbox: tuple[float, float, float, float] | None,
    restrict_to_tile: bool,
) -> tuple[list[dict], int, float]:
    """Search scenes using a fixed fallback sequence.

    Order:
    1. requested_year at configured cloud threshold
    2. requested_year at 0.8
    3. requested_year - 1 at 0.8
    """
    candidates = [
        (requested_year, settings.CLOUD_COVER_THRESHOLD),
        (requested_year, 0.8),
        (requested_year - 1, 0.8),
    ]

    for search_year, cloud_max in candidates:
        if search_year < 2013:
            continue
        scenes = search_scenes(
            tile_id,
            search_year,
            cloud_max,
            bbox=bbox,
            restrict_to_tile=restrict_to_tile,
        )
        if scenes:
            return scenes, search_year, cloud_max

    return [], requested_year, settings.CLOUD_COVER_THRESHOLD


def run_pipeline(
    run_id: str,
    project_id: str,
    req: "AnalyzeRequest",
    results_store: dict,
) -> None:
    """Full analysis pipeline. Runs as a FastAPI background task."""
    created_at = results_store.get(project_id, {}).get("created_at")
    results_store[project_id] = {
        "run_id": run_id,
        "status": "running",
        "project_id": project_id,
        "created_at": created_at,
    }
    warnings: list[str] = []
    diagnostics: dict[str, object] = {}
    update_run(run_id, project_id=project_id, status="running", warnings=warnings)

    try:
        # ── 1. Validate ──────────────────────────────────────────────────────
        geometries, err = validate_analyze_request(
            req.geojson, req.start_year, req.end_year
        )
        if err:
            results_store[project_id] = {
                "run_id": run_id,
                "status": "failed",
                "project_id": project_id,
                "created_at": created_at,
                "error": err,
            }
            update_run(run_id, status="failed", warnings=[err])
            return

        start_year: int = req.start_year
        end_year: int = req.end_year

        # ── 2. Biome params ──────────────────────────────────────────────────
        union_geom = unary_union(geometries)
        centroid = union_geom.centroid
        params = biome_params(centroid.y)
        biome = params["biome"]
        ndvi_threshold = params["ndvi_threshold"]
        seq_rate = params["sequestration_rate"]

        # ── 3. Tile grid + covering tiles ────────────────────────────────────
        tile_grid = load_tile_grid(settings.MGRS_GRID_PATH)
        tile_ids: list[str] = []
        for geom in geometries:
            tile_ids.extend(find_covering_tiles(geom, tile_grid))
        tile_ids = list(dict.fromkeys(tile_ids))  # deduplicate, preserve order
        diagnostics["tile_ids"] = tile_ids

        search_targets: list[dict] = []

        if not tile_ids:
            swapped_geometries = [_swap_xy_geometry(geom) for geom in geometries]
            swapped_tile_ids: list[str] = []
            for geom in swapped_geometries:
                swapped_tile_ids.extend(find_covering_tiles(geom, tile_grid))
            swapped_tile_ids = list(dict.fromkeys(swapped_tile_ids))

            if swapped_tile_ids:
                geometries = swapped_geometries
                tile_ids = swapped_tile_ids
                diagnostics["tile_ids"] = tile_ids
                diagnostics["coordinate_swap_applied"] = True
                warnings.append(
                    "Input polygon appeared to use latitude/longitude order. "
                    "Automatically swapped coordinates to longitude/latitude for tile matching."
                )
            else:
                warnings.append(
                    "No MGRS tiles cover the supplied polygon; results may be incomplete."
                )

        # ── 4. Scene discovery + download ─────────────────────────────────────
        union_geom = unary_union(geometries)
        centroid = union_geom.centroid
        params = biome_params(centroid.y)
        biome = params["biome"]
        ndvi_threshold = params["ndvi_threshold"]
        seq_rate = params["sequestration_rate"]

        year_tile_paths: dict[int, list[Path]] = {start_year: [], end_year: []}
        cloud_fractions: list[float] = []
        year_diagnostics: dict[int, dict[str, object]] = {
            start_year: {"tiles_considered": [], "valid_downloads": 0},
            end_year: {"tiles_considered": [], "valid_downloads": 0},
        }

        if tile_ids:
            search_targets = [tile for tile in tile_grid if tile["tile_id"] in tile_ids]
        else:
            search_targets = [
                {
                    "tile_id": "bbox_fallback",
                    "bbox": union_geom.bounds,
                    "restrict_to_tile": False,
                }
            ]
            diagnostics["bbox_fallback"] = list(union_geom.bounds)
            warnings.append(
                "No covering MGRS tile found locally; falling back to polygon bounding-box scene search."
            )

        for year in (start_year, end_year):
            for tile in search_targets:
                target_tile_id = tile["tile_id"]
                restrict_to_tile = tile.get("restrict_to_tile", True)
                tile_diag = {
                    "tile_id": target_tile_id,
                    "scenes_found": 0,
                    "scenes_selected": 0,
                    "validated_downloads": 0,
                    "search_year_used": year,
                    "cloud_threshold_used": settings.CLOUD_COVER_THRESHOLD,
                }
                year_diagnostics[year]["tiles_considered"].append(tile_diag)
                bbox = tile.get("bbox")
                scenes, search_year, cloud_max = _resolve_scenes_with_fallback(
                    target_tile_id,
                    year,
                    bbox,
                    restrict_to_tile,
                )
                print("Scenes fetched:", len(scenes))
                tile_diag["scenes_found"] = len(scenes)
                tile_diag["search_year_used"] = search_year
                tile_diag["cloud_threshold_used"] = cloud_max
                valid_scenes = select_top_scenes(scenes, settings.SCENES_PER_YEAR)
                print("Scenes after filtering:", len(valid_scenes))
                if scenes and cloud_max > settings.CLOUD_COVER_THRESHOLD:
                    warnings.append(
                        f"No scenes found for tile {target_tile_id} year {year} "
                        f"under cloud threshold {settings.CLOUD_COVER_THRESHOLD:.2f}; "
                        f"using cloud threshold {cloud_max:.2f} instead."
                    )
                if scenes and search_year != year:
                    warnings.append(
                        f"No scenes found for tile {target_tile_id} in requested year {year}; "
                        f"using fallback scenes from {search_year}. This is reflected in the app warnings."
                    )
                tile_diag["scenes_selected"] = len(valid_scenes)
                if not valid_scenes:
                    warnings.append(
                        f"No scenes found for tile {target_tile_id} year {year}."
                    )
                    continue

                for scene in valid_scenes:
                    try:
                        cache_tile_id = (
                            target_tile_id
                            if restrict_to_tile
                            else scene.get("actual_tile_id", target_tile_id)
                        )
                        cache_path = download_scene(
                            scene["granule_id"],
                            scene["download_urls"],
                            cache_tile_id,
                            search_year,
                            settings.CACHE_DIR,
                        )
                        if not validate_download(cache_path):
                            warnings.append(
                                f"Download validation failed for {scene['granule_id']}."
                            )
                            continue

                        # Cloud masking
                        fmask_candidates = list(cache_path.glob("*Fmask*"))
                        if fmask_candidates:
                            band_paths = {}
                            for band in ("B02", "B03", "B04"):
                                band_path = _find_band_file(cache_path, band)
                                if band_path is not None:
                                    band_paths[band] = band_path
                            try:
                                cloud_mask = compute_cloud_mask(
                                    fmask_candidates[0],
                                    union_geom,
                                    band_paths=band_paths if len(band_paths) == 3 else None,
                                )
                                cloud_fractions.append(unusable_fraction(cloud_mask))
                            except Exception as exc:  # noqa: BLE001
                                warnings.append(
                                    f"Cloud masking failed for {scene['granule_id']}: {exc}"
                                )

                        year_tile_paths[year].append(cache_path)
                        tile_diag["validated_downloads"] += 1
                        year_diagnostics[year]["valid_downloads"] += 1
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"Scene {scene['granule_id']}: {exc}")

        unusable_pct = float(np.mean(cloud_fractions)) if cloud_fractions else 0.0
        diagnostics["year_diagnostics"] = year_diagnostics

        # ── 5. Mosaic → clip → align ─────────────────────────────────────────
        target_crs = utm_crs_from_centroid(centroid.x, centroid.y)
        proj_results_dir = settings.RESULTS_DIR / project_id
        proj_results_dir.mkdir(parents=True, exist_ok=True)

        clipped: dict[int, tuple] = {}
        for year in (start_year, end_year):
            paths = year_tile_paths[year]
            if not paths:
                year_summary = year_diagnostics.get(year, {})
                tile_summary = ", ".join(
                    (
                        f"{tile['tile_id']}:found={tile['scenes_found']},"
                        f"selected={tile['scenes_selected']},"
                        f"valid={tile['validated_downloads']}"
                    )
                    for tile in year_summary.get("tiles_considered", [])
                )
                details = (
                    f"tiles={tile_ids or []}; "
                    f"year_valid_downloads={year_summary.get('valid_downloads', 0)}; "
                    f"per_tile=[{tile_summary or 'none'}]"
                )
                raise RuntimeError(
                    f"No valid downloaded tiles for year {year}. Cannot continue. "
                    f"Recent details: {details}"
                )

            if len(paths) > 1:
                # FIX (pipeline.py bug 2): mosaic_tiles expects file paths to
                # GeoTIFF rasters, not cache directories. Build a single stacked
                # GeoTIFF per cache_path first, then mosaic those files.
                # Previously passed raw cache_path directories directly, which
                # caused rasterio.open() to fail on a directory path.
                tif_paths: list[Path] = []
                for cp in paths:
                    stacked_path = cp.parent / f"{cp.name}_stacked.tif"
                    if not stacked_path.exists():
                        arr, tr, cr = _read_cache_to_array(cp)
                        profile = {
                            "driver": "GTiff",
                            "count": arr.shape[0],
                            "dtype": arr.dtype,
                            "crs": cr,
                            "transform": tr,
                            "width": arr.shape[2],
                            "height": arr.shape[1],
                        }
                        with rasterio.open(stacked_path, "w", **profile) as dst:
                            dst.write(arr)
                    tif_paths.append(stacked_path)
                data, transform, crs = mosaic_tiles(tif_paths, target_crs)
            else:
                # Single tile — read directly from the cache directory
                data, transform, crs = _read_cache_to_array(paths[0])

            save_path = proj_results_dir / f"clipped_{year}.tif"
            clipped_data, clipped_transform = clip_to_polygon(
                data, transform, crs, union_geom, save_path
            )
            clipped[year] = (clipped_data, clipped_transform, clipped_data.shape)

        before_data, before_transform, before_shape = clipped[start_year]
        after_data, after_transform, after_shape = clipped[end_year]

        assert_pixel_alignment(
            before_transform, before_shape, after_transform, after_shape
        )

        # ── 6. Normalize + patches ───────────────────────────────────────────
        # Strip QA band (index 6 = Fmask); keep 6 spectral bands only
        before_spec = before_data[:6]
        after_spec = after_data[:6]
        before_qa = before_data[6]
        after_qa = after_data[6]

        before_invalid_mask = hls_invalid_pixel_mask(before_spec, before_qa)
        after_invalid_mask = hls_invalid_pixel_mask(after_spec, after_qa)
        before_norm = normalize_bands(before_spec, before_invalid_mask)
        after_norm = normalize_bands(after_spec, after_invalid_mask)

        patches = generate_patches(before_norm, after_norm, settings.PATCH_SIZE)
        before_invalid_patches = generate_patches(
            before_invalid_mask[np.newaxis, ...].astype(np.uint8),
            before_invalid_mask[np.newaxis, ...].astype(np.uint8),
            settings.PATCH_SIZE,
        )
        after_invalid_patches = generate_patches(
            after_invalid_mask[np.newaxis, ...].astype(np.uint8),
            after_invalid_mask[np.newaxis, ...].astype(np.uint8),
            settings.PATCH_SIZE,
        )

        full_h, full_w = before_norm.shape[1], before_norm.shape[2]

        # ── 7. NDVI baseline masks ────────────────────────────────────────────
        ndvi_patch_masks: list[dict] = []
        for i, patch in enumerate(patches):
            patch_before_invalid = before_invalid_patches[i]["before"][0].astype(bool)
            patch_after_invalid = after_invalid_patches[i]["before"][0].astype(bool)
            loss = compute_forest_loss_mask(
                patch["before"],
                patch["after"],
                ndvi_threshold,
                patch_before_invalid,
                patch_after_invalid,
            )
            ndvi_patch_masks.append(
                {"mask": loss, "row": patch["row"], "col": patch["col"]}
            )

        ndvi_loss_mask = reconstruct_from_patches(ndvi_patch_masks, full_h, full_w)
        ndvi_result = ndvi_loss_mask
        print("NDVI output exists:", ndvi_result is not None)

        # ── 8. Prithvi inference ──────────────────────────────────────────────
        segmentation_method = "ndvi"
        prithvi_loss_mask = None
        avg_iou = 0.0
        iou_score: float | None = None
        f1_score: float | None = None

        try:
            model, config = load_prithvi_model(device="cpu")
            prithvi_patch_masks: list[dict] = []
            iou_scores: list[float] = []

            for i, patch in enumerate(patches):
                try:
                    prithvi_pred = run_prithvi_inference(patch["before"], model, config)
                    ndvi_ref = ndvi_patch_masks[i]["mask"]
                    metrics = evaluate_against_hansen(prithvi_pred, ndvi_ref)
                    iou_scores.append(metrics["iou"])
                    prithvi_patch_masks.append(
                        {"mask": prithvi_pred, "row": patch["row"], "col": patch["col"]}
                    )
                except RuntimeError as exc:
                    warnings.append(f"Prithvi patch {i} failed: {exc}")
                    prithvi_patch_masks.append(
                        {
                            "mask": ndvi_patch_masks[i]["mask"],
                            "row": patch["row"],
                            "col": patch["col"],
                        }
                    )
                    iou_scores.append(0.0)

            avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
            if avg_iou >= 0.6:
                prithvi_loss_mask = reconstruct_from_patches(
                    prithvi_patch_masks, full_h, full_w
                )
                segmentation_method = "prithvi"
                iou_score = avg_iou
                all_pred = np.concatenate(
                    [p["mask"].ravel() for p in prithvi_patch_masks]
                )
                all_ref = np.concatenate([p["mask"].ravel() for p in ndvi_patch_masks])
                m = evaluate_against_hansen(all_pred.astype(bool), all_ref.astype(bool))
                f1_score = m["f1"]
            else:
                warnings.append(
                    f"Prithvi avg IoU {avg_iou:.3f} < 0.6 — falling back to NDVI masks."
                )

        except RuntimeError as exc:
            warnings.append(f"Prithvi load failed, using NDVI masks: {exc}")

        final_loss_mask = (
            prithvi_loss_mask if prithvi_loss_mask is not None else ndvi_loss_mask
        )

        # ── 9. Risk scoring ───────────────────────────────────────────────────
        ndvi_before = compute_ndvi(before_norm, before_invalid_mask)
        ndvi_after = compute_ndvi(after_norm, after_invalid_mask)
        ndvi_valid, ndvi_validation_message = validate_ndvi_for_scoring(
            ndvi_before,
            ndvi_after,
            min_valid_fraction=settings.MIN_VALID_NDVI_FRACTION,
            min_valid_pixels=settings.MIN_VALID_NDVI_PIXELS,
        )
        overlap_valid_pixels = int(
            (np.isfinite(ndvi_before) & np.isfinite(ndvi_after)).sum()
        )
        total_ndvi_pixels = int(ndvi_before.size)
        diagnostics["ndvi_overlap_valid_pixels"] = overlap_valid_pixels
        diagnostics["ndvi_overlap_valid_fraction"] = (
            (overlap_valid_pixels / total_ndvi_pixels) if total_ndvi_pixels else 0.0
        )
        diagnostics["ndvi_min_valid_fraction_required"] = settings.MIN_VALID_NDVI_FRACTION
        diagnostics["ndvi_min_valid_pixels_required"] = settings.MIN_VALID_NDVI_PIXELS
        diagnostics["ndvi_valid_for_scoring"] = ndvi_valid
        if not ndvi_valid:
            warnings.append(
                ndvi_validation_message
                + " Skipping forest loss and risk scoring for this run."
            )

        loss_ha = None
        loss_pct = None
        if ndvi_valid:
            loss_ha = forest_loss_hectares(final_loss_mask)
            total_pixels = final_loss_mask.size
            loss_pct = (
                (float(final_loss_mask.sum()) / total_pixels * 100)
                if total_pixels
                else 0.0
            )

        request_offset = getattr(req, "annual_offset_tco2", None)
        verra_offset = request_offset
        offset_source = "request" if request_offset is not None else None
        if verra_offset is None:
            verra_offset = load_verra_offset(
                project_id, settings.REPO_ROOT / "data" / "verra_offsets.csv"
            )
            if verra_offset is not None:
                offset_source = "csv"
        if verra_offset is None:
            warnings.append(
                "Risk scoring metadata missing: no annual_offset_tco2 was provided in the request "
                "and no matching row was found in data/verra_offsets.csv."
            )
        diagnostics["annual_offset_tco2"] = verra_offset
        diagnostics["annual_offset_source"] = offset_source
        if ndvi_valid:
            risk = compute_risk_score(
                loss_ha, seq_rate, verra_offset, start_year, end_year
            )
        else:
            risk = {
                "risk_score": None,
                "risk_flag": "DATA_MISSING",
                "annual_loss_ha": None,
                "num_years": end_year - start_year,
            }

        # ── 10. PNGs ──────────────────────────────────────────────────────────
        loss_png_path = proj_results_dir / "forest_loss.png"
        if ndvi_valid:
            generate_forest_loss_png(final_loss_mask, after_spec, loss_png_path)

        ndvi_png_path = proj_results_dir / "ndvi_overlay.png"
        _ndvi_overlay_png(before_norm, after_norm, ndvi_png_path)

        # ── 11. NDVI stats ────────────────────────────────────────────────────
        ndvi_before_stats = ndvi_stats(ndvi_before)
        ndvi_after_stats = ndvi_stats(ndvi_after)

        # ── 12. MLflow ────────────────────────────────────────────────────────
        mlflow_params = {
            "project_id": project_id,
            "start_year": start_year,
            "end_year": end_year,
            "segmentation_method": segmentation_method,
            "biome": biome,
            "ndvi_threshold": ndvi_threshold,
            "cloud_cover_threshold": settings.CLOUD_COVER_THRESHOLD,
            "sequestration_rate": seq_rate,
            "scenes_per_year": settings.SCENES_PER_YEAR,
            "annual_offset_tco2": verra_offset,
            "annual_offset_source": offset_source,
        }
        mlflow_metrics = {
            "iou_score": iou_score,
            "f1_score": f1_score,
            "forest_loss_ha": loss_ha,
            "forest_loss_pct": loss_pct,
            "risk_score": risk.get("risk_score"),
            "unusable_pixel_pct": unusable_pct,
        }
        # FIX (pipeline.py bug 3): mlflow_metrics may contain None values (e.g.
        # iou_score and f1_score when Prithvi is not used). mlflow.log_metrics()
        # rejects None — filter them out before logging.
        mlflow_metrics_clean = {
            k: v for k, v in mlflow_metrics.items() if v is not None
        }

        artifacts = [p for p in (loss_png_path, ndvi_png_path) if p.exists()]
        mlflow_run_id = None
        mlflow_tracking_uri = None
        try:
            mlflow_run_id, mlflow_tracking_uri = log_run(
                mlflow_params,
                mlflow_metrics_clean,
                artifacts,
                run_name=project_id,
                tags={
                    "project_id": project_id,
                    "status": "complete",
                    "risk_flag": risk.get("risk_flag"),
                },
                extra_json={
                    "project_id": project_id,
                    "warnings": warnings,
                    "diagnostics": diagnostics,
                    "risk": risk,
                },
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"MLflow logging failed: {exc}")

        # ── 13. Write final result ────────────────────────────────────────────
        results_store[project_id] = {
            "run_id": run_id,
            "project_id": project_id,
            "status": "complete",
            "created_at": created_at,
            "start_year": start_year,
            "end_year": end_year,
            "segmentation_method": segmentation_method,
            "biome": biome,
            "ndvi_threshold_used": ndvi_threshold,
            "sequestration_rate_used": seq_rate,
            "forest_loss_ha": loss_ha,
            "forest_loss_pct": loss_pct,
            "risk_score": risk.get("risk_score"),
            "risk_flag": risk.get("risk_flag"),
            "iou_score": iou_score,
            "f1_score": f1_score,
            "ndvi_before_mean": ndvi_before_stats["mean"],
            "ndvi_after_mean": ndvi_after_stats["mean"],
            # FIX (pipeline.py bug 4): URL used /static/{project_id}/... but
            # main.py mounts RESULTS_DIR as /static. The clipped PNGs are saved
            # under RESULTS_DIR/{project_id}/, so the correct URL is:
            # /static/{project_id}/forest_loss.png — this was already correct,
            # but note RESULTS_DIR must equal the directory mounted at /static.
            "forest_loss_map_url": (
                f"/static/{project_id}/forest_loss.png" if ndvi_valid else None
            ),
            "ndvi_overlay_url": f"/static/{project_id}/ndvi_overlay.png",
            "mlflow_run_id": mlflow_run_id,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "warnings": warnings,
            "diagnostics": diagnostics,
        }
        update_run(
            run_id,
            status="complete",
            risk_score=risk.get("risk_score"),
            warnings=warnings,
        )

    except Exception:  # noqa: BLE001
        results_store[project_id] = {
            "run_id": run_id,
            "status": "failed",
            "project_id": project_id,
            "created_at": created_at,
            "error": traceback.format_exc(),
            "warnings": warnings,
            "diagnostics": diagnostics,
        }
        update_run(run_id, status="failed", warnings=warnings)
