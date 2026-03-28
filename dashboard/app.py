"""
Carbon Project Land Cover Monitor — Streamlit Dashboard
Stage 8: Complete dashboard application
"""

import os
import json
import time
import requests
import pandas as pd
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Carbon Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #238636;
    color: #ffffff !important;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    width: 100%;
    padding: 0.6rem;
    margin-top: 0.5rem;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #2ea043;
}

/* Main area */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Risk badge */
.badge-HIGH {
    background: #da3633;
    color: #fff;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    display: inline-block;
}
.badge-LOW {
    background: #238636;
    color: #fff;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    display: inline-block;
}
.badge-DATA_MISSING {
    background: #6e7681;
    color: #fff;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    display: inline-block;
}

/* Section header */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.3rem;
}

/* Metric label override */
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None
if "result" not in st.session_state:
    st.session_state.result = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def post_analyze(
    project_id: str,
    geojson: dict,
    start_year: int,
    end_year: int,
    annual_offset_tco2: float | None,
):
    url = f"{API_BASE}/projects/{project_id}/analyze"
    payload = {
        "geojson": geojson,
        "start_year": start_year,
        "end_year": end_year,
        "annual_offset_tco2": annual_offset_tco2,
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_results(project_id: str):
    url = f"{API_BASE}/projects/{project_id}/results"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def absolute_url(relative_url: str) -> str:
    """Prepend API base if URL is relative."""
    if relative_url and not relative_url.startswith("http"):
        return f"{API_BASE}{relative_url}"
    return relative_url


def risk_badge(flag: str) -> str:
    cls = f"badge-{flag}" if flag in ("HIGH", "LOW", "DATA_MISSING") else "badge-DATA_MISSING"
    label = flag if flag else "—"
    return f'<span class="{cls}">{label}</span>'


def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def mlflow_backend_hint(tracking_uri: str | None) -> str:
    if not tracking_uri:
        return "MLflow backend not reported by the pipeline."
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        return f"Run stored in SQLite backend: `{db_path}`"
    if tracking_uri.startswith("file:///"):
        store_path = tracking_uri.removeprefix("file:///")
        return (
            f"Run stored in file backend: `{store_path}`. "
            f"Start UI with `mlflow ui --backend-store-uri \"{tracking_uri}\"`."
        )
    return f"MLflow tracking URI: `{tracking_uri}`"


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Carbon Monitor")
    st.markdown('<p style="color:#8b949e;font-size:0.78rem;margin-top:-0.5rem;">Land Cover Change Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    project_id = st.text_input("Project ID", placeholder="VCS-1234", key="project_id_input")

    uploaded_file = st.file_uploader(
        "Upload GeoJSON",
        type=["geojson", "json"],
        help="Upload the project boundary as a GeoJSON FeatureCollection",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=2013, max_value=2024, value=2020, step=1)
    with col2:
        end_year = st.number_input("End Year", min_value=2014, max_value=2025, value=2023, step=1)

    annual_offset_tco2 = st.number_input(
        "Annual Offset (tCO2/yr)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        help="Optional project metadata used by risk scoring. Leave 0 if unknown.",
    )

    run_clicked = st.button("▶  Run Analysis")

    if run_clicked:
        if not project_id:
            st.error("Enter a Project ID first.")
        elif not uploaded_file:
            st.error("Upload a GeoJSON file first.")
        elif start_year >= end_year:
            st.error("Start Year must be before End Year.")
        else:
            try:
                geojson_data = json.load(uploaded_file)
                post_analyze(
                    project_id,
                    geojson_data,
                    int(start_year),
                    int(end_year),
                    float(annual_offset_tco2) if annual_offset_tco2 > 0 else None,
                )
                st.session_state.active_project_id = project_id
                st.session_state.result = None
                st.success(f"Queued: {project_id}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API at " + API_BASE + ". Is the server running?")
            except Exception as exc:
                st.error(f"Submission failed: {exc}")

    st.markdown("---")
    st.markdown('<p style="color:#8b949e;font-size:0.72rem;">API: ' + API_BASE + '</p>', unsafe_allow_html=True)


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown("# Carbon Project Land Cover Monitor")
st.markdown('<p style="color:#8b949e;">Satellite-derived forest loss · NDVI analysis · Risk scoring · MLflow tracking</p>', unsafe_allow_html=True)

active_id = st.session_state.active_project_id

# ── SECTION 2 — Status polling ─────────────────────────────────────────────────
if active_id and st.session_state.result is None:
    st.markdown(f'<div class="section-header">Polling: {active_id}</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()

    with st.spinner(f"Running analysis for **{active_id}** …"):
        for attempt in range(120):  # max 600 s
            try:
                data = get_results(active_id)
            except requests.exceptions.ConnectionError:
                status_placeholder.error("Lost connection to API.")
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    status_placeholder.info("Waiting for pipeline to start …")
                else:
                    status_placeholder.error(str(e))
                    break

            status = data.get("status", "unknown")

            if status == "running":
                status_placeholder.info(f"⏳ Analysis running … (poll #{attempt + 1})")
                time.sleep(5)
                continue

            elif status == "failed":
                error_msg = data.get("error", "Unknown error")
                status_placeholder.error(f"❌ Pipeline failed: {error_msg}")
                failed_warnings = data.get("warnings") or []
                for warning in failed_warnings:
                    st.warning(warning)
                diagnostics = data.get("diagnostics")
                if diagnostics:
                    st.json(diagnostics)
                # record in history
                st.session_state.history.append({
                    "Project ID": active_id,
                    "Status": "failed",
                    "Risk Flag": "—",
                })
                st.session_state.active_project_id = None
                break

            elif status == "complete":
                st.session_state.result = data
                # record in history
                st.session_state.history.append({
                    "Project ID": active_id,
                    "Status": "complete",
                    "Risk Flag": data.get("risk_flag", "—"),
                })
                status_placeholder.empty()
                st.rerun()

            else:
                time.sleep(5)

# ── SECTION 3 — Results display ────────────────────────────────────────────────
result = st.session_state.result

if result and result.get("status") == "complete":
    pid = result.get("project_id", active_id)
    st.success(f"✅ Analysis complete — {pid}")

    # Warnings
    warnings = result.get("warnings") or []
    for w in warnings:
        st.warning(w)

    # ── Metadata row
    st.markdown('<div class="section-header">Project Metadata</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Biome", result.get("biome") or "—")
    m2.metric("Segmentation Method", result.get("segmentation_method") or "—")
    m3.metric("NDVI Threshold", fmt(result.get("ndvi_threshold_used"), 2))
    m4.metric("Seq. Rate (tCO₂/ha/yr)", fmt(result.get("sequestration_rate_used"), 1))

    # ── Risk panel
    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 1, 2])
    risk_score = result.get("risk_score")
    risk_flag = result.get("risk_flag") or "DATA_MISSING"
    r1.metric("Risk Score", fmt(risk_score, 4))
    with r2:
        st.markdown("**Risk Flag**")
        st.markdown(risk_badge(risk_flag), unsafe_allow_html=True)
    with r3:
        if risk_flag == "HIGH":
            st.error("⚠️ Risk exceeds threshold — potential carbon reversal detected.")
        elif risk_flag == "LOW":
            st.info("✅ Risk is within acceptable bounds.")
        else:
            st.warning("⚪ Insufficient data to compute risk score.")

    # ── Metrics table
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame([
        {"Metric": "IoU Score",             "Value": fmt(result.get("iou_score"), 4)},
        {"Metric": "F1 Score",              "Value": fmt(result.get("f1_score"), 4)},
        {"Metric": "Forest Loss (ha)",      "Value": fmt(result.get("forest_loss_ha"), 2)},
        {"Metric": "Forest Loss (%)",       "Value": fmt(result.get("forest_loss_pct"), 2)},
        {"Metric": "NDVI Before (mean)",    "Value": fmt(result.get("ndvi_before_mean"), 4)},
        {"Metric": "NDVI After (mean)",     "Value": fmt(result.get("ndvi_after_mean"), 4)},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Forest loss map
    st.markdown('<div class="section-header">Forest Loss Map</div>', unsafe_allow_html=True)
    fmap_url = result.get("forest_loss_map_url")
    if fmap_url:
        st.image(absolute_url(fmap_url), caption="Red overlay = detected forest loss", use_column_width=True)
    else:
        st.info("Forest loss map not available.")

    # ── NDVI overlay
    st.markdown('<div class="section-header">NDVI Overlay</div>', unsafe_allow_html=True)
    ndvi_url = result.get("ndvi_overlay_url")
    if ndvi_url:
        st.image(absolute_url(ndvi_url), caption="NDVI before / after overlay", use_column_width=True)
    else:
        st.info("NDVI overlay not available.")

    # ── MLflow link
    st.markdown('<div class="section-header">MLflow Run</div>', unsafe_allow_html=True)
    mlflow_run_id = result.get("mlflow_run_id")
    mlflow_tracking_uri = result.get("mlflow_tracking_uri")
    if mlflow_run_id:
        st.markdown(
            f'📊 [Open MLflow run `{mlflow_run_id[:8]}…`](http://localhost:5000) '
            f'— full metrics, params, and artifacts',
            unsafe_allow_html=False,
        )
        st.caption(mlflow_backend_hint(mlflow_tracking_uri))
    else:
        st.markdown("📊 [Open MLflow UI](http://localhost:5000)")
        st.caption(mlflow_backend_hint(mlflow_tracking_uri))

elif active_id is None and st.session_state.result is None:
    # No job submitted yet — show landing state
    st.markdown("""
    <div style="padding:3rem 0;text-align:center;color:#8b949e;">
        <div style="font-size:3rem;margin-bottom:1rem;">🌍</div>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;letter-spacing:0.1em;">
            UPLOAD A GEOJSON AND CLICK RUN ANALYSIS TO BEGIN
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── SECTION 4 — Past results ───────────────────────────────────────────────────
with st.expander("Past results this session"):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No analyses submitted yet in this session.")
