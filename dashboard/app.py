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
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --bg-main: #f4f7f4;
    --surface: #ffffff;
    --surface-muted: #eef3ef;
    --surface-strong: #e3ece5;
    --text-main: #17211b;
    --text-muted: #526157;
    --border: #d7e0d9;
    --accent: #2f5d50;
    --accent-soft: #dce9e2;
    --accent-warm: #9a6a3a;
    --loss: #b14f3b;
    --loss-soft: #f4dfdb;
    --ndvi-high: #2f6b45;
    --ndvi-mid: #93af67;
    --ndvi-low: #d2b55b;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-main);
}

p, li, label, span, div {
    color: inherit;
}

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(180deg, #edf3ef 0%, #f7faf8 18%, #f4f7f4 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #193028 0%, #111d18 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] * {
    color: #eaf0ec !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #eaf0ec !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #d7e4dc;
    color: #183028 !important;
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.14);
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    width: 100%;
    padding: 0.6rem;
    margin-top: 0.5rem;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #e7efe9;
    border-color: rgba(255, 255, 255, 0.22);
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid var(--border);
    color: #17211b !important;
}

/* Sidebar inputs need explicit dark text since sidebar forces white on everything */
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input::placeholder {
    color: #17211b !important;
    -webkit-text-fill-color: #17211b !important;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input::placeholder {
    color: #6b7c72 !important;
    -webkit-text-fill-color: #6b7c72 !important;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] label,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #f3f7f4 !important;
    font-weight: 600;
}

/* Main area */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.main .block-container,
.main .block-container p,
.main .block-container li,
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container h4,
.main .block-container label,
.main .block-container strong,
.main .block-container div,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stWidgetLabel"],
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: var(--text-main) !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

.hero-panel h1,
.hero-panel p,
.hero-panel div {
    color: var(--text-main) !important;
}

h1 {
    color: var(--text-main);
    letter-spacing: -0.02em;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface-muted) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    box-shadow: 0 8px 24px rgba(35, 52, 42, 0.05);
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.75);
}

button[data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.72);
    color: var(--text-muted);
    padding: 0.3rem 0.9rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent-soft);
    border-color: rgba(47, 93, 80, 0.28);
    color: #ffffff !important;
    color: var(--accent) !important;
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 14px 30px rgba(29, 44, 36, 0.08);
}

/* Risk badge */
.badge-HIGH {
    background: var(--loss);
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
    background: var(--accent);
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
    background: #728177;
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
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
}

.hero-panel {
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(230,239,233,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 12px 30px rgba(35, 52, 42, 0.06);
}

.hero-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.45rem;
}

.hero-subtitle {
    color: var(--text-muted);
    margin: 0;
}

.legend-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(242,246,243,0.98) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.8rem;
}

.legend-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.7rem;
}

.legend-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.5rem;
    color: var(--text-main);
}

.legend-swatch {
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    flex: 0 0 16px;
}

.map-note {
    color: var(--text-muted);
    font-size: 0.94rem;
    margin: 0.4rem 0 0.8rem 0;
}

/* Metric label override */
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem !important;
    color: var(--text-main) !important;
}
[data-testid="stMetricDelta"] {
    color: var(--accent) !important;
}

.download-progress-card {
    background: rgba(220, 233, 226, 0.72);
    border: 1px solid #c4d9cc;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 0.3rem 0 0.5rem 0;
    color: #17211b !important;
}

.download-progress-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #214b3e !important;
    margin-bottom: 0.45rem;
}

.download-progress-summary {
    font-size: 0.96rem;
    color: #17211b !important;
    margin-bottom: 0.35rem;
}

.download-progress-current {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #2e443b !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.download-progress-chip {
    display: inline-block;
    margin-left: 0.45rem;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    background: #dce9e2;
    color: #214b3e !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}
</style>
""",
    unsafe_allow_html=True,
)


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
    resp = requests.get(url, timeout=(5, 30))
    resp.raise_for_status()
    return resp.json()


def absolute_url(relative_url: str) -> str:
    """Prepend API base if URL is relative."""
    if relative_url and not relative_url.startswith("http"):
        return f"{API_BASE}{relative_url}"
    return relative_url


def risk_badge(flag: str) -> str:
    cls = (
        f"badge-{flag}"
        if flag in ("HIGH", "LOW", "DATA_MISSING")
        else "badge-DATA_MISSING"
    )
    label = flag if flag else "—"
    return f'<span class="{cls}">{label}</span>'


def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def fmt_pct(val, decimals=1):
    if val is None:
        return "-"
    return f"{val:.{decimals}f}%"


def fmt_delta(val, decimals=4, unit=""):
    if val is None:
        return "-"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}{unit}"


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
            f'Start UI with `mlflow ui --backend-store-uri "{tracking_uri}"`.'
        )
    return f"MLflow tracking URI: `{tracking_uri}`"


def mlflow_run_url(run_id: str | None, experiment_id: str | None) -> str:
    if run_id and experiment_id:
        return f"http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"
    if experiment_id:
        return f"http://localhost:5000/#/experiments/{experiment_id}"
    return "http://localhost:5000"


def confidence_label(score: float | None) -> tuple[str, str]:
    if score is None:
        return "Not available", "No benchmark score was produced for this run."
    if score >= 0.75:
        return "High", "The detected loss map aligns well with the reference data."
    if score >= 0.5:
        return (
            "Moderate",
            "The loss map is directionally useful, but should be reviewed alongside the imagery.",
        )
    return (
        "Low",
        "The loss map may be noisy or incomplete, so rely more heavily on the visual overlays.",
    )


def risk_summary(flag: str, score: float | None) -> str:
    if flag == "HIGH":
        score_text = fmt(score, 4)
        return (
            f"High reversal risk. The pipeline flagged this project because the computed risk score "
            f"({score_text}) crossed the alert threshold."
        )
    if flag == "LOW":
        score_text = fmt(score, 4)
        return f"Low reversal risk. The computed risk score ({score_text}) stayed within the acceptable range."
    return "Risk could not be scored confidently because the pipeline did not have enough usable data."


def build_takeaways(result: dict) -> list[str]:
    takeaways: list[str] = []

    loss_pct = result.get("forest_loss_pct")
    loss_ha = result.get("forest_loss_ha")
    ndvi_before = result.get("ndvi_before_mean")
    ndvi_after = result.get("ndvi_after_mean")
    risk_flag = result.get("risk_flag") or "DATA_MISSING"
    iou = result.get("iou_score")

    if loss_pct is not None and loss_ha is not None:
        takeaways.append(
            f"Estimated forest loss is {loss_ha:.2f} ha, covering {loss_pct:.2f}% of the analyzed area."
        )

    if ndvi_before is not None and ndvi_after is not None:
        ndvi_change = ndvi_after - ndvi_before
        direction = "decreased" if ndvi_change < 0 else "increased"
        takeaways.append(
            f"Mean NDVI {direction} by {abs(ndvi_change):.4f} between the start and end years."
        )

    if risk_flag == "HIGH":
        takeaways.append(
            "The project was flagged as HIGH risk, so this run deserves follow-up review."
        )
    elif risk_flag == "LOW":
        takeaways.append("The project stayed in the LOW risk band for this run.")
    else:
        takeaways.append(
            "Risk scoring was inconclusive because required inputs were missing or insufficient."
        )

    if iou is not None:
        confidence, explanation = confidence_label(iou)
        takeaways.append(f"Detection confidence is {confidence.lower()}: {explanation}")

    return takeaways


def render_legend_card(title: str, items: list[tuple[str, str]]) -> None:
    rows = "".join(
        (
            f'<div class="legend-row">'
            f'<span class="legend-swatch" style="background:{color};"></span>'
            f"<span>{label}</span>"
            f"</div>"
        )
        for color, label in items
    )
    st.markdown(
        f'<div class="legend-card"><div class="legend-title">{title}</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def render_download_progress_card(
    dl_complete: int,
    dl_total: int,
    dl_cached: int,
    dl_current: str | None,
) -> None:
    cached_chip = (
        f'<span class="download-progress-chip">{dl_cached} cached</span>'
        if dl_cached > 0
        else ""
    )
    current_line = (
        f'<div class="download-progress-current">Current: {dl_current}</div>'
        if dl_current
        else ""
    )
    st.markdown(
        (
            '<div class="download-progress-card">'
            '<div class="download-progress-title">Tile Download Progress</div>'
            f'<div class="download-progress-summary"><strong>{dl_complete} of {dl_total}</strong> scenes ready{cached_chip}</div>'
            f"{current_line}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Carbon Monitor")
    st.markdown(
        '<p style="color:#dbe8e1;font-size:0.78rem;margin-top:-0.5rem;">Land Cover Change Analysis</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    project_id = st.text_input(
        "Project ID", placeholder="VCS-1234", key="project_id_input"
    )

    uploaded_file = st.file_uploader(
        "Upload GeoJSON",
        type=["geojson", "json"],
        help="Upload the project boundary as a GeoJSON FeatureCollection",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year", min_value=2013, max_value=2024, value=2020, step=1
        )
    with col2:
        end_year = st.number_input(
            "End Year", min_value=2014, max_value=2025, value=2023, step=1
        )

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
    st.markdown(
        '<p style="color:#dbe8e1;font-size:0.72rem;">API: ' + API_BASE + "</p>",
        unsafe_allow_html=True,
    )


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-kicker">Carbon Monitor</div>
        <h1 style="margin:0 0 0.35rem 0;">Carbon Project Land Cover Monitor</h1>
        <p class="hero-subtitle">Satellite-derived forest loss, NDVI trend interpretation, risk scoring, and MLflow-backed analysis review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

active_id = st.session_state.active_project_id

# ── SECTION 2 — Status polling ─────────────────────────────────────────────────
# Poll count lives in session_state so each rerun increments it without a
# blocking loop — this prevents Streamlit's watchdog from timing out on long runs.
if "poll_count" not in st.session_state:
    st.session_state.poll_count = 0

if active_id and st.session_state.result is None:
    st.markdown(
        f'<div class="section-header">Polling: {active_id}</div>',
        unsafe_allow_html=True,
    )

    try:
        data = get_results(active_id)
    except requests.exceptions.ConnectionError:
        st.error("Lost connection to API. Is the server still running?")
        data = None
    except requests.exceptions.ReadTimeout:
        data = {"status": "running"}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            data = {"status": "running"}
        else:
            st.error(str(e))
            data = None

    if data is not None:
        status = data.get("status", "unknown")

        if status == "running":
            st.session_state.poll_count += 1
            poll_n = st.session_state.poll_count

            # ── Progress bar ──────────────────────────────────────────────
            step = data.get("step", 0)
            total = data.get("total_steps", 8)
            stage = data.get("stage", "Working…")

            progress_pct = step / total if total else 0.0
            # Clamp to [0.02, 0.95] so bar is always visible and never
            # falsely shows 100% while still running
            progress_pct = max(0.02, min(0.95, progress_pct))

            st.progress(progress_pct, text=None)

            # Stage label card
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.82);
                    border: 1px solid #d7e0d9;
                    border-radius: 12px;
                    padding: 0.85rem 1.1rem;
                    margin: 0.5rem 0 0.3rem 0;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                ">
                    <span style="font-size:1.4rem;">⏳</span>
                    <div>
                        <div style="
                            font-family:'IBM Plex Mono',monospace;
                            font-size:0.68rem;
                            letter-spacing:0.12em;
                            text-transform:uppercase;
                            color:#526157;
                            margin-bottom:0.2rem;
                        ">Step {step} of {total} · poll #{poll_n}</div>
                        <div style="font-size:0.97rem;color:#17211b;">{stage}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Download sub-progress panel (only visible during step 3) ──
            dl_total = data.get("downloads_total", 0)
            dl_done = data.get("downloads_done", 0)
            dl_cached = data.get("downloads_cached", 0)
            dl_current = data.get("current_download")

            if step == 3 and dl_total > 0:
                dl_complete = dl_done + dl_cached
                dl_pct = dl_complete / dl_total if dl_total else 0.0
                dl_pct = max(0.01, min(1.0, dl_pct))

                # Mini progress bar for downloads
                st.progress(dl_pct)

                # Build status chips
                cached_chip = (
                    f'<span style="background:#dce9e2;color:#2f5d50;'
                    f'padding:2px 9px;border-radius:12px;font-size:0.75rem;'
                    f'font-family:IBM Plex Mono,monospace;margin-left:6px;">'
                    f'⚡ {dl_cached} cached</span>'
                ) if dl_cached > 0 else ""

                current_label = (
                    f'<div style="margin-top:0.35rem;font-size:0.78rem;'
                    f'color:#526157;font-family:IBM Plex Mono,monospace;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                    f'max-width:560px;">'
                    f'↓ {dl_current}</div>'
                ) if dl_current else ""

                render_download_progress_card(
                    dl_complete,
                    dl_total,
                    dl_cached,
                    dl_current,
                )

            # Auto-rerun after 5 s — no blocking sleep in a loop
            time.sleep(5)
            st.rerun()

        elif status == "failed":
            st.session_state.poll_count = 0
            error_msg = data.get("error", "Unknown error")
            st.error(f"❌ Pipeline failed: {error_msg}")
            failed_warnings = data.get("warnings") or []
            for warning in failed_warnings:
                st.warning(warning)
            diagnostics = data.get("diagnostics")
            if diagnostics:
                st.json(diagnostics)
            st.session_state.history.append(
                {"Project ID": active_id, "Status": "failed", "Risk Flag": "—"}
            )
            st.session_state.active_project_id = None

        elif status == "complete":
            st.session_state.poll_count = 0
            st.session_state.result = data
            st.session_state.history.append(
                {
                    "Project ID": active_id,
                    "Status": "complete",
                    "Risk Flag": data.get("risk_flag", "—"),
                }
            )
            st.rerun()

# ── SECTION 3 — Results display ────────────────────────────────────────────────
result = st.session_state.result

if result and result.get("status") == "complete":
    pid = result.get("project_id", active_id)
    st.success(f"✅ Analysis complete — {pid}")

    ndvi_before = result.get("ndvi_before_mean")
    ndvi_after = result.get("ndvi_after_mean")
    ndvi_change = (
        ndvi_after - ndvi_before
        if ndvi_before is not None and ndvi_after is not None
        else None
    )
    iou_score = result.get("iou_score")
    f1_score = result.get("f1_score")
    loss_ha = result.get("forest_loss_ha")
    loss_pct = result.get("forest_loss_pct")
    risk_score = result.get("risk_score")
    risk_flag = result.get("risk_flag") or "DATA_MISSING"
    confidence, confidence_explainer = confidence_label(iou_score)

    # Warnings
    warnings = result.get("warnings") or []
    for w in warnings:
        st.warning(w)

    # â”€â”€ Executive summary
    st.markdown('<div class="section-header">At a Glance</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Forest Loss Detected",
        fmt(loss_ha, 2) + (" ha" if loss_ha is not None else ""),
        delta=fmt_pct(loss_pct, 2) if loss_pct is not None else None,
        help="Absolute area flagged as forest loss, with the percentage of analyzed area shown as the delta.",
    )
    s2.metric(
        "Mean NDVI in End Year",
        fmt(ndvi_after, 4),
        delta=fmt_delta(ndvi_change, 4),
        help="Delta is the change from start-year NDVI to end-year NDVI. Negative values suggest reduced vegetation greenness.",
    )
    s3.metric(
        "Detection Confidence",
        confidence,
        delta=f"IoU {fmt(iou_score, 4)}" if iou_score is not None else None,
        help="Confidence is a plain-language interpretation of the IoU benchmark score.",
    )
    st.caption("Quick read: " + " ".join(build_takeaways(result)))

    # ── Metadata row
    st.markdown(
        '<div class="section-header">Project Metadata</div>', unsafe_allow_html=True
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Biome", result.get("biome") or "—")
    m2.metric("Segmentation Method", result.get("segmentation_method") or "—")
    m3.metric("NDVI Threshold", fmt(result.get("ndvi_threshold_used"), 2))
    m4.metric("Seq. Rate (tCO₂/ha/yr)", fmt(result.get("sequestration_rate_used"), 1))

    # ── Risk panel
    st.markdown(
        '<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True
    )
    r1, r2, r3 = st.columns([1, 1, 2])
    r1.metric("Risk Score", fmt(risk_score, 4))
    with r2:
        st.markdown("**Risk Flag**")
        st.markdown(risk_badge(risk_flag), unsafe_allow_html=True)
    with r3:
        if risk_flag == "HIGH":
            st.error(risk_summary(risk_flag, risk_score))
        elif risk_flag == "LOW":
            st.info(risk_summary(risk_flag, risk_score))
        else:
            st.warning(risk_summary(risk_flag, risk_score))

    # ── Metrics table
    with st.expander("How to read the risk output"):
        st.markdown(
            """
            - `HIGH` means the computed score crossed the pipeline's alert threshold and may indicate reversal risk.
            - `LOW` means the score stayed below the alert threshold for this run.
            - `DATA_MISSING` means the pipeline could not score risk confidently from the available data.
            - Review the forest-loss and NDVI overlays alongside the risk flag before making a project decision.
            """
        )

    st.markdown(
        '<div class="section-header">Detailed Metrics</div>', unsafe_allow_html=True
    )
    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "Forest loss area",
                "Value": fmt(loss_ha, 2) + (" ha" if loss_ha is not None else ""),
                "Interpretation": "Higher values mean more area was classified as forest loss inside the project boundary.",
            },
            {
                "Metric": "Forest loss share",
                "Value": fmt_pct(loss_pct, 2),
                "Interpretation": "Shows how much of the analyzed area was classified as forest loss.",
            },
            {
                "Metric": "Mean NDVI before",
                "Value": fmt(ndvi_before, 4),
                "Interpretation": "Baseline vegetation greenness in the start year.",
            },
            {
                "Metric": "Mean NDVI after",
                "Value": fmt(ndvi_after, 4),
                "Interpretation": "Vegetation greenness in the end year.",
            },
            {
                "Metric": "NDVI change",
                "Value": fmt_delta(ndvi_change, 4),
                "Interpretation": "Negative values suggest vegetation declined; positive values suggest recovery or denser vegetation.",
            },
            {
                "Metric": "IoU score",
                "Value": fmt(iou_score, 4),
                "Interpretation": "Agreement between detected loss and reference data. Closer to 1 is better.",
            },
            {
                "Metric": "F1 score",
                "Value": fmt(f1_score, 4),
                "Interpretation": "Balances missed detections and false alarms. Closer to 1 is better.",
            },
        ]
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.caption(f"Confidence note: {confidence_explainer}")

    imagery_tab, methods_tab = st.tabs(["Visual Evidence", "How To Read These Outputs"])

    with imagery_tab:
        l1, l2 = st.columns(2)
        with l1:
            render_legend_card(
                "Forest Loss Map Legend",
                [
                    ("#b14f3b", "Detected forest loss hotspot"),
                    ("#f0efe9", "Background area with no detected loss signal"),
                ],
            )
        with l2:
            render_legend_card(
                "NDVI Overlay Legend",
                [
                    ("#2f6b45", "Higher vegetation greenness"),
                    ("#93af67", "Moderate greenness or mixed vegetation"),
                    ("#d2b55b", "Lower greenness or stressed vegetation"),
                    ("#9a4f3b", "Vegetation decline in the change panel"),
                ],
            )

        st.markdown(
            '<div class="section-header">Forest Loss Map</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<p class="map-note">Use the loss map to focus on concentrated red zones. Small isolated patches can be less reliable than contiguous clusters.</p>',
            unsafe_allow_html=True,
        )
        fmap_url = result.get("forest_loss_map_url")
        if fmap_url:
            st.image(
                absolute_url(fmap_url),
                caption="Red overlay marks pixels classified as forest loss.",
                use_container_width=True,
            )
        else:
            st.info("Forest loss map not available.")

        st.markdown(
            '<div class="section-header">NDVI Overlay</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<p class="map-note">The updated image now shows before, after, and a dedicated change panel. Read brown as vegetation decline, off-white as stable conditions, and green as stronger vegetation in the end year.</p>',
            unsafe_allow_html=True,
        )
        ndvi_url = result.get("ndvi_overlay_url")
        if ndvi_url:
            st.image(
                absolute_url(ndvi_url),
                caption="NDVI comparison includes before, after, and change views for faster interpretation.",
                use_container_width=True,
            )
        else:
            st.info("NDVI overlay not available.")

    with methods_tab:
        st.markdown(
            """
            - `Forest loss map`: use this to see where loss was detected spatially. Clusters are usually more meaningful than isolated pixels.
            - `NDVI overlay`: compare vegetation greenness between years. Lower NDVI often indicates reduced vegetation cover or stress.
            - `IoU` and `F1`: these are quality checks against reference labels. Higher scores mean the automated detection is more trustworthy.
            - `Risk flag`: this is a decision aid, not a final verdict. Use it together with the maps and any project context you have.
            """
        )

    # ── MLflow link
    st.markdown('<div class="section-header">MLflow Run</div>', unsafe_allow_html=True)
    mlflow_run_id = result.get("mlflow_run_id")
    mlflow_tracking_uri = result.get("mlflow_tracking_uri")
    mlflow_experiment_id = result.get("mlflow_experiment_id")
    if mlflow_run_id:
        st.markdown(
            f"📊 [Open MLflow run `{mlflow_run_id[:8]}…`]({mlflow_run_url(mlflow_run_id, mlflow_experiment_id)}) "
            f"— full metrics, params, and artifacts",
            unsafe_allow_html=False,
        )
        st.caption(mlflow_backend_hint(mlflow_tracking_uri))
    else:
        st.markdown(
            f"📊 [Open MLflow UI]({mlflow_run_url(None, mlflow_experiment_id)})"
        )
        st.caption(mlflow_backend_hint(mlflow_tracking_uri))

elif active_id is None and st.session_state.result is None:
    # No job submitted yet — show landing state
    st.markdown(
        """
    <div style="padding:3rem 0;text-align:center;color:#8b949e;">
        <div style="font-size:3rem;margin-bottom:1rem;">🌍</div>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;letter-spacing:0.1em;">
            UPLOAD A GEOJSON AND CLICK RUN ANALYSIS TO BEGIN
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── SECTION 4 — Past results ───────────────────────────────────────────────────
with st.expander("Past results this session"):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No analyses submitted yet in this session.")
