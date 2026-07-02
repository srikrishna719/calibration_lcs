"""Premium 8-step Streamlit UI for the Air Quality Sensor Calibration Lab.

Provides a refined, research-grade workflow with extensive user controls,
dark-themed premium styling, and interactive Plotly visualisations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that keeps numeric types and converts only truly unserializable objects to str."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def _cfg_to_json(config: Any) -> str:
    """Serialize config to JSON preserving int/float/bool types."""
    return json.dumps(config, cls=_SafeEncoder)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.predict import predict_with_model
from models.model_registry import MODEL_GROUPS, MODEL_DISPLAY_NAMES
from modules.drift_analysis import (
    create_predicted_vs_actual_figure,
    create_qq_plot,
    create_residual_histogram,
    create_residual_vs_predicted_figure,
    generate_post_analysis_outputs,
)
from modules.download_helpers import render_chart_download, render_df_download
from modules.normalization import get_normalization_summary, normalize_dataset
from modules.diagnostics import compute_vif, shapiro_wilk_test
from modules.plots import (
    create_bland_altman_plot,
    create_multi_model_metrics_bar,
    create_multi_model_scatter,
    create_multi_model_timeseries,
    create_scatter_with_fit,
)
from pipeline.run_pipeline import (
    build_export_bundle,
    load_config,
    load_input_data,
    run_alignment_stage,
    run_eda_stage,
    run_modeling_stage,
    run_post_analysis_stage,
    run_preprocessing_stage,
    train_on_prepared_dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
DEFAULT_REF = SAMPLE_DATA_DIR / "reference_dataset.csv"
DEFAULT_LCS = SAMPLE_DATA_DIR / "low_cost_sensor_dataset.csv"

STEPS = [
    "📤 Upload Data",
    "🧹 Preprocessing",
    "🔗 Alignment",
    "📊 EDA",
    "🎯 Variable Selection",
    "🧬 Feature Engineering",
    "📏 Normalization",
    "🤖 Modelling",
    "✅ Validation & Results",
    "📐 Statistical Diagnostics",
    "🔬 Residual Analysis",
    "💾 Export",
    "📖 README",
]

STEP_KEYS = [
    "upload", "preprocessing", "alignment", "eda",
    "variable_selection", "feature_engineering", "normalization",
    "modelling", "validation_results", "statistical_diagnostics",
    "residual_analysis", "export", "readme",
]

# Preprocessing UI options (full display names mapped to backend aliases)
MISSING_OPTIONS = [
    "No Treatment",
    "Forward Fill",
    "Backward Fill",
    "Linear Interpolation",
    "Linear Interpolation + Forward Fill",
    "Linear Interpolation + Backward Fill",
    "Drop Missing Rows",
]

OUTLIER_OPTIONS = ["iqr", "zscore", "none"]

# Maps internal/aliased missing-strategy keys back to the display labels above,
# so a previously-saved config selects the right option in the selectbox.
_INTERNAL_TO_DISPLAY = {
    "none": "No Treatment",
    "no treatment": "No Treatment",
    "forward_fill": "Forward Fill",
    "ffill": "Forward Fill",
    "backward_fill": "Backward Fill",
    "bfill": "Backward Fill",
    "linear_interpolation": "Linear Interpolation",
    "interpolate": "Linear Interpolation",
    "linear_interpolation_forward_fill": "Linear Interpolation + Forward Fill",
    "interpolate_ffill": "Linear Interpolation + Forward Fill",
    "linear_interpolation_backward_fill": "Linear Interpolation + Backward Fill",
    "interpolate_bfill": "Linear Interpolation + Backward Fill",
    "drop_missing_rows": "Drop Missing Rows",
    "drop": "Drop Missing Rows",
}

# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(67, 56, 202, 0.3);
}
.main-header h1 {
    color: #e0e7ff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #a5b4fc;
    font-size: 0.9rem;
    margin: 0.3rem 0 0 0;
}

/* Step progress */
.step-progress {
    display: flex;
    gap: 4px;
    margin-bottom: 1.2rem;
}
.step-dot {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: #1e1b4b;
    transition: background 0.3s ease;
}
.step-dot.active {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.5);
}
.step-dot.done {
    background: #10b981;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
}
.metric-card .label {
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    color: #e0e7ff;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.25rem;
}
.metric-card .value.good { color: #34d399; }
.metric-card .value.warn { color: #fbbf24; }
.metric-card .value.bad  { color: #f87171; }

/* Section card */
.section-card {
    background: rgba(30, 27, 75, 0.4);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* Sidebar refinements */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0a2e 0%, #1e1b4b 100%);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    font-weight: 500;
    padding: 0.4rem 0;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

/* Dataframes */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* Info cards */
.info-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-bottom: 0.3rem;
}

/* ============================================================
   Fix: Streamlit 1.56 expander icon fallback text overlap
   Root cause: 'keyboard_arrow_right'/'keyboard_arrow_down' text
   renders as literal characters when Material Icons font fails.
   Structure: details > summary > span > span > span (icon text)
   ============================================================ */

/* The icon span — completely suppress fallback text, show only the glyph */
[data-testid="stExpander"] details summary span span span {
    font-family: 'Material Icons', 'Material Icons Outlined', serif;
    font-size: 0 !important;  /* hide raw text fallback */
    display: inline-block;
    width: 0;
    height: 0;
    overflow: hidden;
    flex-shrink: 0;
}
/* Use parent span to show a clean arrow via CSS */
[data-testid="stExpander"] details summary > span > span:first-child::before {
    content: '▶';
    font-size: 0.7rem;
    color: #a5b4fc;
    display: inline-block;
    transition: transform 0.2s ease;
    margin-right: 2px;
}
[data-testid="stExpander"] details[open] summary > span > span:first-child::before {
    content: '▼';
}

/* The summary row itself: flex layout to prevent overflow bleeding */
[data-testid="stExpander"] details summary > span {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    overflow: hidden;
}

/* Title text container: let it fill remaining space with ellipsis */
[data-testid="stExpander"] details summary > span > div {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    min-width: 0;
}

/* Sidebar label visibility */
section[data-testid="stSidebar"] .stRadio > label {
    color: #c7d2fe !important;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* File uploader button — suppress Material Icons fallback text ('upload' literal) */
/* Structure: button > span > span > span (icon text) + div > p (label text) */
[data-testid="stFileUploader"] button span span span {
    font-size: 0 !important;
    width: 0;
    height: 0;
    overflow: hidden;
    display: inline-block;
}
/* Show a clean upload arrow via the parent span instead */
[data-testid="stFileUploader"] button > span > span:first-child::before {
    content: '⬆';
    font-size: 0.85rem;
    margin-right: 4px;
    display: inline-block;
}
/* Prevent button label text from being clipped */
[data-testid="stFileUploader"] button {
    overflow: visible;
    white-space: nowrap;
}

/* Compact download buttons */
.stDownloadButton > button {
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    min-height: 2rem;
}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    st.markdown(
        '<div class="main-header">'
        '<h1>🌬️ Air Quality Sensor Calibration Lab</h1>'
        '<p>Research-grade calibration pipeline for low-cost air quality sensors</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_step_progress(current_index: int):
    dots = ""
    for i in range(len(STEPS)):
        cls = "done" if i < current_index else ("active" if i == current_index else "")
        dots += f'<div class="step-dot {cls}"></div>'
    st.markdown(f'<div class="step-progress">{dots}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, quality: str = "") -> str:
    cls = f" {quality}" if quality else ""
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value{cls}">{value}</div>'
        f'</div>'
    )


def render_metric_row(metrics: Dict[str, float], keys: List[str], labels: Optional[List[str]] = None):
    labels = labels or keys
    cols = st.columns(len(keys))
    for col, key, label in zip(cols, keys, labels):
        val = metrics.get(key, float("nan"))
        if isinstance(val, float):
            txt = f"{val:.2f}"
            # Color coding
            q = ""
            if key == "r2" or key == "pearson_r":
                q = "good" if val > 0.9 else ("warn" if val > 0.7 else "bad")
            elif key in ("rmse", "mae", "mape"):
                q = "good" if val < 5 else ("warn" if val < 15 else "bad")
            elif key == "bias":
                q = "good" if abs(val) < 2 else ("warn" if abs(val) < 5 else "bad")
            elif key == "slope":
                q = "good" if abs(val - 1.0) < 0.1 else ("warn" if abs(val - 1.0) < 0.3 else "bad")
        else:
            txt, q = str(val), ""
        col.markdown(metric_card(label, txt, q), unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-card"><strong>{title}</strong></div>', unsafe_allow_html=True)


def info_pill(text: str) -> str:
    return f'<span class="info-pill">{text}</span>'


def _model_label(key: str) -> str:
    """Human-readable label for a model registry key."""
    return key.replace("_", " ").title()


def _format_metric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns to 2 decimal places for display."""
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].round(2)
    return out


def _best_row_style(row: pd.Series) -> list:
    """Highlight the best (rank 1) row in a leaderboard."""
    if row.get("rank") == 1 or row.name == 0:
        return ["background-color: rgba(16, 185, 129, 0.15)"] * len(row)
    return [""] * len(row)


def _format_coefficient_table(coef_table) -> pd.DataFrame:
    """Format a coefficient table with proper decimal places."""
    if coef_table is None or (isinstance(coef_table, pd.DataFrame) and coef_table.empty):
        return pd.DataFrame(columns=["Variable", "Coefficient", "Std Error", "t-Statistic", "P-value"])
    df = coef_table.copy()
    for col in ["Coefficient", "Std Error", "t-Statistic"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    if "P-value" in df.columns:
        df["P-value"] = df["P-value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    return df


def _chart_customization(key_prefix: str, default_title: str, default_x: str, default_y: str):
    """Render editable title/axis inputs and return (title, x_label, y_label)."""
    with st.expander("\u2699\ufe0f Chart Customization", expanded=False):
        title = st.text_input("Title", value=default_title, key=f"{key_prefix}_title")
        c1, c2 = st.columns(2)
        x_label = c1.text_input("X-axis label", value=default_x, key=f"{key_prefix}_x")
        y_label = c2.text_input("Y-axis label", value=default_y, key=f"{key_prefix}_y")
    return title, x_label, y_label


def _display_chart_with_downloads(fig, source_df, key: str, filename_prefix: str):
    """Display a Plotly chart with PNG and source-data CSV download buttons."""
    st.plotly_chart(fig, width='stretch')
    render_chart_download(fig, source_df, key=key, filename_prefix=filename_prefix)


def _normalization_summary_tables(summary: pd.DataFrame):
    """Split normalization summary into before/after display tables."""
    before_cols = [c for c in summary.columns if c.startswith("before_") or c == "column"]
    after_cols = [c for c in summary.columns if c.startswith("after_") or c == "column"]
    before_tbl = summary[before_cols].copy() if before_cols else summary.copy()
    after_tbl = summary[after_cols].copy() if after_cols else summary.copy()
    return before_tbl, after_tbl


def _parse_positive_int_list(text: str) -> list:
    """Parse comma-separated positive integers."""
    result = []
    for part in text.split(","):
        part = part.strip()
        if part:
            val = int(part)
            if val <= 0:
                raise ValueError(f"Expected positive integer, got {val}")
            result.append(val)
    return result


# ---------------------------------------------------------------------------
# Caching wrappers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_load_sample():
    return pd.read_csv(DEFAULT_REF), pd.read_csv(DEFAULT_LCS)


@st.cache_data(show_spinner=False)
def cached_config_from_text(text: str, suffix: str):
    if suffix in {".yaml", ".yml"}:
        import yaml
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError("Config must be YAML or JSON.")


@st.cache_data(show_spinner=False)
def cached_load_input(ref_df, sen_df, cfg_text):
    return load_input_data(ref_df, sen_df, json.loads(cfg_text))


@st.cache_data(show_spinner=False)
def cached_preprocessing(ref_df, sen_df, cfg_text):
    return run_preprocessing_stage(ref_df, sen_df, json.loads(cfg_text))


@st.cache_data(show_spinner=False)
def cached_alignment(ref_df, sen_df, cfg_text):
    return run_alignment_stage(ref_df, sen_df, json.loads(cfg_text))


@st.cache_data(show_spinner=False)
def cached_eda(merged_df, cfg_text, raw_df=None):
    return run_eda_stage(merged_df, json.loads(cfg_text), raw_merged_df=raw_df)


@st.cache_data(show_spinner=False)
def cached_modeling(merged_df, cfg_text, feature_subset_json="null"):
    subset = json.loads(feature_subset_json)
    return run_modeling_stage(merged_df, json.loads(cfg_text), feature_subset=subset)


@st.cache_data(show_spinner=False)
def cached_train_prepared(prepared_df, target_column, cfg_text, feature_subset_json="null"):
    subset = json.loads(feature_subset_json)
    return train_on_prepared_dataset(
        prepared_df, target_column, json.loads(cfg_text), feature_subset=subset
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "current_step": STEPS[0],
        "config": None,
        "input_label": None,
        "data_outputs": None,
        "dropped_ref_cols": [],
        "dropped_sen_cols": [],
        "preprocessing_outputs": None,
        "alignment_outputs": None,
        "eda_outputs": None,
        "modeling_outputs": None,
        "selected_target": None,
        "selected_predictors": None,
        "selected_model_name": None,
        "variable_selection_outputs": None,
        "feature_engineering_outputs": None,
        "normalization_outputs": None,
        "residual_analysis_outputs": None,
        "export_bundle": None,
        # Feature engineering extras
        "featured_preview": None,
        "selected_features": None,
        # README editable content
        "readme_content": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v



def _reset_downstream(*keys: str):
    """Nil-out downstream pipeline outputs so stale results aren't carried forward."""
    for k in keys:
        st.session_state[k] = None


# Common downstream reset groups
_DOWNSTREAM_FROM_UPLOAD = (
    "preprocessing_outputs", "alignment_outputs", "eda_outputs",
    "modeling_outputs", "post_analysis_outputs", "export_bundle",
)
_DOWNSTREAM_FROM_PREPROCESSING = (
    "alignment_outputs", "eda_outputs", "modeling_outputs",
    "post_analysis_outputs", "export_bundle",
)
_DOWNSTREAM_FROM_ALIGNMENT = (
    "eda_outputs", "modeling_outputs", "post_analysis_outputs", "export_bundle",
)
_DOWNSTREAM_FROM_MODELING = ("post_analysis_outputs", "export_bundle")


def step_nav() -> str:
    return st.sidebar.radio("Workflow Steps", STEPS, index=STEPS.index(st.session_state.current_step))


def go_next():
    idx = STEPS.index(st.session_state.current_step)
    if idx < len(STEPS) - 1:
        st.session_state.current_step = STEPS[idx + 1]
        st.rerun()


# ---------------------------------------------------------------------------
# Resolve data inputs
# ---------------------------------------------------------------------------

def resolve_inputs(ref_file, sen_file, use_sample):
    if ref_file is not None and sen_file is not None:
        return pd.read_csv(ref_file), pd.read_csv(sen_file), "Uploaded files"
    if use_sample:
        return pd.read_csv(DEFAULT_REF), pd.read_csv(DEFAULT_LCS), "Bundled sample data"
    raise ValueError("Upload both CSVs or enable the sample datasets.")


def resolve_config(uploaded_file):
    if uploaded_file is None:
        return load_config(DEFAULT_CONFIG_PATH)
    suffix = Path(uploaded_file.name).suffix.lower()
    text = uploaded_file.getvalue().decode("utf-8")
    return cached_config_from_text(text, suffix)


# ---------------------------------------------------------------------------
# STEP 1: Upload Data
# ---------------------------------------------------------------------------

def render_upload():
    st.subheader("📤 Upload Data")
    st.markdown(
        "Upload your **reference-grade instrument** CSV and **low-cost sensor (LCS)** CSV datasets. "
        "Both files must share a common timestamp column so they can be aligned in the next step."
    )

    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader(
            "Reference CSV",
            type=["csv"],
            key="ref_upload",
            help=(
                "CSV from your reference-grade instrument (e.g. TEOM, BAM, GRIMM, AQMS). "
                "Must contain a timestamp column and at least one pollutant column (e.g. pm25, no2). "
                "Timestamps should be ISO 8601 (e.g. 2024-01-01 10:00:00) or Unix epoch."
            ),
        )
    with col2:
        sen_file = st.file_uploader(
            "LCS (Sensor) CSV",
            type=["csv"],
            key="sen_upload",
            help=(
                "CSV from your low-cost sensor (e.g. OPC-N3, SPS30, PMS5003, AirVisual). "
                "Must contain the same timestamp column as the reference file. "
                "Can include multiple sensor channels — you will pick which ones to use later."
            ),
        )

    cfg_file = st.file_uploader(
        "Optional: Custom Config (YAML/JSON)",
        type=["yaml", "yml", "json"],
        key="cfg_upload",
        help=(
            "Upload a previously saved config file to pre-fill all pipeline settings. "
            "Leave empty to use the built-in defaults (recommended for first-time use). "
            "A config is automatically generated and downloadable from the Export step."
        ),
    )
    use_sample = st.checkbox(
        "Use bundled sample datasets",
        value=True,
        help=(
            "Loads a built-in reference + LCS dataset pair for demonstration. "
            "Uncheck this once you have your own CSV files uploaded above."
        ),
    )

    # Preview
    with st.expander("👀 Preview bundled sample datasets"):
        r, s = cached_load_sample()
        c1, c2 = st.columns(2)
        c1.caption("Reference (first 10 rows)")
        c1.dataframe(r.head(10), width='stretch')
        c2.caption("LCS (first 10 rows)")
        c2.dataframe(s.head(10), width='stretch')

    # Config customisation
    with st.expander("⚙️ Data Configuration"):
        st.caption(
            "Tell the pipeline which columns contain the timestamp and the target pollutant. "
            "These names must match exactly what's in your CSV headers."
        )
        config = resolve_config(cfg_file)
        data_cfg = config.get("data", {})
        c1, c2, c3 = st.columns(3)
        ts_col = c1.text_input(
            "Timestamp column",
            value=data_cfg.get("timestamp_column", "timestamp"),
            help=(
                "Exact column name containing the date/time in both CSV files. "
                "Must be identical in the reference and sensor CSVs. "
                "Example: 'timestamp', 'datetime', 'date_time', 'time'."
            ),
        )
        target_col = c2.text_input(
            "Target column (reference)",
            value=data_cfg.get("target_column", "pm25"),
            help=(
                "The pollutant column in the **reference** CSV that you are calibrating the sensor against. "
                "Example: 'pm25', 'pm2_5', 'no2', 'o3'. "
                "The model will learn to predict this value from the sensor readings."
            ),
        )
        tz = c3.selectbox(
            "Timezone",
            ["UTC", "US/Eastern", "US/Pacific", "Europe/London", "Asia/Kolkata", "Asia/Tokyo"],
            index=0,
            help=(
                "Timezone to apply when parsing timestamps. "
                "Use UTC if your timestamps are already in UTC or you are unsure. "
                "Mismatched timezones will cause the alignment step to fail."
            ),
        )
        config["data"]["timestamp_column"] = ts_col
        config["data"]["target_column"] = target_col
        config["data"]["timezone"] = tz

    if st.button("🚀 Load & Validate Data", key="run_upload", width='stretch',
                 help="Loads, parses and validates both CSV files. Any format errors will be shown below."):
        try:
            ref_src, sen_src, label = resolve_inputs(ref_file, sen_file, use_sample)
            config = resolve_config(cfg_file)
            config["data"]["timestamp_column"] = ts_col
            config["data"]["target_column"] = target_col
            config["data"]["timezone"] = tz
            cfg_text = _cfg_to_json(config)
            data_out = cached_load_input(ref_src, sen_src, cfg_text)
            st.session_state.config = config
            st.session_state.input_label = label
            st.session_state.data_outputs = data_out
            # Reset downstream
            _reset_downstream(*_DOWNSTREAM_FROM_UPLOAD)
            st.success(f"✅ Data loaded successfully from **{label}**")
        except Exception as e:
            st.error(f"❌ {e}")

    if st.session_state.data_outputs is not None:
        pills = (
            info_pill(f"Source: {st.session_state.input_label}")
            + info_pill(f"Ref rows: {len(st.session_state.data_outputs['reference_raw'])}")
            + info_pill(f"LCS rows: {len(st.session_state.data_outputs['sensor_raw'])}")
        )
        st.markdown(pills, unsafe_allow_html=True)

        ts_name = st.session_state.config["data"].get("timestamp_column", "timestamp")
        target_name = st.session_state.config["data"].get("target_column", "pm25")

        with st.expander("🗑️ Drop Unneeded Columns", expanded=False):
            st.caption(
                "Remove any columns you do not want carried into preprocessing and modelling "
                "(e.g. duplicate channels, diagnostics, unused pollutants). "
                "The timestamp column is always kept; the reference target column cannot be dropped."
            )
            ref_df = st.session_state.data_outputs["reference_raw"]
            sen_df = st.session_state.data_outputs["sensor_raw"]
            ref_options = [c for c in ref_df.columns if c not in (ts_name, target_name)]
            sen_options = [c for c in sen_df.columns if c != ts_name]
            dc1, dc2 = st.columns(2)
            dropped_ref = dc1.multiselect(
                "Reference columns to drop",
                ref_options,
                default=[c for c in st.session_state.dropped_ref_cols if c in ref_options],
                key="drop_ref_cols",
                help="Columns removed from the reference dataset before preprocessing.",
            )
            dropped_sen = dc2.multiselect(
                "Sensor columns to drop",
                sen_options,
                default=[c for c in st.session_state.dropped_sen_cols if c in sen_options],
                key="drop_sen_cols",
                help="Columns removed from the sensor dataset before preprocessing.",
            )
            if st.button("Apply column drops", key="apply_drops", width='stretch'):
                st.session_state.dropped_ref_cols = dropped_ref
                st.session_state.dropped_sen_cols = dropped_sen
                st.session_state.data_outputs["reference_raw"] = ref_df.drop(columns=dropped_ref, errors="ignore")
                st.session_state.data_outputs["sensor_raw"] = sen_df.drop(columns=dropped_sen, errors="ignore")
                _reset_downstream(*_DOWNSTREAM_FROM_UPLOAD)
                st.success(f"✅ Dropped {len(dropped_ref)} reference and {len(dropped_sen)} sensor column(s)")
                st.rerun()

        tab1, tab2 = st.tabs(["Reference Data", "Sensor Data"])
        with tab1:
            ref_preview = st.session_state.data_outputs["reference_raw"].head(20)
            st.dataframe(ref_preview, width='stretch')
            render_df_download(ref_preview, key="ref_preview_csv", filename="reference_preview.csv")
        with tab2:
            sen_preview = st.session_state.data_outputs["sensor_raw"].head(20)
            st.dataframe(sen_preview, width='stretch')
            render_df_download(sen_preview, key="sen_preview_csv", filename="sensor_preview.csv")

        if st.button("Next ➡️", key="next_upload", width='stretch'):
            go_next()



# ---------------------------------------------------------------------------
# STEP 2: Preprocessing
# ---------------------------------------------------------------------------

def render_preprocessing():
    st.subheader("🧹 Preprocessing")
    if st.session_state.data_outputs is None or st.session_state.config is None:
        st.info("Complete the **Upload Data** step first.")
        return

    config = st.session_state.config

    st.caption(
        "The **sensor (LCS)** and **reference** datasets are cleaned **independently** — "
        "each has its own missing-value strategy and outlier settings. "
        "Missing values are filled first, then outliers are removed."
    )

    def _preprocess_controls(cfg: dict, key_prefix: str, default_missing: str, default_outlier: str) -> dict:
        c1, c2, c3 = st.columns(3)
        cur_missing = str(cfg.get("missing_strategy", default_missing))
        cur_missing_label = _INTERNAL_TO_DISPLAY.get(cur_missing, cur_missing)
        if cur_missing_label not in MISSING_OPTIONS:
            cur_missing_label = default_missing
        missing_method = c1.selectbox(
            "Missing value strategy",
            MISSING_OPTIONS,
            index=MISSING_OPTIONS.index(cur_missing_label),
            key=f"{key_prefix}_missing",
            help=(
                "How to fill gaps (NaN):\n\n"
                "• **No Treatment** — leave gaps untouched.\n"
                "• **Forward Fill** — copy the last known value forward.\n"
                "• **Backward Fill** — copy the next known value backward.\n"
                "• **Linear Interpolation** — interpolate between known values (edge gaps may remain).\n"
                "• **Linear Interpolation + Forward Fill** *(recommended)* — interpolate then forward-fill edges.\n"
                "• **Linear Interpolation + Backward Fill** — interpolate then back-fill edges.\n"
                "• **Drop Missing Rows** — delete any row containing a missing numeric value."
            ),
        )
        outlier_method = c2.selectbox(
            "Outlier removal method",
            OUTLIER_OPTIONS,
            index=OUTLIER_OPTIONS.index(str(cfg.get("outlier_method", default_outlier)))
            if str(cfg.get("outlier_method", default_outlier)) in OUTLIER_OPTIONS else OUTLIER_OPTIONS.index(default_outlier),
            key=f"{key_prefix}_outlier",
            help=(
                "Statistical method used to flag and remove extreme readings:\n\n"
                "• **iqr** — outside Q1 − k×IQR or Q3 + k×IQR. Robust to non-normal data.\n"
                "• **zscore** — beyond k standard deviations from the mean.\n"
                "• **none** — no outlier removal (threshold disabled)."
            ),
        )
        outlier_threshold = c3.number_input(
            "Outlier threshold",
            min_value=0.5, max_value=10.0,
            value=float(cfg.get("outlier_threshold", 1.5)),
            step=0.1,
            key=f"{key_prefix}_threshold",
            disabled=(outlier_method == "none"),
            help=(
                "Sensitivity of the outlier detector. Disabled when method is **none**.\n\n"
                "• For **IQR**: multiplier on the interquartile range (1.5 = box-plot rule).\n"
                "• For **z-score**: max allowed standard deviations from the mean."
            ),
        )
        return {
            "missing_strategy": missing_method,
            "outlier_method": outlier_method,
            "outlier_threshold": outlier_threshold,
        }

    prep_cfg = config.setdefault("preprocessing", {})
    sensor_cfg = dict(prep_cfg.get("sensor", {}))
    reference_cfg = dict(prep_cfg.get("reference", {}))

    with st.expander("⚙️ Sensor (LCS) Data Preprocessing", expanded=True):
        st.caption("Controls how the low-cost sensor data is cleaned before alignment.")
        sensor_settings = _preprocess_controls(
            sensor_cfg, "sensor_prep",
            default_missing="Linear Interpolation + Forward Fill",
            default_outlier="iqr",
        )

    with st.expander("⚙️ Reference Data Preprocessing", expanded=False):
        st.info(
            "💡 Reference instruments are high-accuracy devices — outlier removal is usually "
            "unnecessary. The target column is always protected from removal."
        )
        st.caption("Controls how the reference data is cleaned — fully independent of the sensor settings above.")
        reference_settings = _preprocess_controls(
            reference_cfg, "reference_prep",
            default_missing="Linear Interpolation + Forward Fill",
            default_outlier="none",
        )

    prep_cfg["sensor"] = sensor_settings
    prep_cfg["reference"] = reference_settings
    st.session_state.config = config

    if st.button("🚀 Run Preprocessing", key="run_preprocess", width='stretch'):
        with st.spinner("Cleaning datasets…"):
            try:
                cfg_text = _cfg_to_json(config)
                outputs = cached_preprocessing(
                    st.session_state.data_outputs["reference_raw"],
                    st.session_state.data_outputs["sensor_raw"],
                    cfg_text,
                )
                st.session_state.preprocessing_outputs = outputs
                _reset_downstream(*_DOWNSTREAM_FROM_PREPROCESSING)
                st.success("✅ Preprocessing completed")
            except Exception as e:
                st.error(f"❌ {e}")

    out = st.session_state.preprocessing_outputs
    if out is not None:
        ref_s = out["preprocessing_summary"]["reference"]
        sen_s = out["preprocessing_summary"]["sensor"]
        c1, c2 = st.columns(2)
        with c1:
            section("Reference Summary")
            st.markdown(
                info_pill(f"Original: {ref_s.original_rows}")
                + info_pill(f"Cleaned: {ref_s.cleaned_rows}")
                + info_pill(f"Outliers removed: {ref_s.rows_removed_as_outliers}"),
                unsafe_allow_html=True,
            )
        with c2:
            section("Sensor Summary")
            st.markdown(
                info_pill(f"Original: {sen_s.original_rows}")
                + info_pill(f"Cleaned: {sen_s.cleaned_rows}")
                + info_pill(f"Outliers removed: {sen_s.rows_removed_as_outliers}"),
                unsafe_allow_html=True,
            )

        tab1, tab2 = st.tabs(["Reference (cleaned)", "Sensor (cleaned)"])
        with tab1:
            st.dataframe(out["reference_processed"].head(20), width='stretch')
            render_df_download(out["reference_processed"], key="ref_cleaned_csv", filename="reference_cleaned.csv")
        with tab2:
            st.dataframe(out["sensor_processed"].head(20), width='stretch')
            render_df_download(out["sensor_processed"], key="sen_cleaned_csv", filename="sensor_cleaned.csv")

        if st.button("Next ➡️", key="next_preprocess", width='stretch'):
            go_next()


# ---------------------------------------------------------------------------
# STEP 3: Alignment
# ---------------------------------------------------------------------------

def render_alignment():
    st.subheader("🔗 Time Alignment & Synchronization")
    if st.session_state.preprocessing_outputs is None or st.session_state.config is None:
        st.info("Complete the **Preprocessing** step first.")
        return

    config = st.session_state.config

    with st.expander("⚙️ Alignment Settings", expanded=True):
        st.caption(
            "Resamples both datasets to a common time grid, auto-detects any clock offset "
            "(lag) between the sensor and reference, then merges them row-by-row."
        )
        c1, c2, c3, c4 = st.columns(4)
        resample_rule = c1.selectbox(
            "Resample frequency",
            ["1min", "5min", "10min", "15min", "30min", "1h", "2h", "3h", "6h", "12h", "1D"],
            index=5,
            help=(
                "All timestamps are rounded and aggregated to this interval before merging. "
                "Choose a frequency that matches your sensor’s native sampling rate, or coarser. "
                "Example: if your sensor logs every 5 min, use ‘5min’ or ‘1h’ for hourly averages."
            ),
        )
        aggregation = c2.selectbox(
            "Aggregation",
            ["mean", "median", "sum", "min", "max"],
            index=0,
            help=(
                "How multiple readings within each time bin are combined.\n\n"
                "• **mean** — average (recommended for most air quality data)\n"
                "• **median** — robust to within-bin spikes\n"
                "• **sum** — use only for accumulation data (e.g. total counts)\n"
                "• **min/max** — for peak or trough analysis"
            ),
        )
        merge_strategy = c3.selectbox(
            "Merge strategy",
            ["inner", "nearest"],
            index=0,
            help=(
                "• **inner** — keeps only timestamps that exist in **both** datasets "
                "(exact match after resampling). Safest — no interpolation across gaps.\n"
                "• **nearest** — matches each sensor timestamp to the closest reference "
                "timestamp (merge_asof). Useful when clocks are slightly offset but "
                "may pair non-simultaneous readings."
            ),
        )
        max_lag = c4.number_input(
            "Max lag steps",
            min_value=0, max_value=24,
            value=int(config["alignment"].get("max_lag_steps", 0)),
            help=(
                "The pipeline tests cross-correlation at offsets from −N to +N steps to "
                "find the best time-shift between sensor and reference. "
                "Set to 0 to skip lag detection. "
                "Example: with 1h resampling and max lag 3, it checks offsets up to ±3 hours."
            ),
        )
        config["alignment"]["resample_rule"] = resample_rule
        config["alignment"]["aggregation"] = aggregation
        config["alignment"]["merge_strategy"] = merge_strategy
        config["alignment"]["max_lag_steps"] = max_lag
        st.session_state.config = config

    if st.button("🚀 Run Alignment", key="run_alignment", width='stretch'):
        with st.spinner("Aligning datasets…"):
            try:
                cfg_text = _cfg_to_json(config)
                outputs = cached_alignment(
                    st.session_state.preprocessing_outputs["reference_processed"],
                    st.session_state.preprocessing_outputs["sensor_processed"],
                    cfg_text,
                )
                st.session_state.alignment_outputs = outputs
                _reset_downstream(*_DOWNSTREAM_FROM_ALIGNMENT)
                st.success("✅ Alignment completed")
            except Exception as e:
                st.error(f"❌ {e}")

    out = st.session_state.alignment_outputs
    if out is not None:
        meta = out["alignment_metadata"]
        pills = (
            info_pill(f"Lag detected: {meta['lag_steps']} steps")
            + info_pill(f"Resample: {meta['resample_rule']}")
            + info_pill(f"Merge: {meta['merge_strategy']}")
            + info_pill(f"Merged rows: {meta['merged_rows']}")
            + info_pill(f"Lag column: {meta['lag_detection_column']}")
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.dataframe(out["merged_data"].head(20), width='stretch')
        render_df_download(out["merged_data"], key="merged_data_csv", filename="merged_aligned_data.csv")

        if st.button("Next ➡️", key="next_alignment", width='stretch'):
            go_next()


# ---------------------------------------------------------------------------
# STEP 4: EDA
# ---------------------------------------------------------------------------

def render_eda():
    st.subheader("📊 Exploratory Data Analysis")
    if st.session_state.alignment_outputs is None or st.session_state.config is None:
        st.info("Complete the **Alignment** step first.")
        return

    merged = st.session_state.alignment_outputs["merged_data"]

    if st.button("🚀 Generate EDA", key="run_eda", width='stretch'):
        with st.spinner("Analysing data…"):
            try:
                cfg_text = _cfg_to_json(st.session_state.config)
                outputs = cached_eda(merged, cfg_text)
                st.session_state.eda_outputs = outputs
                st.success("✅ EDA generated")
            except Exception as e:
                st.error(f"❌ {e}")

    out = st.session_state.eda_outputs
    if out is not None:
        tabs = st.tabs(["📈 Distributions", "🔥 Correlations", "❓ Missing Values", "📉 Time Series", "⚠️ Anomalies"])

        with tabs[0]:
            numeric_cols = out.get("numeric_columns", [])
            if numeric_cols:
                sel_col = st.selectbox("Select column", numeric_cols, key="eda_distcol")
                from modules.eda import create_distribution_figure
                fig = create_distribution_figure(merged, sel_col)
                title, x_label, y_label = _chart_customization(
                    "eda_dist", f"Distribution of {sel_col}", sel_col, "Count"
                )
                fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
                _display_chart_with_downloads(
                    fig, merged[[sel_col]], key="eda_dist", filename_prefix="distribution"
                )
            if "before_after_figure" in out:
                _display_chart_with_downloads(
                    out["before_after_figure"], None, key="eda_before_after",
                    filename_prefix="before_after",
                )

        with tabs[1]:
            corr_source = merged.select_dtypes(include=[np.number]).corr().reset_index()
            _display_chart_with_downloads(
                out["correlation_figure"], corr_source,
                key="eda_corr", filename_prefix="correlation_heatmap",
            )

        with tabs[2]:
            _display_chart_with_downloads(
                out["missing_heatmap"], out.get("missing_summary"),
                key="eda_missing", filename_prefix="missing_heatmap",
            )
            st.dataframe(out["missing_summary"], width='stretch')
            render_df_download(out["missing_summary"], key="eda_missing_csv", filename="missing_summary.csv")

        with tabs[3]:
            _display_chart_with_downloads(
                out["time_series_figure"], merged, key="eda_ts", filename_prefix="time_series",
            )

        with tabs[4]:
            _display_chart_with_downloads(
                out["anomaly_figure"], out.get("anomalies"), key="eda_anomaly",
                filename_prefix="anomalies",
            )
            if not out["anomalies"].empty:
                st.caption(f"Detected {len(out['anomalies'])} anomalous rows")
                st.dataframe(out["anomalies"].head(25), width='stretch')
                render_df_download(out["anomalies"], key="eda_anomalies_csv", filename="anomalies.csv")
            else:
                st.success("No anomalies detected.")

        if st.button("Next ➡️", key="next_eda", width='stretch'):
            go_next()




# ---------------------------------------------------------------------------
# STEP 5: Variable Selection
# ---------------------------------------------------------------------------

def render_variable_selection():
    st.subheader("🎯 Variable Selection")
    if st.session_state.alignment_outputs is None or st.session_state.config is None:
        st.info("Complete the **Alignment** step first.")
        return

    config = st.session_state.config
    merged = st.session_state.alignment_outputs["merged_data"]
    ts_col = config["data"]["timestamp_column"]

    numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c != ts_col]
    if not numeric_cols:
        st.error("No numeric columns available for variable selection.")
        return

    st.markdown("Select the **target** (reference) variable and **predictor** (sensor) variables.")

    target = st.selectbox(
        "Target variable (what you are calibrating against)",
        numeric_cols,
        index=0,
        key="var_sel_target",
        help="The reference measurement column that the model should predict.",
    )

    predictor_options = [c for c in numeric_cols if c != target]
    default_predictors = [c for c in predictor_options if c in (st.session_state.selected_predictors or predictor_options)]
    predictors = st.multiselect(
        "Predictor variables (sensor features)",
        predictor_options,
        default=default_predictors or predictor_options,
        key="var_sel_predictors",
        help="The sensor columns used as inputs to the calibration model.",
    )

    if not predictors:
        st.warning("Select at least one predictor variable.")
        return

    modelling_df = merged[[ts_col, target] + predictors].copy()
    st.session_state.selected_target = target
    st.session_state.selected_predictors = predictors
    st.session_state.variable_selection_outputs = {"modelling_dataset": modelling_df}

    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(modelling_df):,}")
    c2.metric("Columns", f"{len(modelling_df.columns):,}")
    st.dataframe(modelling_df.head(20), width='stretch')

    render_df_download(modelling_df, key="var_sel_dataset_csv", filename="modelling_dataset.csv")

    if st.button("Next \u27a1\ufe0f", key="next_variable_selection", width='stretch'):
        go_next()


# ---------------------------------------------------------------------------
# STEP 6: Feature Engineering
# ---------------------------------------------------------------------------

def render_feature_engineering():
    st.subheader("\U0001f9ec Feature Engineering")
    if st.session_state.variable_selection_outputs is None or st.session_state.config is None:
        st.info("Complete the **Variable Selection** step first.")
        return

    config = st.session_state.config
    modelling_df = st.session_state.variable_selection_outputs["modelling_dataset"]
    ts_col = config["data"]["timestamp_column"]
    target_col = st.session_state.selected_target
    sensor_base_cols = [c for c in modelling_df.columns if c not in [ts_col, target_col]]

    st.markdown(
        info_pill(f"Rows: {len(modelling_df)}")
        + info_pill(f"Base features: {len(sensor_base_cols)}")
        + info_pill(f"Target: {target_col}"),
        unsafe_allow_html=True,
    )

    fe_config = config.setdefault("feature_engineering", {})

    with st.expander("\u23f3 Lag Features", expanded=False):
        lag_steps_str = st.text_input(
            "Lag steps (comma-separated)",
            value=",".join(str(s) for s in fe_config.get("lag_steps", [1, 2, 3])),
            key="fe_lag_steps",
            help="Integers representing how many time-steps back to look.",
        )
        rolling_str = st.text_input(
            "Rolling windows (comma-separated)",
            value=",".join(str(w) for w in fe_config.get("rolling_windows", [3, 6])),
            key="fe_rolling_windows",
            help="Window sizes for rolling mean/std.",
        )
        rolling_std = st.checkbox(
            "Include rolling std",
            value=bool(fe_config.get("rolling_std", True)),
            key="fe_rolling_std",
        )
        try:
            fe_config["lag_steps"] = _parse_positive_int_list(lag_steps_str)
        except ValueError:
            st.warning("Invalid lag steps.")
        try:
            fe_config["rolling_windows"] = _parse_positive_int_list(rolling_str)
        except ValueError:
            st.warning("Invalid rolling windows.")
        fe_config["rolling_std"] = rolling_std

    with st.expander("\U0001f4d0 Polynomial & Interaction Features", expanded=False):
        poly_degree = st.selectbox(
            "Polynomial degree", [1, 2, 3],
            index=[1, 2, 3].index(int(fe_config.get("polynomial_degree", 1))),
            key="fe_poly_degree",
        )
        poly_cols = st.multiselect(
            "Columns for polynomial expansion",
            sensor_base_cols,
            default=[c for c in fe_config.get("polynomial_columns", []) if c in sensor_base_cols],
            key="fe_poly_cols",
        )
        interaction_cols = st.multiselect(
            "Columns for pairwise interaction",
            sensor_base_cols,
            default=[c for c in fe_config.get("interaction_columns", []) if c in sensor_base_cols],
            key="fe_interaction_cols",
        )
        fe_config["polynomial_degree"] = poly_degree
        fe_config["polynomial_columns"] = poly_cols
        fe_config["interaction_columns"] = interaction_cols

    with st.expander("\u23f0 Time Features", expanded=False):
        existing_tf = fe_config.get("time_feature_flags", {})
        add_time = st.checkbox(
            "Enable time features",
            value=bool(fe_config.get("add_time_features", True)),
            key="fe_add_time",
            help="Adds hour-of-day, day-of-week, and day-of-month columns automatically. Extra features below are optional.",
        )
        fe_config["add_time_features"] = add_time
        if add_time:
            st.caption("Basic features (hour 0-23, day-of-week 0-6, day-of-month 1-31) always included. Enable extras:")
            tc1, tc2, tc3 = st.columns(3)
            tf = {"hour_of_day": True, "day_of_week": True, "day_of_month": True}
            tf["unix_timestamp"] = tc1.checkbox("Unix timestamp", value=existing_tf.get("unix_timestamp", False), key="fe_tf_unix",
                help="Seconds since 1970-01-01. Useful as a linear time trend proxy for tree models.")
            tf["julian_date"] = tc1.checkbox("Julian date (DOY)", value=existing_tf.get("julian_date", False), key="fe_tf_julian",
                help="Day-of-year (1-366). Captures seasonal variation without cyclical encoding.")
            tf["calendar_date"] = tc1.checkbox("Calendar date (int)", value=existing_tf.get("calendar_date", False), key="fe_tf_caldate",
                help="Integer days since 1970-01-01, day-resolution. Good for long-term trend.")
            tf["cyclical_hour"] = tc2.checkbox("Cyclical hour sin/cos", value=existing_tf.get("cyclical_hour", False), key="fe_tf_cychour",
                help="Encodes hour as sin/cos so hour 23 is treated as close to hour 0. Better than raw integer for linear models.")
            tf["cyclical_dow"] = tc2.checkbox("Cyclical DOW sin/cos", value=existing_tf.get("cyclical_dow", False), key="fe_tf_cydow",
                help="Circular day-of-week encoding. Sunday wraps around to Monday correctly.")
            tf["cyclical_doy"] = tc2.checkbox("Cyclical DOY sin/cos", value=existing_tf.get("cyclical_doy", False), key="fe_tf_cydoy",
                help="Circular day-of-year encoding. Dec 31 treated as adjacent to Jan 1.")
            tf["season"] = tc3.checkbox("Season (DJF/MAM/JJA/SON)", value=existing_tf.get("season", False), key="fe_tf_season",
                help="Meteorological season as 0-3: DJF(winter)=0, MAM(spring)=1, JJA(summer)=2, SON(autumn)=3.")
            tf["day_name"] = tc3.checkbox("Day name & weekend flag", value=existing_tf.get("day_name", False), key="fe_tf_dayname",
                help="Adds day-of-week number (0-6) and binary is_weekend (1=Sat/Sun). Traffic-related PM differs on weekends.")
            fe_config["time_feature_flags"] = tf

    st.session_state.config = config

    if st.button("\U0001f680 Preview Features", key="preview_features", width='stretch'):
        with st.spinner("Engineering features..."):
            try:
                from modules.feature_engineering import engineer_features
                featured = engineer_features(
                    dataframe=modelling_df,
                    timestamp_column=ts_col,
                    target_column=target_col,
                    config=fe_config,
                )
                st.session_state.feature_engineering_outputs = {"featured_dataset": featured}
                st.session_state.featured_preview = featured
                st.success(f"\u2705 Features created: {len(featured.columns) - 2} features, {len(featured)} rows")
            except Exception as e:
                st.error(f"\u274c {e}")

    fe_out = st.session_state.feature_engineering_outputs
    if fe_out is not None:
        featured = fe_out["featured_dataset"]
        st.dataframe(featured.head(20), width='stretch')
        render_df_download(featured, key="fe_dataset_csv", filename="featured_dataset.csv")

    if st.button("Next \u27a1\ufe0f", key="next_feature_engineering", width='stretch',
                 disabled=(st.session_state.feature_engineering_outputs is None)):
        go_next()


# ---------------------------------------------------------------------------
# STEP 7: Normalization
# ---------------------------------------------------------------------------

def render_normalization():
    st.subheader("\U0001f4cf Normalization")
    if st.session_state.feature_engineering_outputs is None or st.session_state.config is None:
        st.info("Complete the **Feature Engineering** step first.")
        return

    config = st.session_state.config
    featured = st.session_state.feature_engineering_outputs["featured_dataset"]
    ts_col = config["data"]["timestamp_column"]
    target_col = st.session_state.selected_target

    norm_methods = {"None": "none", "StandardScaler": "standard", "MinMaxScaler": "minmax", "RobustScaler": "robust"}
    current = config.get("normalization", {}).get("method", "none")
    current_label = {v: k for k, v in norm_methods.items()}.get(current, "None")

    method_label = st.selectbox(
        "Normalization method",
        list(norm_methods.keys()),
        index=list(norm_methods.keys()).index(current_label),
        key="norm_method",
        help="StandardScaler: zero mean, unit variance. MinMaxScaler: [0,1] range. RobustScaler: robust to outliers.",
    )
    method = norm_methods[method_label]
    config.setdefault("normalization", {})["method"] = method
    st.session_state.config = config

    if st.button("\U0001f680 Apply Normalization", key="apply_normalization", width='stretch'):
        with st.spinner("Normalizing..."):
            try:
                norm_cols = [c for c in featured.columns if c not in [ts_col, target_col]]
                if method == "none":
                    normalized = featured.copy()
                    summary = get_normalization_summary(featured, featured, norm_cols)
                else:
                    before = featured.copy()
                    normalized = normalize_dataset(featured.copy(), norm_cols, method)
                    summary = get_normalization_summary(before, normalized, norm_cols)

                st.session_state.normalization_outputs = {
                    "normalized_dataset": normalized,
                    "method": method,
                    "summary": summary,
                }
                st.success(f"\u2705 Normalization applied: {method_label}")
            except Exception as e:
                st.error(f"\u274c {e}")

    norm_out = st.session_state.normalization_outputs
    if norm_out is not None:
        normalized = norm_out["normalized_dataset"]
        summary = norm_out["summary"]

        if not summary.empty:
            before_tbl, after_tbl = _normalization_summary_tables(summary)
            t1, t2 = st.tabs(["Before", "After"])
            with t1:
                st.dataframe(before_tbl, width='stretch')
                render_df_download(before_tbl, key="norm_before_csv", filename="normalization_before.csv")
            with t2:
                st.dataframe(after_tbl, width='stretch')
                render_df_download(after_tbl, key="norm_after_csv", filename="normalization_after.csv")

        st.dataframe(normalized.head(20), width='stretch')
        render_df_download(normalized, key="norm_dataset_csv", filename="normalized_dataset.csv")

    if st.button("Next \u27a1\ufe0f", key="next_normalization", width='stretch',
                 disabled=(st.session_state.normalization_outputs is None)):
        go_next()


# ---------------------------------------------------------------------------
# STEP 8: Modelling (updated prerequisites)
# ---------------------------------------------------------------------------
def render_modelling():
    st.subheader("🤖 Modelling")
    if st.session_state.normalization_outputs is None or st.session_state.config is None:
        st.info("Complete the **Normalization** step first.")
        return

    config = st.session_state.config
    model_df = st.session_state.normalization_outputs["normalized_dataset"]
    ts_col = config["data"]["timestamp_column"]
    target_col = st.session_state.selected_target
    if target_col is None or target_col not in model_df.columns:
        st.error("Target variable not found. Revisit the **Variable Selection** step.")
        return
    numeric_cols = model_df.select_dtypes(include="number").columns.tolist()
    sensor_base_cols = [c for c in numeric_cols if c != target_col]

    st.markdown(
        info_pill(f"Rows: {len(model_df)}")
        + info_pill(f"Features: {len(sensor_base_cols)}")
        + info_pill(f"Target: {target_col}"),
        unsafe_allow_html=True,
    )

    # ---- Model Reference Guide ----
    with st.expander("📚 Model Reference Guide — How Each Model Works", expanded=False):
        st.caption(
            "Everything you need to know before training: how each model works, "
            "the formula it uses, what inputs it needs, and every hyperparameter available. "
            "Parameters marked ✅ are exposed in the UI below; ❌ use defaults or Auto-Tuning."
        )
        _mtabs = st.tabs([
            "📐 Linear Regression",
            "🔵 Ridge",
            "🟡 Lasso",
            "🌲 Random Forest",
            "⚡ XGBoost",
        ])

        with _mtabs[0]:
            st.markdown("#### Linear Regression (Ordinary Least Squares)")
            st.markdown(
                "The baseline model. Fits a straight-line relationship between your sensor features and "
                "the reference PM value by **minimising the sum of squared errors**. "
                "No regularisation — every feature gets a coefficient. Fast and fully interpretable."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n")
            st.markdown("**Objective (what it minimises):**")
            st.latex(r"\min_{\boldsymbol{\beta}} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2")
            st.markdown("**What inputs does it need?**")
            st.info(
                "• **Numeric feature columns only** (sensor readings, lag features, rolling means, time features).\n"
                "• Works best when the sensor–reference relationship is roughly **linear** "
                "(e.g. raw PM channel vs reference PM).\n"
                "• Sensitive to **correlated features** (multicollinearity) — if you add many lag/rolling columns, "
                "use Ridge or Lasso instead.\n"
                "• Feature scaling is NOT required, but helps compare coefficient magnitudes."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `fit_intercept` | `True` | ❌ | Fit β₀ (intercept). Almost always True. |\n"
                "| `positive` | `False` | ❌ | Force all coefficients ≥ 0. Rarely needed. |\n\n"
                "> Linear Regression has **no regularisation parameters**. "
                "If the model overfits or collinear features are a concern, switch to Ridge or Lasso."
            )

        with _mtabs[1]:
            st.markdown("#### Ridge Regression (L2 Regularisation)")
            st.markdown(
                "Extends OLS with an **L2 penalty** on coefficient size. "
                "All coefficients are **shrunk towards zero** (but never exactly zero). "
                "Ideal when many correlated features are present — it distributes weight across them rather than "
                "picking one arbitrarily."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \beta_0 + \sum_{j=1}^{n} \beta_j x_j")
            st.markdown("**Objective:**")
            st.latex(
                r"\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 "
                r"+ \alpha \sum_{j=1}^{n} \beta_j^2 \right]"
            )
            st.markdown("α controls the trade-off: **α → 0** = plain OLS · **α → ∞** = all coefficients → 0.")
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns.\n"
                "• **Recommended** when you have lag/rolling features that are correlated with each other.\n"
                "• Feature scaling helps (so the penalty is applied equally across all features).\n"
                "• For PM calibration: good with humidity-corrected features (PM × RH terms)."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `alpha` | `1.0` | ✅ | L2 penalty strength. Try: 0.01, 0.1, 1, 10, 100. |\n"
                "| `fit_intercept` | `True` | ❌ | Whether to fit the intercept β₀. |\n"
                "| `solver` | `'auto'` | ❌ | Algorithm: `'auto'`, `'svd'`, `'cholesky'`, `'lsqr'`. |\n"
                "| `max_iter` | `None` | ❌ | Max iterations for iterative solvers. |\n"
                "| `tol` | `1e-4` | ❌ | Convergence tolerance. |\n\n"
                "> Auto-tuning searches: α ∈ {0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100}."
            )

        with _mtabs[2]:
            st.markdown("#### Lasso Regression (L1 Regularisation)")
            st.markdown(
                "Like Ridge, but uses the **absolute value** of coefficients as the penalty. "
                "The key difference: L1 can drive some coefficients to **exactly zero**, "
                "automatically removing irrelevant features. "
                "Acts as a built-in feature selector — useful when you suspect only a few inputs really matter."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \beta_0 + \sum_{j=1}^{n} \beta_j x_j")
            st.markdown("**Objective:**")
            st.latex(
                r"\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 "
                r"+ \alpha \sum_{j=1}^{n} |\beta_j| \right]"
            )
            st.markdown("Higher α → more zero coefficients → sparser model.")
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns.\n"
                "• Very effective after polynomial expansion: Lasso will automatically discard the "
                "polynomial terms that don't improve fit.\n"
                "• Feature scaling is important so all features compete fairly for the L1 budget.\n"
                "• For PM calibration: start with small α (0.001–0.01) and increase until only meaningful features remain."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `alpha` | `0.01` | ✅ | L1 penalty. Higher = more features zeroed out. |\n"
                "| `fit_intercept` | `True` | ❌ | Whether to fit β₀. |\n"
                "| `max_iter` | `1000` | ❌ | Max iterations for coordinate descent. Increase if you see convergence warnings. |\n"
                "| `tol` | `1e-4` | ❌ | Convergence tolerance. |\n"
                "| `selection` | `'cyclic'` | ❌ | `'cyclic'` (round-robin) or `'random'` update order. |\n\n"
                "> Auto-tuning searches: α ∈ {0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}."
            )

        with _mtabs[3]:
            st.markdown("#### Random Forest Regressor")
            st.markdown(
                "Builds **T independent decision trees**, each trained on a random bootstrap sample of the data "
                "and using only a random subset of features at each split (to decorrelate trees). "
                "The final prediction is the **average** across all trees. "
                "Naturally handles non-linear relationships and requires no feature scaling."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{x})")
            st.markdown(
                "Each tree *fₜ* is grown on a bootstrap sample using `max_features` features per split. "
                "Variance is reduced by averaging; bias is controlled by tree depth."
            )
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns — **no scaling required**.\n"
                "• Handles correlated lag/rolling features well (random feature subsets decorrelate trees).\n"
                "• Works out-of-the-box with minimal tuning. Increasing n_estimators always helps (up to a point).\n"
                "• For PM calibration: often outperforms linear models when humidity causes non-linear "
                "hygroscopic particle growth effects."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `n_estimators` | `200` | ✅ | Number of trees. More = stable but slower. 100–500 typical. |\n"
                "| `max_depth` | `10` | ✅ | Max tree depth. None = fully grown. 5–15 prevents overfitting. |\n"
                "| `min_samples_split` | `2` | ❌ (auto-tuned) | Min samples required to split a node. Higher = simpler trees. |\n"
                "| `min_samples_leaf` | `1` | ❌ (auto-tuned) | Min samples in any leaf. Higher = smoother predictions. |\n"
                "| `max_features` | `'sqrt'` | ❌ (auto-tuned) | Features per split: `'sqrt'`, `'log2'`, or float (fraction). |\n"
                "| `bootstrap` | `True` | ❌ | Use bootstrap sampling per tree. |\n"
                "| `oob_score` | `False` | ❌ | Compute out-of-bag validation score for free. |\n"
                "| `n_jobs` | `-1` | ❌ | CPU threads (−1 = all cores). |\n\n"
                "> Auto-tuning searches: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features."
            )

        with _mtabs[4]:
            st.markdown("#### XGBoost (Extreme Gradient Boosting)")
            st.markdown(
                "Builds trees **sequentially** — each new tree is trained to correct the errors (residuals) "
                "of all previous trees. Unlike Random Forest (parallel + average), XGBoost **boosts** performance "
                "step by step. Has built-in L1 + L2 regularisation on leaf weights. "
                "Typically the most accurate model on tabular data, but needs careful tuning."
            )
            st.markdown("**Formula (final prediction after K rounds):**")
            st.latex(r"\hat{y}^{(K)} = \sum_{k=1}^{K} \eta \cdot f_k(\mathbf{x})")
            st.markdown("where η = `learning_rate` and each tree minimises:")
            st.latex(
                r"\mathcal{L}^{(k)} = \sum_{i} l\!\left(y_i,\, \hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)\right) + \Omega(f_k)"
            )
            st.markdown("**Regularisation term on each tree:**")
            st.latex(
                r"\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|"
            )
            st.markdown(
                "T = number of leaves · wⱼ = leaf scores · "
                "γ = min gain to split · λ = L2 (`reg_lambda`) · α = L1 (`reg_alpha`)"
            )
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns — **no scaling required**.\n"
                "• Handles missing values internally, but we pre-impute in the preprocessing step.\n"
                "• Benefits the most from rich feature engineering (lag, rolling, interaction terms).\n"
                "• Best model for large datasets (> 500 rows) with non-linear sensor behaviour.\n"
                "• Requires `xgboost` package: install with `pip install xgboost`."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `n_estimators` | `200` | ✅ | Number of boosting rounds (trees). |\n"
                "| `max_depth` | `4` | ✅ | Max depth per tree. 3–6 is typical; lower = simpler. |\n"
                "| `learning_rate` (η) | `0.05` | ✅ | Step size per round. Lower needs more trees. |\n"
                "| `subsample` | `0.8` | ❌ (auto-tuned) | Fraction of rows sampled per tree. <1 adds randomness. |\n"
                "| `colsample_bytree` | `0.8` | ❌ (auto-tuned) | Fraction of features used per tree. |\n"
                "| `reg_alpha` (α) | `0.0` | ❌ (auto-tuned) | L1 on leaf weights. Promotes sparse leaf scores. |\n"
                "| `reg_lambda` (λ) | `1.0` | ❌ (auto-tuned) | L2 on leaf weights. Smooths predictions. |\n"
                "| `gamma` | `0` | ❌ | Min loss reduction to split a node. Higher = fewer splits. |\n"
                "| `min_child_weight` | `1` | ❌ | Min sum of instance weight in a leaf. Prevents tiny splits. |\n"
                "| `n_jobs` | `-1` | ❌ | CPU threads. |\n\n"
                "> Auto-tuning searches: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda."
            )

    with st.expander("⚙️ Training Settings", expanded=False):
        st.caption(
            "Controls the train/test split, cross-validation strategy, "
            "and which models are trained. All selected models run in sequence."
        )
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider(
            "Test split size",
            0.1, 0.5,
            float(config["training"].get("test_size", 0.2)),
            0.05,
            help=(
                "Fraction of the dataset held out for final evaluation (not seen during training). "
                "0.2 = 20 % test, 80 % train. "
                "For small datasets (< 500 rows) consider 0.1; for large datasets 0.2–0.3 is standard. "
                "Data is split chronologically (no shuffling) to respect time order."
            ),
        )
        cv_folds = c2.number_input(
            "CV folds",
            2, 10,
            int(config["training"].get("cross_validation_folds", 5)),
            help=(
                "Number of cross-validation splits used during training to estimate generalisation. "
                "Uses **TimeSeriesSplit** — folds respect chronological order so future data is "
                "never used to train on past. 5 folds is a good default; reduce to 3 for very small datasets."
            ),
        )
        validation_labels = {
            "TimeSeriesSplit (chronological CV)": "timeseriessplit",
            "K-Fold (random CV)": "kfold",
            "Holdout (single train/test split)": "holdout",
        }
        current_val = str(config["training"].get("validation_method", "timeseriessplit")).strip().lower()
        current_val_label = next(
            (lbl for lbl, key in validation_labels.items() if key == current_val),
            "TimeSeriesSplit (chronological CV)",
        )
        validation_label = c3.selectbox(
            "Validation method",
            list(validation_labels.keys()),
            index=list(validation_labels.keys()).index(current_val_label),
            help=(
                "Choose how models are validated:\n\n"
                "• **TimeSeriesSplit** — chronological cross-validation (default, leakage-safe for time series).\n"
                "• **K-Fold** — standard random k-fold cross-validation.\n"
                "• **Holdout** — a single chronological train/test split (no cross-validation)."
            ),
        )
        config["training"]["test_size"] = test_size
        config["training"]["cross_validation_folds"] = cv_folds
        config["training"]["validation_method"] = validation_labels[validation_label]

        st.markdown("**Model selection**")
        st.caption(
            "Choose a model family, then pick the specific models to train. "
            "All selected models run in sequence and appear in the leaderboard."
        )
        linear_keys = list(MODEL_GROUPS["Statistical Models"])
        nonlinear_keys = list(MODEL_GROUPS["Machine Learning Models"])
        family_choice = st.radio(
            "Model family",
            ["Linear models", "Non-linear models", "Both"],
            index=2,
            horizontal=True,
            help=(
                "• **Linear models** — OLS, Multiple Linear Regression, Ridge, Lasso (interpretable coefficients).\n"
                "• **Non-linear models** — Random Forest, XGBoost (capture non-linear sensor behaviour).\n"
                "• **Both** — choose freely from every available model."
            ),
        )
        if family_choice == "Linear models":
            available_keys = linear_keys
        elif family_choice == "Non-linear models":
            available_keys = nonlinear_keys
        else:
            available_keys = linear_keys + nonlinear_keys

        prev_selected = [m for m in config["training"].get("selected_models", available_keys) if m in available_keys]
        selected_models = st.multiselect(
            "Models to train",
            options=available_keys,
            default=prev_selected or available_keys,
            format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
            help="Pick one or more models from the chosen family.",
        )
        config["training"]["selected_models"] = selected_models if selected_models else available_keys

    with st.expander("⚙️ Manual Hyperparameters", expanded=False):
        st.caption(
            "Set hyperparameters by hand. These values are used **unless** Auto-Tuning is enabled "
            "for that model, in which case the tuner finds better values automatically."
        )
        model_params = config["training"].get("model_params", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ridge**")
            ridge_alpha = st.number_input(
                "Ridge alpha", 0.001, 100.0,
                float(model_params.get("ridge", {}).get("alpha", 1.0)), 0.1,
                help=(
                    "L2 regularisation strength. Higher α = stronger shrinkage of coefficients → simpler model. "
                    "Lower α → closer to plain linear regression. "
                    "Try values like 0.1, 1.0, 10.0 or use Auto-Tuning to search."
                ),
            )
            model_params.setdefault("ridge", {})["alpha"] = ridge_alpha
            st.markdown("**Lasso**")
            lasso_alpha = st.number_input(
                "Lasso alpha", 0.001, 100.0,
                float(model_params.get("lasso", {}).get("alpha", 0.01)), 0.01,
                help=(
                    "L1 regularisation strength. Higher α = more coefficients forced to exactly zero "
                    "(automatic feature selection). Lower α → closer to plain OLS. "
                    "Start small (0.01–0.1) for dense feature sets."
                ),
            )
            model_params.setdefault("lasso", {})["alpha"] = lasso_alpha
        with c2:
            st.markdown("**Random Forest**")
            rf_n = st.number_input(
                "RF n_estimators", 50, 1000,
                int(model_params.get("random_forest", {}).get("n_estimators", 200)), 50,
                help="Number of decision trees. More trees = more stable predictions but slower training. 200–500 is a good range.",
            )
            rf_d = st.number_input(
                "RF max_depth", 2, 50,
                int(model_params.get("random_forest", {}).get("max_depth", 10)),
                help="Maximum depth of each tree. Deeper = more complex, risk of overfitting. 5–15 works for most cases.",
            )
            model_params.setdefault("random_forest", {})["n_estimators"] = rf_n
            model_params.setdefault("random_forest", {})["max_depth"] = rf_d
            st.markdown("**XGBoost**")
            xgb_n = st.number_input(
                "XGB n_estimators", 50, 1000,
                int(model_params.get("xgboost", {}).get("n_estimators", 200)), 50,
                help="Number of boosting rounds. More = potentially better fit, but more training time and risk of overfitting.",
            )
            xgb_d = st.number_input(
                "XGB max_depth", 2, 20,
                int(model_params.get("xgboost", {}).get("max_depth", 4)),
                help="Max depth of each boosted tree. Keep low (3–6) for small datasets to avoid overfitting.",
            )
            xgb_lr = st.number_input(
                "XGB learning_rate", 0.001, 1.0,
                float(model_params.get("xgboost", {}).get("learning_rate", 0.05)), 0.01,
                help=(
                    "Step size for each boosting round. Lower = more conservative learning, "
                    "needs more estimators. Higher = faster but may overshoot. "
                    "Typical range: 0.01–0.2."
                ),
            )
            model_params.setdefault("xgboost", {})["n_estimators"] = xgb_n
            model_params.setdefault("xgboost", {})["max_depth"] = xgb_d
            model_params.setdefault("xgboost", {})["learning_rate"] = xgb_lr
        config["training"]["model_params"] = model_params

    with st.expander("🔧 Auto-Tuning (RandomizedSearchCV + TimeSeriesSplit)", expanded=False):
        st.caption(
            "Automatically searches for the best hyperparameters using randomised search "
            "with time-series cross-validation. "
            "When enabled for a model, the manual hyperparameters above are **ignored** for that model."
        )
        st.info(
            "💡 More iterations (n_iter) = broader search = better results, but slower. "
            "Start with 10–20. Increase to 50–100 for a thorough search on final runs."
        )
        tuning_cfg = config["training"].get("tuning", {})
        for mname in ["random_forest", "xgboost", "ridge", "lasso"]:
            c1t, c2t = st.columns([2, 1])
            enabled = c1t.checkbox(
                f"Tune {mname}",
                value=tuning_cfg.get(mname, {}).get("enabled", False),
                key=f"tune_{mname}",
                help=f"Enable RandomizedSearchCV for **{mname}**. Ignores the manual hyperparameters above for this model.",
            )
            n_iter = c2t.number_input(
                "n_iter",
                2, 500,
                int(tuning_cfg.get(mname, {}).get("n_iter", 10)),
                key=f"niter_{mname}",
                help="Number of random hyperparameter combinations to try. Higher = more thorough but slower.",
            )
            tuning_cfg.setdefault(mname, {})["enabled"] = enabled
            tuning_cfg.setdefault(mname, {})["n_iter"]  = n_iter
        config["training"]["tuning"] = tuning_cfg

    st.session_state.config = config

    # ---- Feature subset selection (from the prepared dataset) ----
    st.markdown("---")
    st.markdown("### 🔢 Feature Subset")
    st.caption(
        "These are the engineered, normalized features from the previous steps. "
        "Optionally restrict training to a subset (leave empty to use all)."
    )
    with st.expander("📋 Prepared dataset preview (first 5 rows)", expanded=False):
        st.dataframe(model_df[sensor_base_cols].head(5), width='stretch')
        render_df_download(model_df, key="model_df_csv", filename="modelling_dataset.csv")

    prev_sel = st.session_state.selected_features or sensor_base_cols
    valid_prev = [c for c in prev_sel if c in sensor_base_cols]
    selected = st.multiselect(
        "Features for training (empty = use all)",
        options=sensor_base_cols,
        default=valid_prev,
        key="feat_multiselect",
    )
    st.session_state.selected_features = selected if selected else None

    st.markdown("---")
    if st.button("🚀 Train Models", key="run_modeling", width='stretch'):
        with st.spinner("Training models… This may take a moment."):
            try:
                cfg_text = _cfg_to_json(config)
                subset_json = json.dumps(st.session_state.selected_features)
                outputs = cached_train_prepared(model_df, target_col, cfg_text, subset_json)
                st.session_state.modeling_outputs = outputs
                st.session_state.selected_model_name = outputs["best_model_name"]
                _reset_downstream(*_DOWNSTREAM_FROM_MODELING)
                st.success("✅ All models trained successfully!")
            except Exception as e:
                st.error(f"❌ {e}")

    out = st.session_state.modeling_outputs
    if out is not None:
        st.markdown("### 🏅 Leaderboard")
        st.dataframe(
            _format_metric_dataframe(out["leaderboard"])
            .style.highlight_min(subset=["rmse", "mae"], color="#065f4630")
            .highlight_max(subset=["r2", "pearson_r"], color="#065f4630"),
            width='stretch',
        )
        st.success(f"🏆 Best model: **{out['best_model_name']}**")
        tuned = {n: r for n, r in out["training_results"].items() if r.best_params}
        if tuned:
            with st.expander("🔧 Best Hyperparameters Found", expanded=False):
                for mname, res in tuned.items():
                    st.markdown(f"**{mname}**")
                    st.json(res.best_params)

        # ---- Multi-Model Comparison ----
        st.markdown("---")
        with st.expander("🔀 Compare Models Side-by-Side", expanded=False):
            st.caption(
                "Select any number of trained models to compare. "
                "All views update instantly when you change the selection. "
                "Use this to decide which model to carry forward into Results and Post-Analysis."
            )
            _all_model_names = list(out["training_results"].keys())
            _selected_compare = st.multiselect(
                "Models to compare",
                options=_all_model_names,
                default=_all_model_names,
                key="compare_models_multi",
                help=(
                    "Select 2 or more models to compare. All trained models are selected by default. "
                    "Remove models you are not interested in to keep the view clean."
                ),
            )

            if len(_selected_compare) < 2:
                st.warning("⚠️ Select at least 2 models to enable comparison.")
            else:
                _sel_results = {m: out["training_results"][m] for m in _selected_compare}
                _ctabs = st.tabs(["📊 Scatter Grid", "📈 Time-Series Overlay", "📋 Metrics Table"])

                # ---- Tab 1: Scatter grid (max 2 per row) ----
                with _ctabs[0]:
                    st.caption(
                        "Each panel shows predicted vs actual for that model, "
                        "with a 1:1 ideal line (gold) and an OLS fit line (green)."
                    )
                    _n = len(_selected_compare)
                    _ncols = min(2, _n)
                    _rows = [_selected_compare[i:i + _ncols] for i in range(0, _n, _ncols)]
                    for _row_models in _rows:
                        _cols = st.columns(len(_row_models))
                        for _col, _mname in zip(_cols, _row_models):
                            with _col:
                                st.markdown(f"**{_mname}**")
                                _display_chart_with_downloads(
                                    create_scatter_with_fit(
                                        _sel_results[_mname].full_predictions,
                                        model_name=_mname,
                                    ),
                                    _sel_results[_mname].full_predictions,
                                    key=f"cmp_scatter_{_mname}",
                                    filename_prefix=f"compare_scatter_{_mname}",
                                )

                # ---- Tab 2: Time-series overlay ----
                with _ctabs[1]:
                    st.caption(
                        "All selected model predictions overlaid on a single time-series chart. "
                        "Reference (actual) values are shown in white."
                    )
                    _display_chart_with_downloads(
                        create_multi_model_timeseries(_sel_results), None,
                        key="cmp_ts", filename_prefix="compare_timeseries",
                    )

                # ---- Tab 3: Metrics table ----
                with _ctabs[2]:
                    _metric_keys = ["rmse", "mae", "r2", "mape", "bias", "pearson_r", "slope", "intercept"]
                    _metric_labels = {
                        "rmse": "RMSE ↓",
                        "mae": "MAE ↓",
                        "r2": "R² ↑",
                        "mape": "MAPE ↓",
                        "bias": "Bias → 0",
                        "pearson_r": "Pearson r ↑",
                        "slope": "Slope → 1.0",
                        "intercept": "Intercept → 0",
                    }
                    _cmp_data = {}
                    for _mk in _metric_keys:
                        _cmp_data[_metric_labels[_mk]] = {
                            m: round(float(_sel_results[m].metrics.get(_mk, float("nan"))), 2)
                            for m in _selected_compare
                        }
                    _cmp_df = pd.DataFrame(_cmp_data).T
                    _cmp_df.index.name = "Metric"

                    # Highlight best value per metric
                    def _highlight_best(row: pd.Series) -> list:
                        label = row.name
                        try:
                            if "↓" in label or "→ 0" in label:
                                best_idx = row.abs().idxmin() if "→ 0" in label else row.idxmin()
                            else:
                                best_idx = row.idxmax()
                            return [
                                "background-color: #065f4660; font-weight: bold;" if c == best_idx else ""
                                for c in row.index
                            ]
                        except Exception:
                            return [""] * len(row)

                    st.dataframe(
                        _cmp_df.style.apply(_highlight_best, axis=1),
                        width='stretch',
                    )
                    render_df_download(
                        _cmp_df.reset_index(), key="cmp_metrics_csv",
                        filename="model_comparison_metrics.csv",
                    )
                    st.caption(
                        "**↓** = lower is better  ·  **↑** = higher is better  ·  **→ 0 / → 1** = closer to target is better. "
                        "Best value in each row is highlighted in green."
                    )

                    # Also show a bar chart of key metrics
                    st.markdown("---")
                    st.markdown("**Visual metric comparison:**")
                    _lb_filtered = out["leaderboard"][out["leaderboard"]["model_name"].isin(_selected_compare)]
                    _display_chart_with_downloads(
                        create_multi_model_metrics_bar(_lb_filtered),
                        _format_metric_dataframe(_lb_filtered),
                        key="cmp_metrics_bar", filename_prefix="compare_metrics_bar",
                    )


    st.markdown("---")
    if st.button("Next ➡️", key="next_modeling_always", width='stretch', disabled=(out is None),
                 help="Train models first, then click Next to proceed to Results."):
        go_next()



# ---------------------------------------------------------------------------
# STEP 6: Results
# ---------------------------------------------------------------------------

def render_results():
    st.subheader("🏆 Results & Model Selection")
    out = st.session_state.modeling_outputs
    if out is None or st.session_state.config is None:
        st.info("Complete the **Modelling** step first.")
        return

    model_names = list(out["training_results"].keys())
    sel_name = st.selectbox(
        "Select model to inspect",
        model_names,
        index=model_names.index(st.session_state.selected_model_name)
        if st.session_state.selected_model_name in model_names else 0,
    )
    st.session_state.selected_model_name = sel_name
    result = out["training_results"][sel_name]

    tabs = st.tabs(["📏 Metrics", "🔍 Explainability", "📊 Scatter", "🔀 Multi-Model"])

    with tabs[0]:
        st.markdown("#### Performance Metrics")
        render_metric_row(result.metrics, ["rmse", "mae", "r2", "mape"], ["RMSE", "MAE", "R²", "MAPE"])
        st.markdown("")
        render_metric_row(result.metrics, ["bias", "pearson_r", "slope", "intercept"],
                          ["Bias", "Pearson r", "Slope", "Intercept"])
        st.markdown("#### Full Leaderboard")
        leaderboard_fmt = _format_metric_dataframe(out["leaderboard"])
        st.dataframe(leaderboard_fmt, width='stretch')
        render_df_download(leaderboard_fmt, key="leaderboard_csv", filename="leaderboard.csv")

    with tabs[1]:
        st.markdown("#### Model Explainability")
        if result.feature_importance is not None:
            imp = result.feature_importance
            top_n = min(20, len(imp))
            names = list(imp.keys())[:top_n]
            values = [imp[n] for n in names]
            fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#6366f1"))
            title, x_label, y_label = _chart_customization(
                "res_imp", f"Feature Importance — {sel_name}", "Importance", "Feature"
            )
            fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                              template="plotly_dark", height=max(300, top_n * 26))
            fig.update_yaxes(autorange="reversed")
            imp_df = pd.DataFrame({"feature": names, "importance": values})
            _display_chart_with_downloads(fig, imp_df, key="res_imp", filename_prefix="feature_importance")
        if result.coefficients is not None:
            coefs = result.coefficients
            top_n = min(20, len(coefs))
            names = list(coefs.keys())[:top_n]
            values = [coefs[n] for n in names]
            colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]
            fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
            title, x_label, y_label = _chart_customization(
                "res_coef", f"Coefficients — {sel_name}", "Coefficient", "Feature"
            )
            fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                              template="plotly_dark", height=max(300, top_n * 26))
            fig.update_yaxes(autorange="reversed")
            coef_df = pd.DataFrame({"feature": names, "coefficient": values})
            _display_chart_with_downloads(fig, coef_df, key="res_coef", filename_prefix="coefficients")
            if result.intercept_value is not None:
                st.markdown(info_pill(f"Intercept: {result.intercept_value:.2f}"), unsafe_allow_html=True)
        if result.feature_importance is None and result.coefficients is None:
            st.info("No explainability data available for this model type.")

    with tabs[2]:
        st.markdown("#### Scatter Plot — Predicted vs Actual (with 1:1 & OLS Fit)")
        fig_scatter = create_scatter_with_fit(result.full_predictions, model_name=sel_name)
        title, x_label, y_label = _chart_customization(
            "res_scatter", f"Predicted vs Actual — {sel_name}", "Actual", "Predicted"
        )
        fig_scatter.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
        _display_chart_with_downloads(
            fig_scatter, result.full_predictions, key="res_scatter", filename_prefix="predicted_vs_actual"
        )

    with tabs[3]:
        st.markdown("#### Multi-Model Comparison")
        comp_tabs = st.tabs(["📊 Scatter Grid", "📈 Time-Series Overlay", "📋 Metrics Bar"])
        with comp_tabs[0]:
            _display_chart_with_downloads(
                create_multi_model_scatter(out["training_results"]), None,
                key="res_multi_scatter", filename_prefix="multi_model_scatter",
            )
        with comp_tabs[1]:
            _display_chart_with_downloads(
                create_multi_model_timeseries(out["training_results"]), None,
                key="res_multi_ts", filename_prefix="multi_model_timeseries",
            )
        with comp_tabs[2]:
            _display_chart_with_downloads(
                create_multi_model_metrics_bar(out["leaderboard"]), _format_metric_dataframe(out["leaderboard"]),
                key="res_multi_bar", filename_prefix="multi_model_metrics",
            )

    if st.button("Next ➡️", key="next_results", width='stretch'):
        go_next()




# ---------------------------------------------------------------------------
# STEP 10: Statistical Diagnostics
# ---------------------------------------------------------------------------

def render_statistical_diagnostics():
    st.subheader("\U0001f4d0 Statistical Diagnostics")
    out = st.session_state.modeling_outputs
    if out is None or st.session_state.config is None:
        st.info("Complete the **Modelling** step first.")
        return

    eligible = [
        model_name for model_name in ["ols_regression", "multiple_linear_regression"]
        if model_name in out["training_results"]
    ]
    if not eligible:
        st.info("Train OLS Regression or Multiple Linear Regression to view statistical diagnostics.")
        return

    selected = st.selectbox(
        "Diagnostic model",
        eligible,
        index=eligible.index(st.session_state.selected_model_name)
        if st.session_state.selected_model_name in eligible else 0,
        format_func=_model_label,
        key="diagnostic_model_select",
    )
    result = out["training_results"][selected]
    model_df = out["featured_data"]
    feature_names = [column for column in result.feature_names if column in model_df.columns]

    coefficient_table = _format_coefficient_table(result.coefficient_table)
    vif_table = compute_vif(model_df[feature_names]) if feature_names else pd.DataFrame(columns=["Variable", "VIF"])
    shapiro_result = shapiro_wilk_test(result.residuals if result.residuals is not None else pd.Series(dtype=float))
    st.session_state.diagnostics_outputs = {
        "model_name": selected,
        "coefficient_table": coefficient_table,
        "vif_table": vif_table,
        "shapiro_wilk": shapiro_result,
    }

    tab1, tab2, tab3 = st.tabs(["Coefficient Table", "VIF Table", "Shapiro-Wilk Test"])
    with tab1:
        st.dataframe(coefficient_table, width='stretch')
        render_df_download(
            coefficient_table,
            key="coefficient_table_download",
            filename=f"{selected}_coefficient_table.csv",
        )
    with tab2:
        st.dataframe(vif_table, width='stretch')
        render_df_download(
            vif_table,
            key="vif_table_download",
            filename=f"{selected}_vif_table.csv",
        )
    with tab3:
        c1, c2 = st.columns(2)
        c1.metric("Statistic", f"{shapiro_result.get('statistic', float('nan')):.2f}")
        c2.metric("P-value", f"{shapiro_result.get('p_value', float('nan')):.4f}")

    if st.button("Next \u27a1\ufe0f", key="next_statistical_diagnostics", width='stretch'):
        go_next()

# ---------------------------------------------------------------------------
# STEP 11: Residual Analysis
# ---------------------------------------------------------------------------

def render_residual_analysis():
    st.subheader("🔬 Residual Analysis")
    out = st.session_state.modeling_outputs
    if out is None or st.session_state.config is None:
        st.info("Complete the **Modelling** step first.")
        return

    sel_name = st.session_state.selected_model_name or out["best_model_name"]
    result = out["training_results"][sel_name]
    predictions_df = result.full_predictions

    # Plot style options
    PLOT_STYLES = {"Scatter": "markers", "Connected Line": "lines", "Dotted Line": "lines"}
    PLOT_DASHES = {"Scatter": None, "Connected Line": None, "Dotted Line": "dot"}

    def _apply_plot_style(fig, style_name):
        """Apply the selected plot style to the first data trace."""
        mode = PLOT_STYLES.get(style_name, "markers")
        dash = PLOT_DASHES.get(style_name)
        if fig.data:
            fig.data[0].mode = mode
            if dash:
                fig.data[0].line = dict(dash=dash)
        return fig

    tabs = st.tabs([
        "📊 Predicted vs Actual",
        "📉 Residual vs Fitted",
        "📊 Residual Histogram",
        "📐 QQ Plot",
    ])

    # --- Tab 1: Predicted vs Actual ---
    with tabs[0]:
        title_1, x_1, y_1 = _chart_customization(
            "ra_pva", "Predicted vs Actual", "Actual (Reference)", "Predicted (Calibrated)",
        )
        style_1 = st.selectbox("Plot style", list(PLOT_STYLES.keys()), key="ra_pva_style")
        fig_pva = create_predicted_vs_actual_figure(predictions_df)
        fig_pva.update_layout(title=title_1, xaxis_title=x_1, yaxis_title=y_1)
        _apply_plot_style(fig_pva, style_1)
        source_pva = predictions_df[["actual", "predicted"]].copy()
        _display_chart_with_downloads(fig_pva, source_pva, key="ra_pva", filename_prefix="predicted_vs_actual")

    # --- Tab 2: Residual vs Fitted ---
    with tabs[1]:
        title_2, x_2, y_2 = _chart_customization(
            "ra_rvf", "Residual vs Fitted", "Predicted Value", "Residual (Predicted − Actual)",
        )
        style_2 = st.selectbox("Plot style", list(PLOT_STYLES.keys()), key="ra_rvf_style")
        fig_rvf = create_residual_vs_predicted_figure(predictions_df)
        fig_rvf.update_layout(title=title_2, xaxis_title=x_2, yaxis_title=y_2)
        _apply_plot_style(fig_rvf, style_2)
        source_rvf = pd.DataFrame({
            "predicted": predictions_df["predicted"],
            "residual": predictions_df["predicted"] - predictions_df["actual"],
        })
        _display_chart_with_downloads(fig_rvf, source_rvf, key="ra_rvf", filename_prefix="residual_vs_fitted")

    # --- Tab 3: Residual Histogram ---
    with tabs[2]:
        title_3, x_3, y_3 = _chart_customization(
            "ra_hist", "Residual Distribution", "Residual (Predicted − Actual)", "Frequency",
        )
        fig_hist = create_residual_histogram(predictions_df)
        fig_hist.update_layout(title=title_3, xaxis_title=x_3, yaxis_title=y_3)
        source_hist = pd.DataFrame({
            "residual": predictions_df["predicted"] - predictions_df["actual"],
        })
        _display_chart_with_downloads(fig_hist, source_hist, key="ra_hist", filename_prefix="residual_histogram")

    # --- Tab 4: QQ Plot ---
    with tabs[3]:
        title_4, x_4, y_4 = _chart_customization(
            "ra_qq", "QQ Plot — Residuals vs Normal Distribution",
            "Theoretical Quantiles", "Sample Quantiles",
        )
        style_4 = st.selectbox("Plot style", list(PLOT_STYLES.keys()), key="ra_qq_style")
        fig_qq = create_qq_plot(predictions_df)
        fig_qq.update_layout(title=title_4, xaxis_title=x_4, yaxis_title=y_4)
        _apply_plot_style(fig_qq, style_4)
        residuals_arr = np.asarray(predictions_df["predicted"] - predictions_df["actual"], dtype=float)
        source_qq = pd.DataFrame({"residual": residuals_arr})
        _display_chart_with_downloads(fig_qq, source_qq, key="ra_qq", filename_prefix="qq_plot")

    st.session_state.residual_analysis_outputs = {"model_name": sel_name}

    if st.button("Next ➡️", key="next_residual_analysis", width='stretch'):
        go_next()


# ---------------------------------------------------------------------------
# STEP 8: Export
# ---------------------------------------------------------------------------

def render_export():
    st.subheader("💾 Export — Zenodo-Ready Outputs")
    out = st.session_state.modeling_outputs
    if out is None or st.session_state.config is None:
        st.info("Complete the **Modelling** step first.")
        return

    config = st.session_state.config
    sel_name = st.session_state.selected_model_name or out["best_model_name"]
    result = out["training_results"][sel_name]
    ts_col = config["data"]["timestamp_column"]
    target_col = st.session_state.selected_target or f"{config['data']['reference_prefix']}_{config['data']['target_column']}"

    calibrated = (
        out["featured_data"][[ts_col, target_col]]
        .merge(
            predict_with_model(result.model, out["featured_data"], target_col, ts_col),
            on=ts_col, how="left",
        )
        .rename(columns={target_col: "reference_value", "prediction": "calibrated_value"})
    )

    st.markdown(info_pill(f"Model: {sel_name}") + info_pill(f"Features: {len(result.feature_names)}"),
                unsafe_allow_html=True)
    st.dataframe(calibrated.head(20), width='stretch')
    render_df_download(calibrated, key="calibrated_csv", filename="calibrated_dataset.csv")

    if st.button("🚀 Prepare Export Bundle", key="run_export", width='stretch'):
        with st.spinner("Packaging artefacts…"):
            try:
                bundle = build_export_bundle(
                    calibrated_dataset=calibrated,
                    selected_model=result.model,
                    model_name=sel_name,
                    metrics=result.metrics,
                    feature_names=result.feature_names,
                    config=config,
                    coefficient_table=getattr(result, "coefficient_table", None),
                    selected_target=st.session_state.selected_target,
                    selected_predictors=st.session_state.selected_predictors,
                    modelling_objective=config.get("modelling", {}).get("objective"),
                )
                st.session_state.export_bundle = bundle
                st.success("✅ Export bundle ready!")
            except Exception as e:
                st.error(f"❌ {e}")

    bundle = st.session_state.export_bundle
    if bundle is not None:
        st.markdown("### 📦 Download Artefacts")
        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 Calibrated Dataset (CSV)", bundle["calibrated_dataset_csv"],
                           file_name="calibrated_dataset.csv", mime="text/csv", width='stretch')
        c2.download_button("🤖 Trained Model (.pkl)", bundle["model_pickle"],
                           file_name=f"{sel_name}.pkl", mime="application/octet-stream", width='stretch')
        c3.download_button("📊 Metrics (JSON)", bundle["metrics_json"],
                           file_name="metrics.json", mime="application/json", width='stretch')
        c4, c5, c6 = st.columns(3)
        c4.download_button("⚙️ Config (JSON)", bundle["config_json"],
                           file_name="config.json", mime="application/json", width='stretch')
        c5.download_button("⚙️ Config (YAML)", bundle["config_yaml"],
                           file_name="config.yaml", mime="text/yaml", width='stretch')
        c6.download_button("📋 Metadata (JSON)", bundle["metadata_json"],
                           file_name="metadata.json", mime="application/json", width='stretch')
        c7, c8, c9 = st.columns(3)
        c7.download_button("📄 project_run.json", bundle["project_run_json"],
                           file_name="project_run.json", mime="application/json", width='stretch')
        if "model_summary_pdf" in bundle:
            c8.download_button("📑 PDF Model Report", bundle["model_summary_pdf"],
                               file_name=f"{sel_name}_report.pdf", mime="application/pdf", width='stretch')


# ---------------------------------------------------------------------------
# STEP 9: README (editable, persistent across steps)
# ---------------------------------------------------------------------------

_DEFAULT_README = """\
# Air Quality Sensor Calibration Lab — User Guide
**Version 5.0** | Research-grade ML pipeline for calibrating low-cost air quality sensors

---

## Quick-Start

| Step | Name | What to do |
|------|------|------------|
| 1 | **Upload Data** | Upload Reference + LCS CSVs, or use bundled samples. Set timestamp column, target pollutant, and timezone. |
| 2 | **Preprocessing** | Choose missing-value strategy (interpolate_ffill / ffill / bfill / interpolate) and outlier method for sensor data. Optionally apply to reference too. |
| 3 | **Alignment** | Resample both datasets to a common frequency, auto-detect time lag, and merge. |
| 4 | **EDA** | Explore distributions, correlations, missing-value heatmap, and anomaly detection. |
| 5 | **Modelling** | Configure lag/rolling/polynomial/interaction/time features; select variables; tune hyperparameters; train and compare models. |
| 6 | **Results** | Inspect leaderboard, feature importance, scatter + time-series per model. |
| 7 | **Post-Analysis** | Rolling error drift, Bland-Altman agreement, residual plots. |
| 8 | **Export** | Download calibrated CSV, model .pkl, metrics JSON, config YAML/JSON, metadata JSON. |

---

## Missing-Value Strategies

| Strategy | What it does | Best for |
|----------|-------------|----------|
| `interpolate_ffill` *(recommended)* | Linear interpolation, then ffill+bfill for edge gaps | Most datasets — handles interior + edge gaps |
| `interpolate` | Linear interpolation only | When you want to leave edge NaNs and inspect them |
| `ffill` | Forward-fill → bfill for leading gaps | Step-change sensors (value holds until next reading) |
| `bfill` | Backward-fill → ffill for trailing gaps | When the next reading is more representative |

---

## Lag & Rolling Features

**What is a lag feature?** A lag feature copies a sensor column from N time-steps *in the past*
and gives it to the model as a new input. This lets the model learn from recent history.

| Setting | Example (hourly data) | Guidance |
|---------|----------------------|---------|
| `lag_steps = 1,2` | 1h ago, 2h ago | ✅ Good for AQ — captures recent pollution history |
| `lag_steps = 6` | 6h ago | ⚠️ Usually meaningless for PM2.5 — events don't persist that long |
| `rolling_windows = 3,6` | 3h / 6h moving avg | Smooths spikes; captures trend |

> ⚠️ Each lag step removes one row from the top via `dropna()`. On short datasets (< 200 rows) keep lags to 1–2.

---

## Feature Engineering Options

| Feature | Description |
|---------|-------------|
| Lag features | Shifted copies of sensor columns (configurable steps) |
| Rolling mean/std | Moving average ± volatility windows |
| Polynomial (deg 2/3) | Squared/cubed terms + cross-products for chosen columns |
| Interaction terms | Pairwise products (col_i × col_j) — useful for PM × RH hygroscopic correction |
| Hour/DOW/DOM | Basic time of day / day of week / day of month |
| Cyclical encodings | sin/cos of hour, DOW, DOY — avoids ordinal discontinuity in linear models |
| Season | DJF / MAM / JJA / SON encoded as 0–3 |
| Day name + weekend | Mon–Sun label + binary is_weekend flag |

---

## Available Models

| Model | Type | Key hyperparameters | Best for |
|-------|------|--------------------|---------|
| Linear Regression | Linear (OLS) | None | Baseline; fully interpretable |
| Ridge | Linear + L2 | `alpha` | Many correlated features (lag/rolling) |
| Lasso | Linear + L1 | `alpha` | Automatic feature selection |
| Random Forest | Tree ensemble | `n_estimators`, `max_depth` | Non-linear PM–humidity relationships |
| XGBoost | Gradient boosting | `n_estimators`, `max_depth`, `learning_rate` | Best accuracy on large datasets (> 500 rows) |

> 💡 Open the **📚 Model Reference Guide** in the Modelling step for full formulas and hyperparameter tables.

---

## Calibration Metrics

| Metric | Ideal | Description |
|--------|-------|-------------|
| RMSE | → 0 | Root mean squared error (same units as pollutant) |
| MAE | → 0 | Mean absolute error |
| R² | → 1 | Coefficient of determination |
| MAPE | → 0 | Mean absolute percentage error (%) |
| Bias | → 0 | Mean error (predicted − actual); sign shows direction |
| Pearson r | → 1 | Linear correlation coefficient |
| Slope | → 1.0 | OLS fit slope — 1.0 means no scale error |
| Intercept | → 0.0 | OLS fit intercept — 0 means no constant offset |

---

## Tips & Notes

- **XGBoost** requires `pip install xgboost`. The other 4 models work without it.
- **Auto-tuning** uses RandomizedSearchCV with TimeSeriesSplit — enable per model; manual hyperparameters are ignored for tuned models.
- **Model Comparison**: After training, open the **🔀 Compare Models** expander to select any N models and compare scatter, time-series, and metrics side-by-side.
- **Feature preview** lets you see all engineered columns before selecting a subset for training.
- **Variable selection** is reset each time you click *Preview Features* — re-select your subset if needed.
- **Reference preprocessing**: missing-value imputation always applies; outlier removal is opt-in.
- **Bland-Altman** plot: toggle in Post-Analysis settings. Points within ±1.96σ = good agreement.
- All exports include provenance metadata (model name, features used, metrics, config) for reproducibility.
- Hover over any **?** icon in the UI for contextual guidance.

---

## My Notes
*(Edit this section freely — your changes persist for the session.)*

"""


def render_readme():
    st.subheader("📖 README & User Guide")

    # Initialise content from README file if not already in state
    if st.session_state.readme_content is None:
        try:
            readme_path = PROJECT_ROOT / "README.md"
            base = readme_path.read_text(encoding="utf-8") if readme_path.exists() else _DEFAULT_README
        except Exception:
            base = _DEFAULT_README
        st.session_state.readme_content = base

    # ---- Read-only rendering ----
    st.caption("Read-only documentation. Download the Markdown file using the button below.")
    st.markdown(st.session_state.readme_content, unsafe_allow_html=False)

    # ---- Download button ----
    st.download_button(
        "💾 Download Markdown",
        data=st.session_state.readme_content.encode("utf-8"),
        file_name="calibration_lab_guide.md",
        mime="text/markdown",
        width='stretch',
        key="readme_download_md",
    )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Air Quality Calibration Lab",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_header()
    init_state()

    st.session_state.current_step = step_nav()
    current_idx = STEPS.index(st.session_state.current_step)
    render_step_progress(current_idx)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.caption("v5.0 · Calibration Lab")
    if st.session_state.config:
        st.sidebar.caption(f"Target: {st.session_state.config['data']['target_column']}")
    if st.session_state.selected_model_name:
        st.sidebar.caption(f"Model: {st.session_state.selected_model_name}")

    dispatch = {
        STEPS[0]: render_upload,
        STEPS[1]: render_preprocessing,
        STEPS[2]: render_alignment,
        STEPS[3]: render_eda,
        STEPS[4]: render_variable_selection,
        STEPS[5]: render_feature_engineering,
        STEPS[6]: render_normalization,
        STEPS[7]: render_modelling,
        STEPS[8]: render_results,
        STEPS[9]: render_statistical_diagnostics,
        STEPS[10]: render_residual_analysis,
        STEPS[11]: render_export,
        STEPS[12]: render_readme,
    }
    dispatch[st.session_state.current_step]()


if __name__ == "__main__":
    main()

