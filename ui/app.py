"""Premium 8-step Streamlit UI for the Air Quality Sensor Calibration Lab.

Provides a refined, research-grade workflow with extensive user controls,
dark-themed premium styling, and interactive Plotly visualisations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from modules.drift_analysis import generate_post_analysis_outputs
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
    "🤖 Modelling",
    "🏆 Results",
    "🔬 Post-Analysis",
    "💾 Export",
    "📖 README",
]

STEP_KEYS = [
    "upload", "preprocessing", "alignment", "eda",
    "modelling", "results", "post_analysis", "export", "readme",
]

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
            txt = f"{val:.4f}"
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


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "current_step": STEPS[0],
        "config": None,
        "input_label": None,
        "data_outputs": None,
        "preprocessing_outputs": None,
        "alignment_outputs": None,
        "eda_outputs": None,
        "modeling_outputs": None,
        "selected_model_name": None,
        "post_analysis_outputs": None,
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
        c1.dataframe(r.head(10), use_container_width=True)
        c2.caption("LCS (first 10 rows)")
        c2.dataframe(s.head(10), use_container_width=True)

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

    if st.button("🚀 Load & Validate Data", key="run_upload", use_container_width=True,
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
            for k in ["preprocessing_outputs","alignment_outputs","eda_outputs","modeling_outputs","post_analysis_outputs","export_bundle"]:
                st.session_state[k] = None
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

        tab1, tab2 = st.tabs(["Reference Data", "Sensor Data"])
        with tab1:
            st.dataframe(st.session_state.data_outputs["reference_raw"].head(20), use_container_width=True)
        with tab2:
            st.dataframe(st.session_state.data_outputs["sensor_raw"].head(20), use_container_width=True)

        if st.button("Next ➡️", key="next_upload", use_container_width=True):
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

    with st.expander("⚙️ Sensor Data Preprocessing", expanded=True):
        st.caption(
            "Controls how the **sensor (LCS) data** is cleaned before alignment. "
            "Missing values are filled first, then statistical outliers are removed."
        )
        c1, c2, c3 = st.columns(3)
        _MISSING_OPTIONS = ["interpolate_ffill", "interpolate", "ffill", "bfill"]
        _cur_missing = str(config["preprocessing"].get("missing_strategy", "interpolate_ffill"))
        _missing_idx = _MISSING_OPTIONS.index(_cur_missing) if _cur_missing in _MISSING_OPTIONS else 0
        missing_method = c1.selectbox(
            "Missing value strategy",
            _MISSING_OPTIONS,
            index=_missing_idx,
            help=(
                "How to fill gaps (NaN) in the sensor data:\n\n"
                "• **interpolate_ffill** *(recommended)* — linearly interpolates between known values, "
                "then forward-fills/back-fills any remaining edge gaps. Most robust.\n"
                "• **interpolate** — linear interpolation only. May leave NaNs at the very "
                "start or end of the series if there is no surrounding data.\n"
                "• **ffill** — copies the last known value forward (and back-fills leading gaps). "
                "Good for step-like signals but can introduce flat periods.\n"
                "• **bfill** — copies the next known value backward (and forward-fills trailing gaps). "
                "Useful when future values are more reliable than past ones."
            ),
        )
        outlier_method = c2.selectbox(
            "Outlier removal method",
            ["iqr", "zscore", "none"],
            index=["iqr", "zscore", "none"].index(str(config["preprocessing"].get("outlier_method", "iqr"))),
            help=(
                "Statistical method used to flag and remove extreme sensor readings:\n\n"
                "• **iqr** *(recommended)* — removes rows where any numeric column falls outside "
                "Q1 − threshold×IQR or Q3 + threshold×IQR. Robust to non-normal distributions.\n"
                "• **zscore** — removes rows whose z-score (standard deviations from the mean) "
                "exceeds the threshold. Sensitive to outliers in small datasets.\n"
                "• **none** — no outlier removal applied."
            ),
        )
        outlier_threshold = c3.number_input(
            "Outlier threshold",
            min_value=0.5, max_value=10.0,
            value=float(config["preprocessing"].get("outlier_threshold", 1.5)),
            step=0.1,
            help=(
                "Sensitivity of the outlier detector.\n\n"
                "• For **IQR**: multiplier applied to the interquartile range. "
                "1.5 = standard box-plot rule; 3.0 = only extreme outliers removed.\n"
                "• For **z-score**: maximum allowed standard deviations from the mean. "
                "2.0 = ~5 % of data flagged; 3.0 = ~0.3 % flagged.\n"
                "Lower values = more aggressive removal."
            ),
        )
        config["preprocessing"]["missing_strategy"] = missing_method
        config["preprocessing"]["outlier_method"] = outlier_method
        config["preprocessing"]["outlier_threshold"] = outlier_threshold

    with st.expander("⚙️ Reference Data Preprocessing", expanded=False):
        st.caption(
            "Missing-value imputation is **always** applied to both datasets. "
            "Toggle below to also apply outlier removal to the reference data. "
            "The target column (e.g. reference PM2.5) is always protected from removal."
        )
        st.info(
            "💡 Reference instruments are high-accuracy devices — their readings should rarely "
            "need outlier removal. Only enable this if you know your reference has sensor noise."
        )
        apply_to_ref = st.checkbox(
            "Apply outlier removal to reference data",
            value=bool(config["preprocessing"].get("apply_to_reference", False)),
            key="apply_ref_outlier",
        )
        config["preprocessing"]["apply_to_reference"] = apply_to_ref
        if apply_to_ref:
            st.info(
                f"Outlier method **{outlier_method}** (threshold {outlier_threshold}) "
                "will be applied to reference numeric columns (excluding the target column)."
            )
        else:
            st.success("Reference outlier removal is **off** — recommended for most workflows.")

    st.session_state.config = config

    if st.button("🚀 Run Preprocessing", key="run_preprocess", use_container_width=True):
        with st.spinner("Cleaning datasets…"):
            try:
                cfg_text = _cfg_to_json(config)
                outputs = cached_preprocessing(
                    st.session_state.data_outputs["reference_raw"],
                    st.session_state.data_outputs["sensor_raw"],
                    cfg_text,
                )
                st.session_state.preprocessing_outputs = outputs
                for k in ["alignment_outputs","eda_outputs","modeling_outputs","post_analysis_outputs","export_bundle"]:
                    st.session_state[k] = None
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
            st.dataframe(out["reference_processed"].head(20), use_container_width=True)
        with tab2:
            st.dataframe(out["sensor_processed"].head(20), use_container_width=True)

        if st.button("Next ➡️", key="next_preprocess", use_container_width=True):
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
            value=int(config["alignment"].get("max_lag_steps", 3)),
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

    if st.button("🚀 Run Alignment", key="run_alignment", use_container_width=True):
        with st.spinner("Aligning datasets…"):
            try:
                cfg_text = _cfg_to_json(config)
                outputs = cached_alignment(
                    st.session_state.preprocessing_outputs["reference_processed"],
                    st.session_state.preprocessing_outputs["sensor_processed"],
                    cfg_text,
                )
                st.session_state.alignment_outputs = outputs
                for k in ["eda_outputs","modeling_outputs","post_analysis_outputs","export_bundle"]:
                    st.session_state[k] = None
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
        st.dataframe(out["merged_data"].head(20), use_container_width=True)

        if st.button("Next ➡️", key="next_alignment", use_container_width=True):
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

    if st.button("🚀 Generate EDA", key="run_eda", use_container_width=True):
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
                st.plotly_chart(fig, use_container_width=True)
            if "before_after_figure" in out:
                st.plotly_chart(out["before_after_figure"], use_container_width=True)

        with tabs[1]:
            st.plotly_chart(out["correlation_figure"], use_container_width=True)

        with tabs[2]:
            st.plotly_chart(out["missing_heatmap"], use_container_width=True)
            st.dataframe(out["missing_summary"], use_container_width=True)

        with tabs[3]:
            st.plotly_chart(out["time_series_figure"], use_container_width=True)

        with tabs[4]:
            st.plotly_chart(out["anomaly_figure"], use_container_width=True)
            if not out["anomalies"].empty:
                st.caption(f"Detected {len(out['anomalies'])} anomalous rows")
                st.dataframe(out["anomalies"].head(25), use_container_width=True)
            else:
                st.success("No anomalies detected.")

        if st.button("Next ➡️", key="next_eda", use_container_width=True):
            go_next()


# ---------------------------------------------------------------------------
# STEP 5: Modelling
# ---------------------------------------------------------------------------

def render_modelling():
    st.subheader("🤖 Modelling")
    if st.session_state.alignment_outputs is None or st.session_state.config is None:
        st.info("Complete the **Alignment** step first.")
        return

    config = st.session_state.config
    merged = st.session_state.alignment_outputs["merged_data"]
    numeric_merged_cols = merged.select_dtypes(include="number").columns.tolist()
    target_col_full = f"{config['data']['reference_prefix']}_{config['data']['target_column']}"
    sensor_base_cols = [c for c in numeric_merged_cols if c != target_col_full]

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

    with st.expander("⚙️ Lag & Rolling Features", expanded=False):
        st.caption(
            "💡 **What are lag features?** A lag feature copies the value of a sensor column from "
            "N time-steps *in the past* and gives it to the model as a new input. "
            "This lets the model learn from recent history — useful when pollution levels "
            "from the previous hour(s) help predict the current reading.\n\n"
            "**Example (hourly data):** lag = 1 means ‘reading 1 hour ago’, lag = 2 means ‘reading 2 hours ago’. "
            "For air quality, lags of 1–2 time-steps are usually most meaningful. "
            "Larger lags (e.g. 6+) rarely help and waste data because `dropna()` removes "
            "the first N rows after lagging."
        )
        c1, c2, c3 = st.columns(3)
        lag_steps_str = c1.text_input(
            "Lag steps (comma-separated)",
            value=",".join(str(s) for s in config["feature_engineering"].get("lag_steps", [1, 2, 3])),
            help=(
                "Integers representing how many time-steps back to look.\n\n"
                "For **hourly data**: ‘1’ = 1h ago, ‘3’ = 3h ago.\n"
                "For **5-min data**: ‘1’ = 5 min ago, ‘6’ = 30 min ago.\n\n"
                "⚠️ **Air quality tip**: lags of 3+ hours are usually meaningless for PM2.5 "
                "calibration — pollution events don’t persist that long in most settings. "
                "Start with ‘1,2’ and add more only if you see improvement in R².\n\n"
                "⚠️ Each lag step removes one row from the top of the dataset via dropna(). "
                "On short datasets (< 200 rows) keep lags small (1–2)."
            ),
        )
        rolling_str = c2.text_input(
            "Rolling windows (comma-separated)",
            value=",".join(str(w) for w in config["feature_engineering"].get("rolling_windows", [3, 6])),
            help=(
                "Window sizes for the rolling mean (and optionally std). "
                "A window of 3 with hourly data gives a 3-hour moving average of the sensor.\n\n"
                "Rolling mean smooths out short spikes and captures trend. "
                "Rolling std measures how variable the sensor has been recently (useful for "
                "detecting unstable conditions).\n\n"
                "Good starting point: ‘3,6’ (short and medium-term smoothing)."
            ),
        )
        rolling_std = c3.checkbox(
            "Include rolling std",
            value=bool(config["feature_engineering"].get("rolling_std", True)),
            help=(
                "Also create rolling standard deviation columns (in addition to rolling mean). "
                "Rolling std captures sensor volatility over the window — a high std means "
                "the sensor was fluctuating a lot recently, which may affect calibration accuracy."
            ),
        )
        try:
            parsed_lags = [int(x.strip()) for x in lag_steps_str.split(",") if x.strip()]
            config["feature_engineering"]["lag_steps"] = parsed_lags
            # Warn if large lags on a small dataset
            n_rows = len(merged)
            max_lag_val = max(parsed_lags) if parsed_lags else 0
            if max_lag_val > 0 and n_rows < max_lag_val * 20:
                st.warning(
                    f"⚠️ Your merged dataset has **{n_rows} rows** and your largest lag is **{max_lag_val}**. "
                    f"After lagging and dropping NaNs, you may lose a significant portion of your data. "
                    f"Consider reducing lag steps or using a finer resample frequency."
                )
        except ValueError:
            st.warning("Invalid lag steps — enter integers separated by commas, e.g. `1,2`.")
        try:
            config["feature_engineering"]["rolling_windows"] = [int(x.strip()) for x in rolling_str.split(",") if x.strip()]
        except ValueError:
            st.warning("Invalid rolling windows — enter integers separated by commas, e.g. `3,6`.")
        config["feature_engineering"]["rolling_std"] = rolling_std

    with st.expander("⚙️ Polynomial & Interaction Features", expanded=False):
        st.caption(
            "Creates non-linear versions of your sensor columns so that linear models "
            "(Ridge, Lasso) can fit curved relationships. Tree-based models (RF, XGBoost) "
            "don’t usually benefit from these, but they won’t hurt either."
        )
        poly_degree = st.selectbox(
            "Polynomial degree",
            [1, 2, 3],
            index=[1,2,3].index(int(config["feature_engineering"].get("polynomial_degree", 1))),
            help=(
                "• **1** — no expansion (columns are used as-is).\n"
                "• **2** — adds squared terms (col²) and pairwise products (col_a × col_b) "
                "for all selected columns. Good for capturing gentle curves.\n"
                "• **3** — adds cubic terms too. Powerful but risks overfitting on small datasets. "
                "Use with regularised models (Ridge/Lasso) if degree 3."
            ),
        )
        poly_cols = st.multiselect(
            "Columns for polynomial expansion",
            options=sensor_base_cols,
            default=[c for c in config["feature_engineering"].get("polynomial_columns", []) if c in sensor_base_cols],
            help=(
                "Pick which sensor columns to expand into polynomial features. "
                "Applying to all columns at once can create a very large feature space — "
                "select only the most important 1–3 columns (e.g. main PM or humidity channel)."
            ),
        )
        interaction_cols = st.multiselect(
            "Columns for pairwise interaction terms",
            options=sensor_base_cols,
            default=[c for c in config["feature_engineering"].get("interaction_columns", []) if c in sensor_base_cols],
            help=(
                "Creates col_i × col_j for every pair in the selection. "
                "Useful when you believe two variables interact (e.g. PM × humidity for "
                "hygroscopic correction). Selecting N columns creates N×(N-1)/2 new features."
            ),
        )
        config["feature_engineering"]["polynomial_degree"] = poly_degree
        config["feature_engineering"]["polynomial_columns"] = poly_cols
        config["feature_engineering"]["interaction_columns"] = interaction_cols

    with st.expander("⚙️ Time & Date Features", expanded=False):
        st.caption(
            "Adds date/time-derived columns to help models capture diurnal cycles and seasonal patterns. "
            "Air quality often has strong time-of-day and day-of-week patterns "
            "(e.g. traffic peaks, cooking, industrial schedules)."
        )
        existing_tf = config["feature_engineering"].get("time_feature_flags", {})
        add_time = st.checkbox(
            "Enable time features",
            value=bool(config["feature_engineering"].get("add_time_features", True)),
            help="Adds hour-of-day, day-of-week, and day-of-month columns automatically. Extra features below are optional.",
        )
        config["feature_engineering"]["add_time_features"] = add_time
        if add_time:
            st.caption("Basic features (hour 0-23, day-of-week 0-6, day-of-month 1-31) always included. Enable extras:")
            c1, c2, c3 = st.columns(3)
            tf = {"hour_of_day": True, "day_of_week": True, "day_of_month": True}
            tf["unix_timestamp"] = c1.checkbox("Unix timestamp",        value=existing_tf.get("unix_timestamp", False),
                help="Seconds since 1970-01-01. Useful as a linear time trend proxy for tree models.")
            tf["julian_date"]    = c1.checkbox("Julian date (DOY)",     value=existing_tf.get("julian_date", False),
                help="Day-of-year (1-366). Captures seasonal variation without cyclical encoding.")
            tf["calendar_date"]  = c1.checkbox("Calendar date (int)",   value=existing_tf.get("calendar_date", False),
                help="Integer days since 1970-01-01, day-resolution. Good for long-term trend.")
            tf["cyclical_hour"]  = c2.checkbox("Cyclical hour sin/cos", value=existing_tf.get("cyclical_hour", False),
                help="Encodes hour as sin/cos so hour 23 is treated as close to hour 0. Better than raw integer for linear models.")
            tf["cyclical_dow"]   = c2.checkbox("Cyclical DOW sin/cos",  value=existing_tf.get("cyclical_dow", False),
                help="Circular day-of-week encoding. Sunday wraps around to Monday correctly.")
            tf["cyclical_doy"]   = c2.checkbox("Cyclical DOY sin/cos",  value=existing_tf.get("cyclical_doy", False),
                help="Circular day-of-year encoding. Dec 31 treated as adjacent to Jan 1.")
            tf["season"]         = c3.checkbox("Season (DJF/MAM/JJA/SON)", value=existing_tf.get("season", False),
                help="Meteorological season as 0-3: DJF(winter)=0, MAM(spring)=1, JJA(summer)=2, SON(autumn)=3.")
            tf["day_name"]       = c3.checkbox("Day name & weekend flag",   value=existing_tf.get("day_name", False),
                help="Adds day-of-week number (0-6) and binary is_weekend (1=Sat/Sun). Traffic-related PM differs on weekends.")
            config["feature_engineering"]["time_feature_flags"] = tf

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
        all_models = ["linear_regression", "ridge", "lasso", "random_forest", "xgboost"]
        current = config["training"].get("selected_models", all_models)
        selected_models = c3.multiselect(
            "Models to train",
            all_models,
            default=current,
            help=(
                "Choose which models to train. All selected models run in sequence and appear in the leaderboard.\n\n"
                "• **linear_regression** — baseline OLS, no regularisation\n"
                "• **ridge** — L2 regularisation, shrinks coefficients smoothly\n"
                "• **lasso** — L1 regularisation, can zero out irrelevant features\n"
                "• **random_forest** — ensemble of decision trees, handles non-linearity\n"
                "• **xgboost** — gradient-boosted trees, often best on tabular data (requires install)"
            ),
        )
        config["training"]["test_size"] = test_size
        config["training"]["cross_validation_folds"] = cv_folds
        config["training"]["selected_models"] = selected_models if selected_models else all_models

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

    # ---- Feature preview & variable selection ----
    st.markdown("---")
    st.markdown("### 🔢 Variable Selection")
    st.caption("Preview engineered features, then pick exactly which columns to pass to the models.")
    if st.button("👁️ Preview Features", key="preview_features"):
        from modules.feature_engineering import engineer_features
        try:
            preview = engineer_features(
                dataframe=merged,
                timestamp_column=config["data"]["timestamp_column"],
                target_column=target_col_full,
                config=config["feature_engineering"],
            )
            st.session_state.featured_preview = preview
            st.session_state.selected_features = None
        except Exception as e:
            st.error(f"❌ Feature preview error: {e}")

    if st.session_state.featured_preview is not None:
        fp = st.session_state.featured_preview
        all_feat_cols = [c for c in fp.columns if c not in [config["data"]["timestamp_column"], target_col_full]]
        st.markdown(info_pill(f"Available features: {len(all_feat_cols)}"), unsafe_allow_html=True)
        with st.expander("📋 Feature preview (first 5 rows)", expanded=False):
            st.dataframe(fp[all_feat_cols].head(5), use_container_width=True)
        prev_sel = st.session_state.selected_features or all_feat_cols
        valid_prev = [c for c in prev_sel if c in all_feat_cols]
        selected = st.multiselect(
            "Select features for training (empty = use all)",
            options=all_feat_cols, default=valid_prev, key="feat_multiselect",
        )
        st.session_state.selected_features = selected if selected else None

    st.markdown("---")
    if st.button("🚀 Train Models", key="run_modeling", use_container_width=True):
        with st.spinner("Training models… This may take a moment."):
            try:
                cfg_text = _cfg_to_json(config)
                subset_json = json.dumps(st.session_state.selected_features)
                outputs = cached_modeling(merged, cfg_text, subset_json)
                st.session_state.modeling_outputs = outputs
                st.session_state.selected_model_name = outputs["best_model_name"]
                for k in ["post_analysis_outputs", "export_bundle"]:
                    st.session_state[k] = None
                st.success("✅ All models trained successfully!")
            except Exception as e:
                st.error(f"❌ {e}")

    out = st.session_state.modeling_outputs
    if out is not None:
        st.markdown("### 🏅 Leaderboard")
        st.dataframe(
            out["leaderboard"].style.highlight_min(subset=["rmse", "mae"], color="#065f4630")
            .highlight_max(subset=["r2", "pearson_r"], color="#065f4630"),
            use_container_width=True,
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
                                st.plotly_chart(
                                    create_scatter_with_fit(
                                        _sel_results[_mname].full_predictions,
                                        model_name=_mname,
                                    ),
                                    use_container_width=True,
                                )

                # ---- Tab 2: Time-series overlay ----
                with _ctabs[1]:
                    st.caption(
                        "All selected model predictions overlaid on a single time-series chart. "
                        "Reference (actual) values are shown in white."
                    )
                    st.plotly_chart(
                        create_multi_model_timeseries(_sel_results),
                        use_container_width=True,
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
                            m: round(float(_sel_results[m].metrics.get(_mk, float("nan"))), 4)
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
                        use_container_width=True,
                    )
                    st.caption(
                        "**↓** = lower is better  ·  **↑** = higher is better  ·  **→ 0 / → 1** = closer to target is better. "
                        "Best value in each row is highlighted in green."
                    )

                    # Also show a bar chart of key metrics
                    st.markdown("---")
                    st.markdown("**Visual metric comparison:**")
                    _lb_filtered = out["leaderboard"][out["leaderboard"]["model_name"].isin(_selected_compare)]
                    st.plotly_chart(
                        create_multi_model_metrics_bar(_lb_filtered),
                        use_container_width=True,
                    )


    st.markdown("---")
    if st.button("Next ➡️", key="next_modeling_always", use_container_width=True, disabled=(out is None),
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
        st.dataframe(out["leaderboard"], use_container_width=True)

    with tabs[1]:
        st.markdown("#### Model Explainability")
        if result.feature_importance is not None:
            imp = result.feature_importance
            top_n = min(20, len(imp))
            names = list(imp.keys())[:top_n]
            values = [imp[n] for n in names]
            fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#6366f1"))
            fig.update_layout(title=f"Feature Importance — {sel_name}", xaxis_title="Importance",
                              yaxis=dict(autorange="reversed"), template="plotly_dark",
                              height=max(300, top_n * 26))
            st.plotly_chart(fig, use_container_width=True)
        if result.coefficients is not None:
            coefs = result.coefficients
            top_n = min(20, len(coefs))
            names = list(coefs.keys())[:top_n]
            values = [coefs[n] for n in names]
            colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]
            fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
            fig.update_layout(title=f"Coefficients — {sel_name}", xaxis_title="Coefficient",
                              yaxis=dict(autorange="reversed"), template="plotly_dark",
                              height=max(300, top_n * 26))
            st.plotly_chart(fig, use_container_width=True)
            if result.intercept_value is not None:
                st.markdown(info_pill(f"Intercept: {result.intercept_value:.4f}"), unsafe_allow_html=True)
        if result.feature_importance is None and result.coefficients is None:
            st.info("No explainability data available for this model type.")

    with tabs[2]:
        st.markdown("#### Scatter Plot — Predicted vs Actual (with 1:1 & OLS Fit)")
        fig_scatter = create_scatter_with_fit(result.full_predictions, model_name=sel_name)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tabs[3]:
        st.markdown("#### Multi-Model Comparison")
        comp_tabs = st.tabs(["📊 Scatter Grid", "📈 Time-Series Overlay", "📋 Metrics Bar"])
        with comp_tabs[0]:
            st.plotly_chart(create_multi_model_scatter(out["training_results"]), use_container_width=True)
        with comp_tabs[1]:
            st.plotly_chart(create_multi_model_timeseries(out["training_results"]), use_container_width=True)
        with comp_tabs[2]:
            st.plotly_chart(create_multi_model_metrics_bar(out["leaderboard"]), use_container_width=True)

    if st.button("Next ➡️", key="next_results", use_container_width=True):
        go_next()


# ---------------------------------------------------------------------------
# STEP 7: Post-Analysis
# ---------------------------------------------------------------------------

def render_post_analysis():
    st.subheader("🔬 Post-Calibration Analysis")
    out = st.session_state.modeling_outputs
    if out is None or st.session_state.config is None:
        st.info("Complete the **Modelling** step first.")
        return

    config = st.session_state.config
    sel_name = st.session_state.selected_model_name or out["best_model_name"]
    result = out["training_results"][sel_name]

    with st.expander("⚙️ Drift Analysis Settings", expanded=False):
        c1, c2 = st.columns(2)
        rolling_window = c1.number_input("Rolling window", 3, 48,
                                         int(config.get("drift_analysis", {}).get("rolling_window", 6)))
        drift_thresh = c2.number_input("Drift threshold multiplier", 1.0, 5.0,
                                       float(config.get("drift_analysis", {}).get("drift_threshold", 1.5)), 0.1)
        show_ba = st.checkbox("Show Bland-Altman agreement plot", value=False, key="show_bland_altman")
        config.setdefault("drift_analysis", {})["rolling_window"] = rolling_window
        config["drift_analysis"]["drift_threshold"] = drift_thresh
        st.session_state.config = config

    if st.button("🚀 Run Post-Analysis", key="run_post", use_container_width=True):
        with st.spinner("Analysing calibration quality…"):
            try:
                pa_out = generate_post_analysis_outputs(
                    result.full_predictions,
                    rolling_window=rolling_window,
                    drift_threshold=drift_thresh,
                )
                st.session_state.post_analysis_outputs = pa_out
                st.success("✅ Post-analysis completed")
            except Exception as e:
                st.error(f"❌ {e}")

    pa = st.session_state.post_analysis_outputs
    if pa is not None:
        tab_labels = ["📊 Pred vs Actual", "📉 Residuals", "📈 Time Series", "🌊 Rolling Error", "📋 Residual Dist"]
        if show_ba:
            tab_labels.append("⚖️ Bland-Altman")
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            # Enhanced scatter with 1:1 line and OLS fit
            fig = create_scatter_with_fit(result.full_predictions, model_name=sel_name)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            st.plotly_chart(pa["residual_plot_fig"], use_container_width=True)
        with tabs[2]:
            st.plotly_chart(pa["time_series_overlay_fig"], use_container_width=True)
        with tabs[3]:
            st.plotly_chart(pa["rolling_error_fig"], use_container_width=True)
            if not pa["drift_periods"].empty:
                st.warning(f"⚠️ Drift detected in {len(pa['drift_periods'])} time steps")
                st.dataframe(
                    pa["drift_periods"][["timestamp", "rolling_rmse", "rolling_bias", "drift_threshold"]].head(20),
                    use_container_width=True,
                )
            else:
                st.success("✅ No significant drift detected.")
        with tabs[4]:
            st.plotly_chart(pa["residual_histogram_fig"], use_container_width=True)
        if show_ba and len(tabs) > 5:
            with tabs[5]:
                fig_ba = create_bland_altman_plot(result.full_predictions, model_name=sel_name)
                st.plotly_chart(fig_ba, use_container_width=True)
                st.caption(
                    "Bland-Altman plot: points should cluster around the mean bias line "
                    "within ±1.96σ limits of agreement."
                )

        if st.button("Next ➡️", key="next_post", use_container_width=True):
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
    target_col = f"{config['data']['reference_prefix']}_{config['data']['target_column']}"

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
    st.dataframe(calibrated.head(20), use_container_width=True)

    if st.button("🚀 Prepare Export Bundle", key="run_export", use_container_width=True):
        with st.spinner("Packaging artefacts…"):
            try:
                bundle = build_export_bundle(
                    calibrated_dataset=calibrated,
                    selected_model=result.model,
                    model_name=sel_name,
                    metrics=result.metrics,
                    feature_names=result.feature_names,
                    config=config,
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
                           file_name="calibrated_dataset.csv", mime="text/csv", use_container_width=True)
        c2.download_button("🤖 Trained Model (.pkl)", bundle["model_pickle"],
                           file_name=f"{sel_name}.pkl", mime="application/octet-stream", use_container_width=True)
        c3.download_button("📊 Metrics (JSON)", bundle["metrics_json"],
                           file_name="metrics.json", mime="application/json", use_container_width=True)
        c4, c5, c6 = st.columns(3)
        c4.download_button("⚙️ Config (JSON)", bundle["config_json"],
                           file_name="config.json", mime="application/json", use_container_width=True)
        c5.download_button("⚙️ Config (YAML)", bundle["config_yaml"],
                           file_name="config.yaml", mime="text/yaml", use_container_width=True)
        c6.download_button("📋 Metadata (JSON)", bundle["metadata_json"],
                           file_name="metadata.json", mime="application/json", use_container_width=True)


# ---------------------------------------------------------------------------
# STEP 9: README (editable, persistent across steps)
# ---------------------------------------------------------------------------

_DEFAULT_README = """\
# Air Quality Sensor Calibration Lab — User Guide
**Version 4.0** | Research-grade ML pipeline for calibrating low-cost air quality sensors

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
        st.session_state.readme_content = base + "\n\n---\n" + _DEFAULT_README.split("## My Notes")[1]

    # ---- Toggle state ----
    if "readme_edit_open" not in st.session_state:
        st.session_state.readme_edit_open = False

    # ---- Top action row: small Edit toggle + download ----
    hdr_l, hdr_r = st.columns([8, 2])
    with hdr_r:
        btn_label = "✏️ Close Editor" if st.session_state.readme_edit_open else "✏️ Edit"
        if st.button(btn_label, key="readme_toggle_edit", use_container_width=True,
                     help="Toggle the Markdown editor on/off"):
            st.session_state.readme_edit_open = not st.session_state.readme_edit_open
            st.rerun()
    with hdr_l:
        st.caption(
            "Live Markdown preview. Click **✏️ Edit** to open the editor. "
            "Your changes persist for the full session."
        )

    # ---- Preview (always visible) ----
    st.markdown(st.session_state.readme_content, unsafe_allow_html=False)

    # ---- Editor panel (shown only when toggled open) ----
    if st.session_state.readme_edit_open:
        st.markdown("---")
        st.markdown("#### ✏️ Markdown Editor")
        edited = st.text_area(
            "Edit the guide below:",
            value=st.session_state.readme_content,
            height=500,
            key="readme_textarea",
            label_visibility="collapsed",
            help="Standard Markdown supported. Changes are applied to the preview above in real-time.",
        )
        st.session_state.readme_content = edited

        btn1, btn2, btn3 = st.columns(3)
        btn1.download_button(
            "💾 Download .md",
            data=edited.encode("utf-8"),
            file_name="calibration_lab_guide.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if btn2.button("↩️ Reset to Default", use_container_width=True, key="reset_readme"):
            st.session_state.readme_content = None
            st.session_state.readme_edit_open = False
            st.rerun()
        if btn3.button("✅ Close Editor", use_container_width=True, key="readme_close_bottom"):
            st.session_state.readme_edit_open = False
            st.rerun()



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
    st.sidebar.caption("v4.0 · UX & Transparency Release")
    if st.session_state.config:
        st.sidebar.caption(f"Target: {st.session_state.config['data']['target_column']}")
    if st.session_state.selected_model_name:
        st.sidebar.caption(f"Model: {st.session_state.selected_model_name}")

    dispatch = {
        STEPS[0]: render_upload,
        STEPS[1]: render_preprocessing,
        STEPS[2]: render_alignment,
        STEPS[3]: render_eda,
        STEPS[4]: render_modelling,
        STEPS[5]: render_results,
        STEPS[6]: render_post_analysis,
        STEPS[7]: render_export,
        STEPS[8]: render_readme,
    }
    dispatch[st.session_state.current_step]()


if __name__ == "__main__":
    main()

