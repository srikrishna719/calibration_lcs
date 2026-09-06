"""Theme palettes and the CSS block for the Streamlit UI.

Extracted from ui/app.py: several hundred lines of styling with no logic,
which made the module holding the workflow much harder to read.
"""

from __future__ import annotations

from string import Template
from typing import Dict


# ---------------------------------------------------------------------------
# Premium CSS — theme-parameterized (dark / light)
# ---------------------------------------------------------------------------
_PALETTES: Dict[str, Dict[str, str]] = {
    "dark": {
        "grad1": "#1e1b4b", "grad2": "#312e81", "grad3": "#4338ca",
        "sidebar1": "#0f0a2e", "sidebar2": "#1e1b4b",
        "header_text": "#e0e7ff", "subtext": "#a5b4fc",
        "value_text": "#e0e7ff", "sidebar_label": "#c7d2fe",
        "section_bg": "rgba(30, 27, 75, 0.4)",
        "pill_bg": "rgba(99, 102, 241, 0.15)",
        "page_bg": "#0f0a2e", "page_text": "#e0e7ff",
        "step_dot_idle": "#1e1b4b",
        "widget_bg": "#171433", "widget_border": "#4338ca",
        "alert_border": "rgba(226, 232, 240, 0.15)",
    },
    "light": {
        "grad1": "#e0e7ff", "grad2": "#c7d2fe", "grad3": "#818cf8",
        "sidebar1": "#f8fafc", "sidebar2": "#eef2ff",
        "header_text": "#1e1b4b", "subtext": "#4338ca",
        "value_text": "#1e1b4b", "sidebar_label": "#312e81",
        "section_bg": "rgba(99, 102, 241, 0.06)",
        "pill_bg": "rgba(99, 102, 241, 0.10)",
        "page_bg": "#f8fafc", "page_text": "#1e1b4b",
        "step_dot_idle": "#c7d2fe",
        "widget_bg": "#ffffff", "widget_border": "#c7d2fe",
        "alert_border": "rgba(30, 27, 75, 0.10)",
    },
}

_CUSTOM_CSS_TEMPLATE = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Streamlit draws its icons as Material Symbols ligatures, and its icon spans
   carry st- classes, so the rule above was overriding the icon font and the
   ligature fell back to its own name as literal text ("keyboard_double_arrow_left"
   on the sidebar toggle, "upload" on the file uploader). The font is self-hosted
   by Streamlit under this exact family name. */
span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    letter-spacing: normal;
    text-transform: none;
    white-space: nowrap;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

/* Page-wide chrome (keeps native Streamlit containers in sync with the toggle,
   overriding Streamlit's own auto-detected OS/browser theme so the in-app
   toggle is authoritative regardless of the visitor's environment) */
[data-testid="stAppViewContainer"] {
    background-color: $page_bg;
    color: $page_text;
}
[data-testid="stHeader"] {
    background-color: transparent;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5 {
    color: $page_text;
}
label, [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p {
    color: $page_text !important;
}
[data-testid="stExpander"] {
    background-color: $section_bg;
    border-radius: 10px;
}
[data-testid="stExpander"] summary p {
    color: $page_text !important;
}

/* Native input/control chrome — Streamlit bakes these to its own detected
   OS/browser theme, not this app's custom CSS, so they need explicit
   overrides to actually follow the Dark/Light toggle. */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="stFileUploader"] button {
    background-color: $widget_bg !important;
    color: $page_text !important;
    border: 1px solid $widget_border !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: $widget_bg !important;
    border: 1px dashed $widget_border !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: $page_text !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color: $widget_bg !important;
    color: $page_text !important;
    border-color: $widget_border !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stMultiSelect"] div[data-baseweb="select"] span {
    color: $page_text !important;
}
[data-testid="stTabs"] button p {
    color: $subtext !important;
}
[data-testid="stTabs"] button[aria-selected="true"] p {
    color: $page_text !important;
}
[data-testid="stDataFrame"] {
    background-color: $widget_bg;
    border: 1px solid $widget_border;
}
[data-testid="stAlert"] {
    border: 1px solid $alert_border;
}

/* Tooltips and dropdown menus (BaseWeb) are rendered in a portal appended
   near <body>, outside the themed app container, so they need their own
   explicit overrides — they don't inherit anything from the rules above. */
[data-baseweb="tooltip"] {
    background-color: transparent !important;
    color: $page_text !important;
}
[data-baseweb="tooltip"] > div {
    background-color: $widget_bg !important;
    border: 1px solid $widget_border !important;
}
[data-baseweb="tooltip"] * {
    color: $page_text !important;
}
[data-baseweb="popover"] {
    background-color: $widget_bg !important;
    border: 1px solid $widget_border !important;
    color: $page_text !important;
}
[data-baseweb="popover"] * {
    color: $page_text !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: $section_bg !important;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, $grad1 0%, $grad2 50%, $grad3 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(67, 56, 202, 0.3);
}
.main-header h1 {
    color: $header_text;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: $subtext;
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
    background: $step_dot_idle;
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
    background: linear-gradient(135deg, $grad1, $grad2);
    border: 1px solid $grad3;
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
    color: $subtext;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    color: $value_text;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.25rem;
}
.metric-card .value.good { color: #34d399; }
.metric-card .value.warn { color: #fbbf24; }
.metric-card .value.bad  { color: #f87171; }

/* Section card */
.section-card {
    background: $section_bg;
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* Sidebar refinements */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, $sidebar1 0%, $sidebar2 100%);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    font-weight: 500;
    padding: 0.4rem 0;
}

/* Buttons */
.stButton button {
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}
.stButton button:hover {
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
    background: $pill_bg;
    color: $subtext;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-bottom: 0.3rem;
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
    color: $sidebar_label !important;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
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
.stDownloadButton button {
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    min-height: 2rem;
}
</style>
""")


def build_custom_css(theme: str) -> str:
    """Render the premium CSS block for the given theme ('dark' or 'light')."""
    tokens = _PALETTES.get(theme, _PALETTES["light"])
    return _CUSTOM_CSS_TEMPLATE.safe_substitute(tokens)
