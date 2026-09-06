"""Reusable Streamlit download helpers for tables and Plotly charts.

Download payloads are produced lazily. ``st.download_button`` accepts a
callable for ``data`` and runs it only when the user actually clicks, off the
script thread. That matters here because rendering a Plotly figure to PNG via
Kaleido costs seconds, and a page carrying six charts used to pay that on every
rerun -- every checkbox, selectbox and radio -- for images nobody had asked for.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    import plotly.io as pio  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    go = None
    pio = None

PNG_WIDTH = 1200
PNG_HEIGHT = 700
PNG_SCALE = 2


@st.cache_data(show_spinner=False, max_entries=32)
def _png_from_figure_json(figure_json: str, width: int, height: int, scale: int) -> bytes:
    """Render a serialized Plotly figure to PNG.

    Keyed on the figure JSON rather than the figure object, which is not
    hashable, so clicking the same chart's download twice is free.
    """
    if pio is None:  # pragma: no cover - plotly is a hard dependency in practice
        raise RuntimeError("Plotly is required to export charts as PNG.")
    return pio.from_json(figure_json).to_image(
        format="png", width=width, height=height, scale=scale
    )


def _csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a dataframe as UTF-8 CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def _deferred_csv(dataframe: pd.DataFrame) -> Callable[[], bytes]:
    """Defer CSV serialization until the download is clicked."""

    def _generate() -> bytes:
        return _csv_bytes(dataframe)

    return _generate


def _deferred_png(fig: "go.Figure") -> Callable[[], bytes]:
    """Defer Kaleido rendering until the download is clicked.

    Serializing the figure happens inside the callable too, so a rerun that
    nobody downloads from does no figure work at all.
    """

    def _generate() -> bytes:
        try:
            return _png_from_figure_json(fig.to_json(), PNG_WIDTH, PNG_HEIGHT, PNG_SCALE)
        except Exception as exc:  # pragma: no cover - depends on the Kaleido runtime
            raise RuntimeError(
                "PNG export failed. Plotly/Kaleido may need a compatible browser "
                f"runtime on this machine. Details: {exc}"
            ) from exc

    return _generate


def render_df_download(
    df: pd.DataFrame,
    key: str,
    filename: str,
) -> None:
    """Render a compact CSV download button for a displayed dataframe."""
    cols = st.columns([1, 5], gap="small")
    with cols[0]:
        st.download_button(
            label="📥 CSV",
            data=_deferred_csv(df),
            file_name=filename,
            mime="text/csv",
            key=key,
            help=f"Download this table as {filename}",
            width='stretch',
        )


def render_chart_download(
    fig: "go.Figure",
    source_df: Optional[pd.DataFrame],
    key: str,
    filename_prefix: str,
) -> None:
    """Render compact PNG and source-data downloads for a Plotly chart."""
    cols = st.columns([1, 1, 4], gap="small") if source_df is not None else st.columns([1, 5], gap="small")

    with cols[0]:
        st.download_button(
            label="📥 PNG",
            data=_deferred_png(fig),
            file_name=f"{filename_prefix}.png",
            mime="image/png",
            key=f"{key}_png",
            help="Download this chart as a PNG image (rendered when you click)",
            width='stretch',
        )

    if source_df is not None:
        with cols[1]:
            st.download_button(
                label="📊 Data",
                data=_deferred_csv(source_df),
                file_name=f"{filename_prefix}_source_data.csv",
                mime="text/csv",
                key=f"{key}_source_csv",
                help="Download the exact source data used to build this chart",
                width='stretch',
            )
