"""Reusable Streamlit download helpers for tables and Plotly charts."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    go = None


def _csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a dataframe as UTF-8 CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def _plotly_png_bytes(fig: "go.Figure") -> tuple[Optional[bytes], Optional[str]]:
    """Render a Plotly figure to PNG bytes, returning an error tooltip on failure."""
    try:
        return fig.to_image(format="png", width=1200, height=700, scale=2), None
    except Exception as exc:  # pragma: no cover - depends on kaleido/runtime support
        return (
            None,
            "PNG export is unavailable. Plotly/Kaleido may need a compatible browser runtime. "
            f"Details: {exc}",
        )


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
            data=_csv_bytes(df),
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
    """Render compact PNG and source-data CSV downloads for a Plotly chart."""
    cols = st.columns([1, 1, 4], gap="small") if source_df is not None else st.columns([1, 5], gap="small")

    png_bytes, png_error = _plotly_png_bytes(fig)
    with cols[0]:
        st.download_button(
            label="📥 PNG",
            data=png_bytes or b"",
            file_name=f"{filename_prefix}.png",
            mime="image/png",
            key=f"{key}_png",
            help=png_error or "Download this chart as a PNG image",
            disabled=png_bytes is None,
            width='stretch',
        )

    if source_df is not None:
        with cols[1]:
            st.download_button(
                label="📊 Data",
                data=_csv_bytes(source_df),
                file_name=f"{filename_prefix}_source_data.csv",
                mime="text/csv",
                key=f"{key}_source_csv",
                help="Download the exact source data used to build this chart",
                width='stretch',
            )

    if png_error:
        st.caption(png_error)
