"""Exploratory data analysis helpers with rich visualizations.

Provides distribution plots, correlation heatmaps, missing-value analysis,
anomaly detection, and time-series visualizations for the merged dataset.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Missing value analysis
# ---------------------------------------------------------------------------

def summarize_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column."""
    summary = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": dataframe.isna().sum().values,
        }
    )
    summary["missing_pct"] = (summary["missing_count"] / max(len(dataframe), 1)) * 100
    return summary.sort_values("missing_count", ascending=False).reset_index(drop=True)


def create_missing_value_heatmap(dataframe: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing the position of missing values across columns."""
    missing_matrix = dataframe.isna().astype(int)
    fig = go.Figure(
        data=go.Heatmap(
            z=missing_matrix.values.T,
            x=list(range(len(dataframe))),
            y=list(missing_matrix.columns),
            colorscale=[[0, "#1e1b4b"], [1, "#ef4444"]],
            showscale=True,
            colorbar=dict(title="Missing", tickvals=[0, 1], ticktext=["Present", "Missing"]),
        )
    )
    fig.update_layout(
        title="Missing Value Heatmap",
        xaxis_title="Row Index",
        yaxis_title="Column",
        template="plotly_dark",
        height=max(300, len(missing_matrix.columns) * 28),
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_basic_anomalies(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag rows with large absolute z-scores."""
    if not numeric_columns:
        return dataframe.iloc[0:0].copy()

    values = dataframe[numeric_columns]
    std = values.std(ddof=0).replace(0, np.nan)
    zscores = ((values - values.mean()) / std).abs().fillna(0)
    anomaly_mask = (zscores > threshold).any(axis=1)
    anomalies = dataframe.loc[anomaly_mask].copy()
    anomalies["max_abs_zscore"] = zscores.loc[anomaly_mask].max(axis=1)
    return anomalies.sort_values("max_abs_zscore", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def create_distribution_figure(
    dataframe: pd.DataFrame,
    column: str,
    title_suffix: str = "",
) -> go.Figure:
    """Create a histogram for a selected column."""
    title = f"Distribution: {column}"
    if title_suffix:
        title += f" ({title_suffix})"
    fig = px.histogram(
        dataframe,
        x=column,
        nbins=30,
        marginal="box",
        title=title,
        color_discrete_sequence=["#6366f1"],
    )
    fig.update_layout(template="plotly_dark")
    return fig


def create_before_after_distributions(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    column: str,
) -> go.Figure:
    """Overlay histograms showing a column before and after cleaning."""
    fig = go.Figure()
    if column in before_df.columns:
        fig.add_trace(
            go.Histogram(
                x=before_df[column].dropna(),
                name="Before Cleaning",
                opacity=0.5,
                marker_color="#ef4444",
                nbinsx=30,
            )
        )
    if column in after_df.columns:
        fig.add_trace(
            go.Histogram(
                x=after_df[column].dropna(),
                name="After Cleaning",
                opacity=0.6,
                marker_color="#10b981",
                nbinsx=30,
            )
        )
    fig.update_layout(
        title=f"Before vs After Cleaning — {column}",
        barmode="overlay",
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def create_correlation_heatmap(dataframe: pd.DataFrame) -> go.Figure:
    """Create a correlation heatmap for numeric columns."""
    numeric_df = dataframe.select_dtypes(include="number")
    correlation = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )
    fig.update_layout(template="plotly_dark")
    return fig


# ---------------------------------------------------------------------------
# Time-series visualization
# ---------------------------------------------------------------------------

def create_time_series_figure(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    columns: Optional[List[str]] = None,
) -> go.Figure:
    """Create a time-series overlay of numeric columns."""
    if columns is None:
        columns = dataframe.select_dtypes(include="number").columns.tolist()[:6]

    fig = go.Figure()
    palette = px.colors.qualitative.Vivid
    for idx, col in enumerate(columns):
        if col in dataframe.columns:
            fig.add_trace(
                go.Scatter(
                    x=dataframe[timestamp_column],
                    y=dataframe[col],
                    mode="lines",
                    name=col,
                    line=dict(color=palette[idx % len(palette)], width=1.5),
                )
            )

    fig.update_layout(
        title="Time-Series Overview",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly figure
# ---------------------------------------------------------------------------

def create_anomaly_figure(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    anomalies: pd.DataFrame,
) -> go.Figure:
    """Create a time-series anomaly plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataframe[timestamp_column],
            y=dataframe[value_column],
            mode="lines+markers",
            name="Normal",
            marker=dict(color="#6366f1", size=4),
            line=dict(color="#6366f1", width=1),
        )
    )
    if not anomalies.empty and timestamp_column in anomalies.columns and value_column in anomalies.columns:
        fig.add_trace(
            go.Scatter(
                x=anomalies[timestamp_column],
                y=anomalies[value_column],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#ef4444", size=8, symbol="x"),
            )
        )
    fig.update_layout(
        title=f"Anomaly Detection — {value_column}",
        xaxis_title="Time",
        yaxis_title=value_column,
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_eda_outputs(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    raw_dataframe: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Generate all EDA tables and figures for the merged dataset.

    Parameters
    ----------
    dataframe:
        Cleaned/merged dataset.
    timestamp_column:
        Name of the timestamp column.
    raw_dataframe:
        Optional raw (pre-cleaning) dataset for before/after comparisons.

    Returns
    -------
    Dict[str, object]
        Collection of DataFrames and Plotly figures.
    """
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        raise ValueError("EDA requires at least one numeric column.")

    distribution_column = numeric_columns[0]
    missing_summary = summarize_missing_values(dataframe)
    anomalies = detect_basic_anomalies(dataframe, numeric_columns)

    outputs: Dict[str, object] = {
        "missing_summary": missing_summary,
        "anomalies": anomalies,
        "distribution_column": distribution_column,
        "numeric_columns": numeric_columns,
        "distribution_figure": create_distribution_figure(dataframe, distribution_column),
        "correlation_figure": create_correlation_heatmap(dataframe),
        "missing_heatmap": create_missing_value_heatmap(dataframe),
        "time_series_figure": create_time_series_figure(dataframe, timestamp_column),
        "anomaly_figure": create_anomaly_figure(
            dataframe=dataframe,
            timestamp_column=timestamp_column,
            value_column=distribution_column,
            anomalies=anomalies,
        ),
    }

    if raw_dataframe is not None and distribution_column in raw_dataframe.columns:
        outputs["before_after_figure"] = create_before_after_distributions(
            raw_dataframe, dataframe, distribution_column
        )

    return outputs
