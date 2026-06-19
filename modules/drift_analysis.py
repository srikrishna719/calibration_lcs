"""Drift analysis and rolling-error diagnostics for post-calibration QA.

Provides rolling error computation, drift detection, and Plotly figures
for diagnosing model performance degradation over time.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def compute_rolling_errors(
    predictions_df: pd.DataFrame,
    window: int = 6,
) -> pd.DataFrame:
    """Compute rolling RMSE, MAE, and Bias over a prediction DataFrame.

    Parameters
    ----------
    predictions_df:
        Must contain ``timestamp``, ``actual``, ``predicted`` columns.
    window:
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``rolling_rmse``, ``rolling_mae``,
        ``rolling_bias``, and ``residual``.
    """
    df = predictions_df.sort_values("timestamp").copy()
    df["residual"] = df["predicted"] - df["actual"]
    df["sq_error"] = df["residual"] ** 2
    df["abs_error"] = df["residual"].abs()

    df["rolling_rmse"] = (
        df["sq_error"].rolling(window=window, min_periods=1).mean().apply(np.sqrt)
    )
    df["rolling_mae"] = df["abs_error"].rolling(window=window, min_periods=1).mean()
    df["rolling_bias"] = df["residual"].rolling(window=window, min_periods=1).mean()

    df = df.drop(columns=["sq_error", "abs_error"])
    return df


def detect_drift_periods(
    rolling_df: pd.DataFrame,
    metric_column: str = "rolling_rmse",
    threshold_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Flag rows where rolling error exceeds a dynamic threshold.

    The threshold is ``median(metric) * threshold_multiplier``.

    Parameters
    ----------
    rolling_df:
        Output of ``compute_rolling_errors``.
    metric_column:
        Which rolling metric to test.
    threshold_multiplier:
        Multiplicative factor above the median.

    Returns
    -------
    pd.DataFrame
        Rows flagged as drift with an added ``drift_threshold`` column.
    """
    median_val = rolling_df[metric_column].median()
    threshold = median_val * threshold_multiplier
    flagged = rolling_df[rolling_df[metric_column] > threshold].copy()
    flagged["drift_threshold"] = threshold
    return flagged


def create_rolling_error_figure(
    rolling_df: pd.DataFrame,
    drift_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Create a Plotly figure showing rolling RMSE and MAE over time.

    Parameters
    ----------
    rolling_df:
        Output from ``compute_rolling_errors``.
    drift_df:
        Optional drift-flagged rows to highlight.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_df["timestamp"],
            y=rolling_df["rolling_rmse"],
            mode="lines",
            name="Rolling RMSE",
            line=dict(color="#6366f1", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_df["timestamp"],
            y=rolling_df["rolling_mae"],
            mode="lines",
            name="Rolling MAE",
            line=dict(color="#f59e0b", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_df["timestamp"],
            y=rolling_df["rolling_bias"],
            mode="lines",
            name="Rolling Bias",
            line=dict(color="#10b981", width=2, dash="dot"),
        )
    )

    if drift_df is not None and not drift_df.empty:
        fig.add_trace(
            go.Scatter(
                x=drift_df["timestamp"],
                y=drift_df["rolling_rmse"],
                mode="markers",
                name="Drift Detected",
                marker=dict(color="#ef4444", size=8, symbol="x"),
            )
        )

    fig.update_layout(
        title="Rolling Error Analysis — Drift Detection",
        xaxis_title="Time",
        yaxis_title="Error",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def create_residual_histogram(predictions_df: pd.DataFrame) -> go.Figure:
    """Create a histogram of prediction residuals."""
    residuals = predictions_df["predicted"] - predictions_df["actual"]
    fig = px.histogram(
        x=residuals,
        nbins=30,
        marginal="box",
        title="Residual Distribution",
        labels={"x": "Residual (Predicted − Actual)", "count": "Frequency"},
        color_discrete_sequence=["#8b5cf6"],
    )
    fig.update_layout(template="plotly_dark")
    return fig


def create_predicted_vs_actual_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of predicted vs actual with a 1:1 reference line."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions_df["actual"],
            y=predictions_df["predicted"],
            mode="markers",
            name="Predictions",
            marker=dict(color="#6366f1", size=5, opacity=0.7),
        )
    )
    all_vals = pd.concat([predictions_df["actual"], predictions_df["predicted"]])
    lo, hi = all_vals.min(), all_vals.max()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="1:1 Line",
            line=dict(color="#f59e0b", dash="dash", width=2),
        )
    )
    fig.update_layout(
        title="Predicted vs Actual",
        xaxis_title="Actual (Reference)",
        yaxis_title="Predicted (Calibrated)",
        template="plotly_dark",
    )
    return fig


def create_residual_vs_predicted_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Residual vs predicted scatter."""
    residuals = predictions_df["predicted"] - predictions_df["actual"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions_df["predicted"],
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(color="#10b981", size=5, opacity=0.7),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#f59e0b", line_width=2)
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Predicted − Actual)",
        template="plotly_dark",
    )
    return fig


def create_time_series_overlay_figure(predictions_df: pd.DataFrame) -> go.Figure:
    """Time-series overlay of actual vs predicted."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions_df["timestamp"],
            y=predictions_df["actual"],
            mode="lines",
            name="Reference (Actual)",
            line=dict(color="#6366f1", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions_df["timestamp"],
            y=predictions_df["predicted"],
            mode="lines",
            name="Calibrated (Predicted)",
            line=dict(color="#f59e0b", width=2),
        )
    )
    fig.update_layout(
        title="Time-Series Comparison — Reference vs Calibrated",
        xaxis_title="Time",
        yaxis_title="Concentration",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def generate_post_analysis_outputs(
    predictions_df: pd.DataFrame,
    rolling_window: int = 6,
    drift_threshold: float = 1.5,
) -> Dict[str, object]:
    """Generate all post-calibration analysis outputs.

    Parameters
    ----------
    predictions_df:
        DataFrame with ``timestamp``, ``actual``, ``predicted``.
    rolling_window:
        Window for rolling error computation.
    drift_threshold:
        Multiplier for drift detection.

    Returns
    -------
    Dict[str, object]
        Figures and data tables for the UI.
    """
    rolling_df = compute_rolling_errors(predictions_df, window=rolling_window)
    drift_df = detect_drift_periods(
        rolling_df, metric_column="rolling_rmse", threshold_multiplier=drift_threshold
    )
    return {
        "rolling_errors": rolling_df,
        "drift_periods": drift_df,
        "predicted_vs_actual_fig": create_predicted_vs_actual_figure(predictions_df),
        "residual_plot_fig": create_residual_vs_predicted_figure(predictions_df),
        "time_series_overlay_fig": create_time_series_overlay_figure(predictions_df),
        "rolling_error_fig": create_rolling_error_figure(rolling_df, drift_df),
        "residual_histogram_fig": create_residual_histogram(predictions_df),
    }
