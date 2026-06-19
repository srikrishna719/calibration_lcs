"""Enhanced plotting utilities for calibration science.

Provides scatter plots with OLS fit and 1:1 reference lines,
Bland-Altman agreement plots, multi-model comparison figures,
and grouped metric bar charts.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_fit(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, x_range, y_fit) for OLS y~x, or None."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return None
    coefs = np.polyfit(x[mask], y[mask], 1)
    x_range = np.linspace(x[mask].min(), x[mask].max(), 200)
    return float(coefs[0]), float(coefs[1]), x_range, np.polyval(coefs, x_range)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# ---------------------------------------------------------------------------
# Single-model scatter with 1:1 + OLS fit
# ---------------------------------------------------------------------------

def create_scatter_with_fit(
    predictions_df: pd.DataFrame,
    title: str = "Predicted vs Actual",
    model_name: str = "",
) -> go.Figure:
    """Scatter plot with 1:1 ideal line, OLS regression fit line, and Pearson r."""
    actual = predictions_df["actual"].values.astype(float)
    predicted = predictions_df["predicted"].values.astype(float)

    r = _pearson(actual, predicted)
    fit_result = _ols_fit(actual, predicted)

    fig = go.Figure()

    # Scatter coloured by predicted value
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers",
        name="Observations",
        marker=dict(
            color=predicted,
            colorscale="Viridis",
            size=6,
            opacity=0.75,
            showscale=True,
            colorbar=dict(title="Predicted", thickness=12),
        ),
    ))

    # 1:1 perfect-agreement line
    lo = float(min(np.nanmin(actual), np.nanmin(predicted)))
    hi = float(max(np.nanmax(actual), np.nanmax(predicted)))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="1:1 Line (ideal)",
        line=dict(color="#f59e0b", dash="dash", width=2),
    ))

    # OLS regression fit
    if fit_result is not None:
        slope, intercept, xr, yf = fit_result
        fig.add_trace(go.Scatter(
            x=xr, y=yf,
            mode="lines",
            name=f"OLS Fit  slope={slope:.3f}  int={intercept:.3f}",
            line=dict(color="#10b981", width=2),
        ))

    # Pearson r annotation
    r_txt = f"r = {r:.4f}" if not np.isnan(r) else "r = N/A"
    fig.add_annotation(
        x=0.04, y=0.96, xref="paper", yref="paper",
        text=f"<b>{r_txt}</b>",
        showarrow=False,
        font=dict(color="#a5b4fc", size=13),
        bgcolor="rgba(30,27,75,0.8)",
        bordercolor="#6366f1", borderwidth=1, borderpad=5,
    )

    label = f" — {model_name}" if model_name else ""
    fig.update_layout(
        title=f"{title}{label}",
        xaxis_title="Actual (Reference)",
        yaxis_title="Predicted (Calibrated)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Bland-Altman agreement plot
# ---------------------------------------------------------------------------

def create_bland_altman_plot(
    predictions_df: pd.DataFrame,
    model_name: str = "",
) -> go.Figure:
    """Bland-Altman (Tukey mean-difference) agreement plot."""
    actual = predictions_df["actual"].values.astype(float)
    predicted = predictions_df["predicted"].values.astype(float)

    mean_vals = (actual + predicted) / 2.0
    diff_vals = predicted - actual
    mean_diff = float(np.nanmean(diff_vals))
    std_diff = float(np.nanstd(diff_vals, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_vals, y=diff_vals,
        mode="markers",
        name="Observations",
        marker=dict(color="#8b5cf6", size=5, opacity=0.7),
    ))
    fig.add_hline(
        y=mean_diff, line_color="#f59e0b", line_width=2,
        annotation_text=f"Mean bias: {mean_diff:.3f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=loa_upper, line_color="#ef4444", line_dash="dash", line_width=1.5,
        annotation_text=f"+1.96σ: {loa_upper:.3f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=loa_lower, line_color="#ef4444", line_dash="dash", line_width=1.5,
        annotation_text=f"−1.96σ: {loa_lower:.3f}",
        annotation_position="bottom right",
    )

    label = f" — {model_name}" if model_name else ""
    fig.update_layout(
        title=f"Bland-Altman Agreement Plot{label}",
        xaxis_title="Mean of Actual & Predicted",
        yaxis_title="Difference (Predicted − Actual)",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-model scatter grid
# ---------------------------------------------------------------------------

def create_multi_model_scatter(
    training_results: Dict,
    max_cols: int = 3,
) -> go.Figure:
    """Subplot grid: scatter + 1:1 + OLS fit for every trained model."""
    model_names = list(training_results.keys())
    n = len(model_names)
    if n == 0:
        return go.Figure()

    cols = min(n, max_cols)
    rows = int(np.ceil(n / cols))
    palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=model_names,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    for idx, name in enumerate(model_names):
        result = training_results[name]
        pdf = result.full_predictions
        actual = pdf["actual"].values.astype(float)
        predicted = pdf["predicted"].values.astype(float)
        color = palette[idx % len(palette)]
        r = _pearson(actual, predicted)
        fit_result = _ols_fit(actual, predicted)
        row, col = idx // cols + 1, idx % cols + 1

        fig.add_trace(go.Scatter(
            x=actual, y=predicted, mode="markers",
            marker=dict(color=color, size=4, opacity=0.6),
            showlegend=False,
        ), row=row, col=col)

        lo = float(min(np.nanmin(actual), np.nanmin(predicted)))
        hi = float(max(np.nanmax(actual), np.nanmax(predicted)))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#f59e0b", dash="dash", width=1.5),
            showlegend=False,
        ), row=row, col=col)

        if fit_result is not None:
            _, _, xr, yf = fit_result
            fig.add_trace(go.Scatter(
                x=xr, y=yf, mode="lines",
                line=dict(color="#10b981", width=1.5),
                showlegend=False,
            ), row=row, col=col)

        r_str = f"r={r:.3f}" if not np.isnan(r) else "r=N/A"
        axis_id = "" if idx == 0 else str(idx + 1)
        fig.add_annotation(
            x=0.05, y=0.92,
            xref=f"x{axis_id} domain", yref=f"y{axis_id} domain",
            text=f"<b>{r_str}</b>", showarrow=False,
            font=dict(color="#a5b4fc", size=10),
        )

    fig.update_layout(
        title="Multi-Model Scatter: Predicted vs Actual (with 1:1 & OLS lines)",
        template="plotly_dark",
        height=320 * rows,
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-model time-series overlay
# ---------------------------------------------------------------------------

def create_multi_model_timeseries(training_results: Dict) -> go.Figure:
    """Overlay time-series of all model predictions against the reference."""
    if not training_results:
        return go.Figure()

    first = next(iter(training_results.values()))
    base = first.full_predictions.sort_values("timestamp")
    palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["timestamp"], y=base["actual"],
        mode="lines", name="Reference (Actual)",
        line=dict(color="#ffffff", width=2.5),
    ))

    for idx, (name, result) in enumerate(training_results.items()):
        df = result.full_predictions.sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted"],
            mode="lines", name=name,
            line=dict(color=palette[idx % len(palette)], width=1.8),
        ))

    fig.update_layout(
        title="Multi-Model Time-Series Overlay",
        xaxis_title="Time", yaxis_title="Concentration",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-model metrics bar chart
# ---------------------------------------------------------------------------

def create_multi_model_metrics_bar(leaderboard: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing RMSE, MAE, R², Pearson r across models."""
    if leaderboard.empty:
        return go.Figure()

    metrics = [c for c in ["rmse", "mae", "r2", "pearson_r"] if c in leaderboard.columns]
    palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=[m.upper() for m in metrics],
        shared_yaxes=False,
    )

    for i, metric in enumerate(metrics):
        for j, row_data in leaderboard.iterrows():
            fig.add_trace(go.Bar(
                name=row_data["model_name"],
                x=[row_data["model_name"]],
                y=[row_data[metric]],
                marker_color=palette[int(j) % len(palette)],
                showlegend=(i == 0),
                legendgroup=row_data["model_name"],
            ), row=1, col=i + 1)

    fig.update_layout(
        title="Multi-Model Metric Comparison",
        template="plotly_dark",
        barmode="group",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig
