"""Regression evaluation metrics for calibration science.

Provides RMSE, MAE, R², MAPE, Bias, Pearson correlation,
and OLS slope/intercept for predicted-vs-reference fit quality.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate a safe MAPE value that ignores zero targets.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values.

    Returns
    -------
    float
        MAPE in percent, or NaN when all targets are zero.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = y_true_arr != 0
    if not np.any(mask):
        return float("nan")
    return float(
        np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    )


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean error (predicted − actual).  Positive = over-prediction."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def pearson_r(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Pearson correlation coefficient between actual and predicted."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) < 2 or np.std(y_t) == 0 or np.std(y_p) == 0:
        return float("nan")
    corr_matrix = np.corrcoef(y_t, y_p)
    return float(corr_matrix[0, 1])


def fit_slope_intercept(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """OLS slope and intercept of predicted vs actual (fit quality).

    Returns
    -------
    tuple[float, float]
        (slope, intercept).  Perfect calibration → (1.0, 0.0).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) < 2:
        return float("nan"), float("nan")
    coefficients = np.polyfit(y_t, y_p, 1)
    return float(coefficients[0]), float(coefficients[1])


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    cv_predictions: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Calculate the full suite of calibration-grade regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth values from the test split.
    y_pred:
        Predicted values for the test split.
    cv_predictions:
        Optional cross-validation prediction DataFrame with columns
        ``actual`` and ``predicted``.

    Returns
    -------
    Dict[str, Any]
        Dictionary of metric names to values.
    """
    slope, intercept = fit_slope_intercept(y_true, y_pred)

    metrics: Dict[str, Any] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "slope": slope,
        "intercept": intercept,
    }

    if cv_predictions is not None and not cv_predictions.empty:
        cv_slope, cv_intercept = fit_slope_intercept(
            cv_predictions["actual"], cv_predictions["predicted"]
        )
        metrics["cv_rmse"] = float(
            np.sqrt(mean_squared_error(cv_predictions["actual"], cv_predictions["predicted"]))
        )
        metrics["cv_mae"] = float(
            mean_absolute_error(cv_predictions["actual"], cv_predictions["predicted"])
        )
        metrics["cv_r2"] = float(
            r2_score(cv_predictions["actual"], cv_predictions["predicted"])
        )
        metrics["cv_bias"] = bias(cv_predictions["actual"], cv_predictions["predicted"])
        metrics["cv_pearson_r"] = pearson_r(cv_predictions["actual"], cv_predictions["predicted"])
        metrics["cv_slope"] = cv_slope
        metrics["cv_intercept"] = cv_intercept

    return metrics
