"""Regression evaluation metrics."""

from __future__ import annotations

from typing import Dict

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
        Mean absolute percentage error in percent.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    non_zero_mask = y_true_array != 0
    if not np.any(non_zero_mask):
        return float("nan")
    ratio = np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask])
    return float(np.mean(ratio) * 100)


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_true_full: pd.Series | None = None,
    y_pred_full: pd.Series | None = None,
) -> Dict[str, float]:
    """Calculate core regression metrics for model comparison.

    Parameters
    ----------
    y_true:
        Evaluation targets.
    y_pred:
        Evaluation predictions.
    y_true_full:
        Optional full dataset targets for cross-validated metrics.
    y_pred_full:
        Optional full dataset predictions for cross-validated metrics.

    Returns
    -------
    Dict[str, float]
        Metric dictionary.
    """
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }

    if y_true_full is not None and y_pred_full is not None:
        metrics["cv_rmse"] = float(np.sqrt(mean_squared_error(y_true_full, y_pred_full)))
        metrics["cv_mae"] = float(mean_absolute_error(y_true_full, y_pred_full))

    return metrics
