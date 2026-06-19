"""Preprocessing utilities for sensor calibration datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class PreprocessingSummary:
    """Summary information captured during preprocessing."""

    original_rows: int
    cleaned_rows: int
    rows_removed_as_outliers: int
    numeric_columns: List[str]


def get_numeric_columns(
    dataframe: pd.DataFrame,
    exclude: Iterable[str] | None = None,
) -> List[str]:
    """Return numeric columns excluding any provided fields."""
    excluded = set(exclude or [])
    return [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if column not in excluded
    ]


def clean_missing_values(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    numeric_columns: List[str],
    method: str = "interpolate_ffill",
) -> pd.DataFrame:
    """Clean missing numeric values using the chosen strategy.

    Parameters
    ----------
    dataframe:
        Dataset to transform.
    timestamp_column:
        Timestamp field used to preserve ordering.
    numeric_columns:
        Numeric columns to clean.
    method:
        Missing-value strategy. One of:
        - ``"interpolate_ffill"`` — linear interpolation, then ffill+bfill for any
          remaining gaps (edges). Most robust; **recommended**.
        - ``"interpolate"``      — linear interpolation only (no fill fallback).
          May leave NaNs at the very start/end of the series.
        - ``"ffill"``            — forward-fill (carry last known value forward),
          then bfill for leading NaNs.
        - ``"bfill"``            — backward-fill (use the next known value),
          then ffill for trailing NaNs.

    Returns
    -------
    pd.DataFrame
        Dataset with cleaned missing values.
    """
    transformed = dataframe.sort_values(timestamp_column).copy()
    if not numeric_columns:
        return transformed

    if method == "interpolate_ffill":
        transformed[numeric_columns] = transformed[numeric_columns].interpolate(
            method="linear",
            limit_direction="both",
        )
        transformed[numeric_columns] = transformed[numeric_columns].ffill().bfill()
    elif method == "interpolate":
        transformed[numeric_columns] = transformed[numeric_columns].interpolate(
            method="linear",
            limit_direction="both",
        )
    elif method == "ffill":
        transformed[numeric_columns] = transformed[numeric_columns].ffill().bfill()
    elif method == "bfill":
        transformed[numeric_columns] = transformed[numeric_columns].bfill().ffill()
    else:
        raise ValueError(f"Unsupported missing value strategy '{method}'.")

    return transformed


def detect_outlier_mask_iqr(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    multiplier: float,
) -> pd.Series:
    """Create an IQR-based row mask for non-outlier observations."""
    mask = pd.Series(True, index=dataframe.index)
    for column in numeric_columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask &= dataframe[column].between(lower, upper) | dataframe[column].isna()
    return mask


def detect_outlier_mask_zscore(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    threshold: float,
) -> pd.Series:
    """Create a z-score-based row mask for non-outlier observations."""
    if not numeric_columns:
        return pd.Series(True, index=dataframe.index)

    values = dataframe[numeric_columns]
    std = values.std(ddof=0).replace(0, np.nan)
    zscores = ((values - values.mean()) / std).abs()
    zscores = zscores.fillna(0)
    return (zscores <= threshold).all(axis=1)


def remove_outliers(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Remove outlier rows from numeric columns."""
    if not numeric_columns or method == "none":
        return dataframe.copy()

    if method == "iqr":
        mask = detect_outlier_mask_iqr(dataframe, numeric_columns, threshold)
    elif method == "zscore":
        mask = detect_outlier_mask_zscore(dataframe, numeric_columns, threshold)
    else:
        raise ValueError(f"Unsupported outlier method '{method}'.")

    return dataframe.loc[mask].reset_index(drop=True)


def preprocess_dataset(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    config: Dict[str, object],
    exclude_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, PreprocessingSummary]:
    """Run preprocessing for a single dataset.

    Parameters
    ----------
    dataframe:
        Dataset to preprocess.
    timestamp_column:
        Timestamp field.
    config:
        Preprocessing settings.
    exclude_columns:
        Columns excluded from numeric preprocessing.

    Returns
    -------
    tuple[pd.DataFrame, PreprocessingSummary]
        Cleaned dataset and summary metadata.
    """
    original_rows = len(dataframe)
    numeric_columns = get_numeric_columns(
        dataframe=dataframe,
        exclude=[timestamp_column, *(exclude_columns or [])],
    )
    transformed = clean_missing_values(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        numeric_columns=numeric_columns,
        method=str(config.get("missing_strategy", "interpolate_ffill")),
    )
    transformed = remove_outliers(
        dataframe=transformed,
        numeric_columns=numeric_columns,
        method=str(config.get("outlier_method", "iqr")),
        threshold=float(config.get("outlier_threshold", 1.5)),
    )
    summary = PreprocessingSummary(
        original_rows=original_rows,
        cleaned_rows=len(transformed),
        rows_removed_as_outliers=max(0, original_rows - len(transformed)),
        numeric_columns=numeric_columns,
    )
    return transformed, summary
