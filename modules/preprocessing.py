"""Preprocessing utilities for sensor calibration datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessingSummary:
    """Summary information captured during preprocessing."""

    original_rows: int
    cleaned_rows: int
    rows_removed_as_outliers: int
    numeric_columns: List[str]
    percentage_removed: float = 0.0
    # Rows each column would remove on its own. A row is dropped if *any*
    # screened column calls it an outlier, so the total is far larger than any
    # single entry once several columns are screened -- with independent
    # columns the survival rates multiply. Showing the breakdown is what makes
    # a surprising total explainable.
    outlier_rows_by_column: Dict[str, int] = field(default_factory=dict)
    outlier_columns: List[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        """Total rows removed by missing-value and outlier treatment."""
        return max(0, self.original_rows - self.cleaned_rows)

    @property
    def rows_remaining(self) -> int:
        """Rows remaining after preprocessing."""
        return self.cleaned_rows


_MISSING_STRATEGY_ALIASES = {
    "no treatment": "none",
    "none": "none",
    "forward fill": "forward_fill",
    "ffill": "forward_fill",
    "backward fill": "backward_fill",
    "bfill": "backward_fill",
    "linear interpolation": "linear_interpolation",
    "interpolate": "linear_interpolation",
    "linear interpolation + forward fill": "linear_interpolation_forward_fill",
    "interpolate_ffill": "linear_interpolation_forward_fill",
    "interpolate ffill": "linear_interpolation_forward_fill",
    "linear interpolation forward fill": "linear_interpolation_forward_fill",
    "linear interpolation + backward fill": "linear_interpolation_backward_fill",
    "interpolate_bfill": "linear_interpolation_backward_fill",
    "interpolate bfill": "linear_interpolation_backward_fill",
    "linear interpolation backward fill": "linear_interpolation_backward_fill",
    "drop missing rows": "drop_missing_rows",
    "drop": "drop_missing_rows",
}

_OUTLIER_METHOD_ALIASES = {
    "none": "none",
    "iqr": "iqr",
    "i.q.r.": "iqr",
    "z-score": "zscore",
    "zscore": "zscore",
    "z score": "zscore",
}


def _canonical_missing_strategy(method: str) -> str:
    key = str(method).strip().lower().replace("_", " ")
    if key not in _MISSING_STRATEGY_ALIASES:
        raise ValueError(f"Unsupported missing value strategy '{method}'.")
    return _MISSING_STRATEGY_ALIASES[key]


def _canonical_outlier_method(method: str) -> str:
    key = str(method).strip().lower().replace("_", "-")
    if key not in _OUTLIER_METHOD_ALIASES:
        key = key.replace("-", " ")
    if key not in _OUTLIER_METHOD_ALIASES:
        raise ValueError(f"Unsupported outlier method '{method}'.")
    return _OUTLIER_METHOD_ALIASES[key]


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
    method: str = "linear_interpolation_forward_fill",
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
        Missing-value strategy. Supported options are No Treatment, Forward
        Fill, Backward Fill, Linear Interpolation, Linear Interpolation +
        Forward Fill, Linear Interpolation + Backward Fill, and Drop Missing
        Rows. Stable internal aliases are also accepted.

    Returns
    -------
    pd.DataFrame
        Dataset with cleaned missing values.
    """
    transformed = dataframe.sort_values(timestamp_column).copy()
    if not numeric_columns:
        return transformed

    method_key = _canonical_missing_strategy(method)

    if method_key == "none":
        return transformed
    elif method_key == "drop_missing_rows":
        return transformed.dropna(subset=numeric_columns).reset_index(drop=True)
    elif method_key == "linear_interpolation_forward_fill":
        transformed[numeric_columns] = transformed[numeric_columns].interpolate(
            method="linear",
            limit_area="inside",
        )
        transformed[numeric_columns] = transformed[numeric_columns].ffill()
    elif method_key == "linear_interpolation":
        transformed[numeric_columns] = transformed[numeric_columns].interpolate(
            method="linear",
            limit_area="inside",
        )
    elif method_key == "linear_interpolation_backward_fill":
        transformed[numeric_columns] = transformed[numeric_columns].interpolate(
            method="linear",
            limit_area="inside",
        )
        transformed[numeric_columns] = transformed[numeric_columns].bfill()
    elif method_key == "forward_fill":
        transformed[numeric_columns] = transformed[numeric_columns].ffill()
    elif method_key == "backward_fill":
        transformed[numeric_columns] = transformed[numeric_columns].bfill()

    return transformed


def outlier_mask_by_column(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    method: str,
    threshold: float,
) -> Dict[str, pd.Series]:
    """Per-column keep-masks, so each column's contribution stays visible."""
    method_key = _canonical_outlier_method(method)
    masks: Dict[str, pd.Series] = {}
    if method_key == "none" or not numeric_columns:
        return masks

    if method_key == "zscore":
        values = dataframe[numeric_columns]
        std = values.std(ddof=0).replace(0, np.nan)
        zscores = ((values - values.mean()) / std).abs().fillna(0)
        for column in numeric_columns:
            masks[column] = zscores[column] <= threshold
        return masks

    for column in numeric_columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        masks[column] = dataframe[column].between(lower, upper) | dataframe[column].isna()
    return masks


def detect_outlier_mask_iqr(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    multiplier: float,
) -> pd.Series:
    """Create an IQR-based row mask for non-outlier observations."""
    mask = pd.Series(True, index=dataframe.index)
    for column_mask in outlier_mask_by_column(dataframe, numeric_columns, "iqr", multiplier).values():
        mask &= column_mask
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
    method_key = _canonical_outlier_method(method)
    if not numeric_columns or method_key == "none":
        return dataframe.copy()

    if method_key == "iqr":
        mask = detect_outlier_mask_iqr(dataframe, numeric_columns, threshold)
    elif method_key == "zscore":
        mask = detect_outlier_mask_zscore(dataframe, numeric_columns, threshold)

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
        Preprocessing settings. ``outlier_columns`` narrows the outlier screen
        to named columns; by default every numeric column is screened, and
        because a row is dropped when *any* of them objects, the loss grows
        with the number of columns even on clean data.
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
        method=str(config.get("missing_strategy", "linear_interpolation_forward_fill")),
    )
    rows_after_missing = len(transformed)

    requested = [str(c) for c in (config.get("outlier_columns") or [])]
    outlier_columns = [c for c in requested if c in numeric_columns] if requested else list(numeric_columns)

    method = str(config.get("outlier_method", "iqr"))
    threshold = float(config.get("outlier_threshold", 1.5))
    per_column = outlier_mask_by_column(transformed, outlier_columns, method, threshold)
    rows_by_column = {column: int((~mask).sum()) for column, mask in per_column.items()}

    transformed = remove_outliers(
        dataframe=transformed,
        numeric_columns=outlier_columns,
        method=method,
        threshold=threshold,
    )
    summary = PreprocessingSummary(
        original_rows=original_rows,
        cleaned_rows=len(transformed),
        rows_removed_as_outliers=max(0, rows_after_missing - len(transformed)),
        numeric_columns=numeric_columns,
        percentage_removed=round(
            (max(0, original_rows - len(transformed)) / original_rows * 100) if original_rows > 0 else 0.0, 2
        ),
        outlier_rows_by_column=dict(sorted(rows_by_column.items(), key=lambda kv: -kv[1])),
        outlier_columns=outlier_columns,
    )
    return transformed, summary
