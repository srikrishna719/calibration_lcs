"""Dataset normalization utilities."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


SCALER_MAP = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}

_METHOD_ALIASES = {
    "standard": "StandardScaler",
    "standardscaler": "StandardScaler",
    "minmax": "MinMaxScaler",
    "minmaxscaler": "MinMaxScaler",
    "robust": "RobustScaler",
    "robustscaler": "RobustScaler",
}

_NONE_METHODS = {"", "none", "no", "null", "false"}


def _canonical_method(method: Optional[str]) -> str:
    """Normalize user/config method names to supported scaler keys."""
    if method is None:
        return "none"

    normalized = str(method).strip()
    lookup = normalized.lower().replace("_", "").replace("-", "").replace(" ", "")
    if lookup in _NONE_METHODS:
        return "none"
    if lookup in _METHOD_ALIASES:
        return _METHOD_ALIASES[lookup]

    supported = ["none", *SCALER_MAP.keys()]
    raise ValueError(f"Unsupported normalization method '{method}'. Choose from: {supported}")


def build_scaler(method: Optional[str]) -> object | None:
    """Return an unfitted scaler for ``method``, or ``None`` for no scaling.

    The scaler emits pandas output so feature names survive the transform. That
    matters when it is composed into a modelling pipeline: downstream
    estimators (notably the statsmodels OLS wrapper) rely on column names.

    Use this when scaling belongs inside a model pipeline, so the scaler is fit
    on training folds only and travels with the exported model. Use
    :func:`normalize_dataset` for standalone/preview scaling of a whole frame.
    """
    method_name = _canonical_method(method)
    if method_name == "none":
        return None

    scaler = SCALER_MAP[method_name]()
    scaler.set_output(transform="pandas")
    return scaler


def _valid_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return requested columns that exist and are numeric."""
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns.tolist())
    return [str(column) for column in columns if str(column) in numeric_cols]


def normalize_dataset(
    df: pd.DataFrame,
    columns: Iterable[str],
    method: Optional[str],
    *,
    return_scaler: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, object | None]:
    """Normalize selected numeric columns with a supported sklearn scaler.

    This fits on the whole frame, so it is for standalone transforms and UI
    previews only. Do **not** use it to scale a modelling dataset before
    training: that leaks test-fold statistics into the fit and leaves the
    scaler behind, so the exported model cannot be applied to raw data. Compose
    :func:`build_scaler` into the model pipeline instead.

    By default this returns only the normalized dataframe. Set
    ``return_scaler=True`` when a caller also needs the fitted scaler object.
    """
    result = df.copy()
    method_name = _canonical_method(method)
    valid_columns = _valid_numeric_columns(result, columns)

    if method_name == "none" or not valid_columns:
        return (result, None) if return_scaler else result

    scaler = SCALER_MAP[method_name]()
    result[valid_columns] = scaler.fit_transform(result[valid_columns])
    return (result, scaler) if return_scaler else result


def _rounded_stat(series: pd.Series, stat_name: str) -> float:
    """Return a rounded numeric summary statistic or NaN for empty data."""
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return float("nan")

    if stat_name == "mean":
        value = cleaned.mean()
    elif stat_name == "std":
        value = cleaned.std()
    elif stat_name == "min":
        value = cleaned.min()
    elif stat_name == "max":
        value = cleaned.max()
    else:  # pragma: no cover - internal guard
        raise ValueError(f"Unknown statistic '{stat_name}'.")

    return round(float(value), 6)


def get_normalization_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Return before/after summary statistics for normalized columns."""
    rows: list[dict[str, float | str]] = []
    for column in columns:
        col = str(column)
        if col not in df_before.columns or col not in df_after.columns:
            continue

        rows.append({
            "column": col,
            "before_mean": _rounded_stat(df_before[col], "mean"),
            "before_std": _rounded_stat(df_before[col], "std"),
            "before_min": _rounded_stat(df_before[col], "min"),
            "before_max": _rounded_stat(df_before[col], "max"),
            "after_mean": _rounded_stat(df_after[col], "mean"),
            "after_std": _rounded_stat(df_after[col], "std"),
            "after_min": _rounded_stat(df_after[col], "min"),
            "after_max": _rounded_stat(df_after[col], "max"),
        })

    return pd.DataFrame(rows)
