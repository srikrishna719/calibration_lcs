"""Feature engineering utilities for calibration models.

Creates lag features, rolling statistics (mean and std), polynomial and
interaction terms, and rich time-derived features from the merged dataset.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

def select_feature_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    config: Dict[str, object],
) -> List[str]:
    """Select numeric feature columns, including optional meteorological variables."""
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col != target_column]

    optional_columns = [str(col) for col in config.get("optional_columns", [])]
    if optional_columns:
        feature_columns = [
            col for col in feature_columns
            if col in optional_columns or "sensor_" in col
        ]

    return feature_columns


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def create_lag_features(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    lag_steps: List[int],
) -> pd.DataFrame:
    """Create lagged copies of selected features."""
    engineered = dataframe.copy()
    for column in feature_columns:
        for lag in lag_steps:
            engineered[f"{column}_lag_{lag}"] = engineered[column].shift(lag)
    return engineered


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def create_rolling_features(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    windows: List[int],
    include_std: bool = True,
) -> pd.DataFrame:
    """Create rolling mean (and optionally std) features."""
    engineered = dataframe.copy()
    for column in feature_columns:
        for window in windows:
            rolling = engineered[column].rolling(window=window, min_periods=1)
            engineered[f"{column}_rolling_mean_{window}"] = rolling.mean()
            if include_std:
                engineered[f"{column}_rolling_std_{window}"] = rolling.std().fillna(0)
    return engineered


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------

def create_polynomial_features(
    dataframe: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
) -> pd.DataFrame:
    """Add polynomial terms (degree 2 or 3) for a user-selected subset of columns.

    Uses sklearn PolynomialFeatures internally, but only appends the *new*
    columns (powers and cross-products) — the originals are kept as-is.

    Parameters
    ----------
    dataframe:
        Input dataset.
    columns:
        Base columns to expand (should be sensor columns, not lags/rolling).
    degree:
        Polynomial degree (2 or 3).

    Returns
    -------
    pd.DataFrame
        Dataset with polynomial feature columns appended.
    """
    if not columns or degree <= 1:
        return dataframe.copy()

    from sklearn.preprocessing import PolynomialFeatures

    valid_cols = [c for c in columns if c in dataframe.columns]
    if not valid_cols:
        return dataframe.copy()

    engineered = dataframe.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    transformed = poly.fit_transform(engineered[valid_cols])
    feat_names = poly.get_feature_names_out(valid_cols)

    for name, values in zip(feat_names, transformed.T):
        if name not in engineered.columns:   # skip duplicates of originals
            engineered[name] = values

    return engineered


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------

def create_interaction_terms(
    dataframe: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Create pairwise interaction terms (col_i × col_j) for selected columns.

    Parameters
    ----------
    dataframe:
        Input dataset.
    columns:
        Columns to cross-multiply.

    Returns
    -------
    pd.DataFrame
        Dataset with interaction columns appended.
    """
    engineered = dataframe.copy()
    valid_cols = [c for c in columns if c in engineered.columns]
    for i, c1 in enumerate(valid_cols):
        for c2 in valid_cols[i + 1:]:
            col_name = f"{c1}_x_{c2}"
            if col_name not in engineered.columns:
                engineered[col_name] = engineered[c1] * engineered[c2]
    return engineered


# ---------------------------------------------------------------------------
# Time features (basic + extended)
# ---------------------------------------------------------------------------

def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


_SEASON_NUM = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
_DOW_ABBR = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def create_time_features(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    config: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Create time-derived features from the timestamp column.

    Parameters
    ----------
    dataframe:
        Input dataset.
    timestamp_column:
        Name of the datetime column.
    config:
        Dict of boolean flags controlling which features to create.
        Keys (all default True for backward-compat unless noted):
        - ``hour_of_day``        : 0-23 integer
        - ``day_of_week``        : 0=Mon … 6=Sun integer
        - ``day_of_month``       : 1-31 integer
        - ``unix_timestamp``     : seconds since Unix epoch  (default False)
        - ``julian_date``        : day-of-year 1-366         (default False)
        - ``calendar_date``      : days since 1970-01-01     (default False)
        - ``cyclical_hour``      : sin/cos of hour           (default False)
        - ``cyclical_dow``       : sin/cos of day-of-week    (default False)
        - ``cyclical_doy``       : sin/cos of day-of-year    (default False)
        - ``season``             : DJF/MAM/JJA/SON numeric   (default False)
        - ``day_name``           : Mon-Sun label + is_weekend (default False)

    Returns
    -------
    pd.DataFrame
        Dataset with added time features.
    """
    cfg = config or {}
    engineered = dataframe.copy()
    ts = pd.to_datetime(engineered[timestamp_column])

    # --- basic (always on unless explicitly disabled) ---
    if cfg.get("hour_of_day", True):
        engineered["hour_of_day"] = ts.dt.hour
    if cfg.get("day_of_week", True):
        engineered["day_of_week"] = ts.dt.dayofweek
    if cfg.get("day_of_month", True):
        engineered["day_of_month"] = ts.dt.day

    # --- unix timestamp ---
    if cfg.get("unix_timestamp", False):
        engineered["unix_timestamp"] = ts.astype(np.int64) // 10 ** 9

    # --- julian date (day of year) ---
    if cfg.get("julian_date", False):
        engineered["julian_date"] = ts.dt.dayofyear

    # --- calendar date (integer days since epoch) ---
    if cfg.get("calendar_date", False):
        engineered["calendar_date"] = (
            ts.dt.normalize() - pd.Timestamp("1970-01-01")
        ).dt.days.astype(float)

    # --- cyclical encodings ---
    if cfg.get("cyclical_hour", False):
        engineered["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        engineered["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    if cfg.get("cyclical_dow", False):
        engineered["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        engineered["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    if cfg.get("cyclical_doy", False):
        doy = ts.dt.dayofyear
        engineered["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        engineered["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # --- season ---
    if cfg.get("season", False):
        season_str = ts.dt.month.map(_season_from_month)
        engineered["season_num"] = season_str.map(_SEASON_NUM).astype(float)

    # --- day name / weekend ---
    if cfg.get("day_name", False):
        dow = ts.dt.dayofweek
        engineered["day_name_num"] = dow.map(_DOW_ABBR).map(
            {v: k for k, v in _DOW_ABBR.items()}
        ).astype(float)
        engineered["is_weekend"] = (dow >= 5).astype(int)

    return engineered


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def engineer_features(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    config: Dict[str, object],
) -> pd.DataFrame:
    """Apply configured feature engineering steps.

    Parameters
    ----------
    dataframe:
        Merged dataset.
    timestamp_column:
        Timestamp column name.
    target_column:
        Target variable column name.
    config:
        Feature-engineering configuration dict.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataset ready for modelling.
    """
    if not bool(config.get("enabled", True)):
        return dataframe.copy()

    feature_columns = select_feature_columns(
        dataframe=dataframe,
        target_column=target_column,
        config=config,
    )

    engineered = create_lag_features(
        dataframe=dataframe,
        feature_columns=feature_columns,
        lag_steps=[int(s) for s in config.get("lag_steps", [1, 2, 3])],
    )
    engineered = create_rolling_features(
        dataframe=engineered,
        feature_columns=feature_columns,
        windows=[int(w) for w in config.get("rolling_windows", [3, 6])],
        include_std=bool(config.get("rolling_std", True)),
    )

    # Polynomial features (user-chosen subset)
    poly_cols = [str(c) for c in config.get("polynomial_columns", [])]
    poly_degree = int(config.get("polynomial_degree", 1))
    if poly_cols and poly_degree > 1:
        engineered = create_polynomial_features(engineered, poly_cols, poly_degree)

    # Interaction terms (user-chosen subset)
    interaction_cols = [str(c) for c in config.get("interaction_columns", [])]
    if interaction_cols:
        engineered = create_interaction_terms(engineered, interaction_cols)

    # Time features
    if bool(config.get("add_time_features", True)):
        time_cfg = config.get("time_feature_flags", {})
        engineered = create_time_features(engineered, timestamp_column, time_cfg)

    return engineered.dropna().reset_index(drop=True)
