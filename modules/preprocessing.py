"""Preprocessing utilities for sensor calibration datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PreprocessingArtifacts:
    """Artifacts captured during preprocessing."""

    imputer: SimpleImputer
    scaler: StandardScaler | MinMaxScaler | None
    numeric_columns: List[str]


def get_numeric_columns(dataframe: pd.DataFrame, exclude: Iterable[str] | None = None) -> List[str]:
    """Return numeric columns excluding any provided fields.

    Parameters
    ----------
    dataframe:
        Input dataset.
    exclude:
        Column names to omit.

    Returns
    -------
    List[str]
        Numeric column names.
    """
    excluded = set(exclude or [])
    return [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if column not in excluded
    ]


def impute_missing_values(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    strategy: str = "median",
) -> Tuple[pd.DataFrame, SimpleImputer]:
    """Fill missing numeric values using a configurable strategy.

    Parameters
    ----------
    dataframe:
        Dataset to transform.
    numeric_columns:
        Numeric columns to impute.
    strategy:
        Scikit-learn imputation strategy.

    Returns
    -------
    Tuple[pd.DataFrame, SimpleImputer]
        Imputed dataset and fitted imputer.
    """
    transformed = dataframe.copy()
    imputer = SimpleImputer(strategy=strategy)

    if numeric_columns:
        transformed[numeric_columns] = imputer.fit_transform(transformed[numeric_columns])

    return transformed, imputer


def remove_outliers_iqr(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Remove rows containing IQR-based outliers in numeric columns.

    Parameters
    ----------
    dataframe:
        Dataset to filter.
    numeric_columns:
        Numeric columns to inspect.
    multiplier:
        IQR multiplier that defines the acceptable bounds.

    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    if not numeric_columns:
        return dataframe.copy()

    filtered = dataframe.copy()
    mask = pd.Series(True, index=filtered.index)

    for column in numeric_columns:
        q1 = filtered[column].quantile(0.25)
        q3 = filtered[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (multiplier * iqr)
        upper = q3 + (multiplier * iqr)
        mask &= filtered[column].between(lower, upper) | filtered[column].isna()

    return filtered.loc[mask].reset_index(drop=True)


def normalize_features(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, StandardScaler | MinMaxScaler | None]:
    """Normalize numeric columns with a selected scaler.

    Parameters
    ----------
    dataframe:
        Dataset to scale.
    numeric_columns:
        Numeric columns to normalize.
    method:
        Either `standard`, `minmax`, or `none`.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler | MinMaxScaler | None]
        Scaled dataset and fitted scaler when applicable.
    """
    transformed = dataframe.copy()
    scaler: StandardScaler | MinMaxScaler | None

    if not numeric_columns or method == "none":
        return transformed, None

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported normalization method '{method}'.")

    transformed[numeric_columns] = scaler.fit_transform(transformed[numeric_columns])
    return transformed, scaler


def preprocess_dataset(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    config: Dict[str, object],
    exclude_columns: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, PreprocessingArtifacts]:
    """Run the configured preprocessing pipeline on a dataset.

    Parameters
    ----------
    dataframe:
        Dataset to preprocess.
    timestamp_column:
        Timestamp field excluded from numeric transformations.
    config:
        Preprocessing settings.
    exclude_columns:
        Columns excluded from numeric preprocessing, such as the training target.

    Returns
    -------
    Tuple[pd.DataFrame, PreprocessingArtifacts]
        Preprocessed dataset and transformation artifacts.
    """
    excluded_columns = [timestamp_column, *(exclude_columns or [])]
    numeric_columns = get_numeric_columns(dataframe, exclude=excluded_columns)
    transformed, imputer = impute_missing_values(
        dataframe=dataframe,
        numeric_columns=numeric_columns,
        strategy=str(config.get("missing_strategy", "median")),
    )

    outlier_method = str(config.get("outlier_method", "iqr"))
    if outlier_method == "iqr":
        transformed = remove_outliers_iqr(
            dataframe=transformed,
            numeric_columns=numeric_columns,
            multiplier=float(config.get("outlier_multiplier", 1.5)),
        )
    elif outlier_method != "none":
        raise ValueError(f"Unsupported outlier method '{outlier_method}'.")

    transformed, scaler = normalize_features(
        dataframe=transformed,
        numeric_columns=numeric_columns,
        method=str(config.get("normalization", "standard")),
    )

    artifacts = PreprocessingArtifacts(
        imputer=imputer,
        scaler=scaler,
        numeric_columns=numeric_columns,
    )
    return transformed, artifacts
