"""Model training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from evaluation.metrics import calculate_regression_metrics
from models.model_registry import get_selected_models


@dataclass
class TrainingResult:
    """Container for model training outputs."""

    model_name: str
    model: RegressorMixin
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a training dataset into features, target, and timestamps.

    Parameters
    ----------
    dataframe:
        Feature-engineered dataset.
    target_column:
        Name of the regression target.
    timestamp_column:
        Timestamp field preserved for traceability.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series]
        Features, target, and timestamps.
    """
    features = dataframe.drop(columns=[target_column])
    if timestamp_column in features.columns:
        timestamps = features.pop(timestamp_column)
    else:
        timestamps = pd.Series(index=dataframe.index, dtype="datetime64[ns]")
    target = dataframe[target_column]
    return features, target, timestamps


def train_models(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    config: Dict[str, object],
    random_state: int = 42,
) -> List[TrainingResult]:
    """Train all configured models and collect comparable evaluation outputs.

    Parameters
    ----------
    dataframe:
        Feature-engineered training dataset.
    target_column:
        Regression target field.
    timestamp_column:
        Timestamp field.
    config:
        Training settings.
    random_state:
        Seed for deterministic splits.

    Returns
    -------
    List[TrainingResult]
        Per-model training results.
    """
    features, target, timestamps = split_features_target(
        dataframe=dataframe,
        target_column=target_column,
        timestamp_column=timestamp_column,
    )

    x_train, x_test, y_train, y_test, ts_train, ts_test = train_test_split(
        features,
        target,
        timestamps,
        test_size=float(config.get("test_size", 0.2)),
        random_state=random_state,
    )

    folds = int(config.get("cross_validation_folds", 5))
    kfold = KFold(n_splits=max(2, folds), shuffle=True, random_state=random_state)
    models = get_selected_models(config=config, random_state=random_state)
    results: List[TrainingResult] = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        test_predictions = model.predict(x_test)
        cv_predictions = cross_val_predict(model, features, target, cv=kfold)
        metrics = calculate_regression_metrics(
            y_true=y_test,
            y_pred=test_predictions,
            y_true_full=target,
            y_pred_full=cv_predictions,
        )
        predictions = pd.DataFrame(
            {
                "timestamp": ts_test.reset_index(drop=True),
                "actual": y_test.reset_index(drop=True),
                "predicted": pd.Series(test_predictions),
            }
        ).sort_values("timestamp")

        results.append(
            TrainingResult(
                model_name=model_name,
                model=model,
                metrics=metrics,
                predictions=predictions.reset_index(drop=True),
            )
        )

    return results
