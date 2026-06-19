"""Model training routines with feature-importance extraction.

Handles time-aware splitting, TimeSeriesSplit cross-validation,
optional RandomizedSearchCV hyperparameter tuning, feature subset
filtering, multi-model training, and explainability capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from evaluation.metrics import calculate_regression_metrics
from models.model_registry import get_selected_models


# ---------------------------------------------------------------------------
# Default hyperparameter search spaces
# ---------------------------------------------------------------------------

_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "random_forest": {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 5, 8, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    },
    "ridge": {
        "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    },
    "lasso": {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    },
    "linear_regression": {},
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Container for model training outputs."""

    model_name: str
    model: RegressorMixin
    metrics: Dict[str, float]
    test_predictions: pd.DataFrame
    full_predictions: pd.DataFrame
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None
    coefficients: Optional[Dict[str, float]] = None
    intercept_value: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None   # populated when tuning is used


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def extract_feature_importance(
    model: RegressorMixin,
    feature_names: List[str],
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[float]]:
    """Extract feature importance or coefficients from a fitted model."""
    importance: Optional[Dict[str, float]] = None
    coefs: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None

    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        importance = {
            name: float(val)
            for name, val in sorted(
                zip(feature_names, raw), key=lambda x: abs(x[1]), reverse=True
            )
        }

    if hasattr(model, "coef_"):
        raw = np.asarray(model.coef_).ravel()
        coefs = {
            name: float(val)
            for name, val in sorted(
                zip(feature_names, raw), key=lambda x: abs(x[1]), reverse=True
            )
        }
    if hasattr(model, "intercept_"):
        # Use np.asarray().ravel() to safely handle all numpy types:
        # - Python float / numpy scalar → ravel gives 1-element array
        # - 0-dimensional ndarray (numpy 2.x no longer supports float() on these)
        # - 1-dimensional ndarray (e.g. Ridge/Lasso with multi-output)
        intercept_arr = np.asarray(model.intercept_).ravel()
        if intercept_arr.size > 0:
            intercept = float(intercept_arr[0])

    return importance, coefs, intercept


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_matrices(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    feature_subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a modeling dataset into features, target, and timestamps.

    Parameters
    ----------
    feature_subset:
        If provided, only these columns are used as features.
        Must all be present in dataframe (after dropping target/timestamp).
    """
    ordered = dataframe.sort_values(timestamp_column).reset_index(drop=True)
    target = ordered[target_column]
    timestamps = ordered[timestamp_column]
    features = ordered.drop(columns=[target_column, timestamp_column], errors="ignore")

    if feature_subset:
        valid_subset = [c for c in feature_subset if c in features.columns]
        if valid_subset:
            features = features[valid_subset]

    return features, target, timestamps


def split_train_test_by_time(
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a leakage-safe chronological train/test split."""
    split_index = max(1, int(len(features) * (1 - test_size)))
    split_index = min(split_index, len(features) - 1)
    return (
        features.iloc[:split_index],
        features.iloc[split_index:],
        target.iloc[:split_index],
        target.iloc[split_index:],
        timestamps.iloc[:split_index],
        timestamps.iloc[split_index:],
    )


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    model_name: str,
    model: RegressorMixin,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    cv_folds: int,
    random_state: int,
) -> Tuple[RegressorMixin, Dict[str, Any]]:
    """Run RandomizedSearchCV with TimeSeriesSplit for a given model.

    Parameters
    ----------
    model_name:
        Registry key used to look up the param grid.
    model:
        Unfitted or pre-configured sklearn regressor.
    x_train, y_train:
        Training data.
    n_iter:
        Number of random combinations to try.
    cv_folds:
        Number of TimeSeriesSplit folds.
    random_state:
        For reproducibility.

    Returns
    -------
    Tuple[best_model, best_params_dict]
    """
    param_grid = _PARAM_GRIDS.get(model_name, {})
    if not param_grid:
        # No grid defined (e.g. LinearRegression) — return as-is
        model.fit(x_train, y_train)
        return model, {}

    tscv = TimeSeriesSplit(n_splits=max(2, cv_folds))
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_)


# ---------------------------------------------------------------------------
# CV predictions
# ---------------------------------------------------------------------------

def generate_time_series_cv_predictions(
    model: RegressorMixin,
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    folds: int,
) -> pd.DataFrame:
    """Generate out-of-fold predictions using time-aware cross-validation."""
    tscv = TimeSeriesSplit(n_splits=max(2, folds))
    predictions: List[pd.DataFrame] = []

    for train_idx, test_idx in tscv.split(features):
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(features.iloc[train_idx], target.iloc[train_idx])
        fold_preds = fold_model.predict(features.iloc[test_idx])
        predictions.append(pd.DataFrame({
            "timestamp": timestamps.iloc[test_idx].values,
            "actual": target.iloc[test_idx].values,
            "predicted": fold_preds,
        }))

    if not predictions:
        return pd.DataFrame(columns=["timestamp", "actual", "predicted"])

    return pd.concat(predictions, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_models(
    dataframe: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    config: Dict[str, object],
    random_state: int = 42,
    feature_subset: Optional[List[str]] = None,
) -> List[TrainingResult]:
    """Train all configured models and collect evaluation outputs.

    Parameters
    ----------
    dataframe:
        Feature-engineered dataset.
    target_column:
        Name of the target variable.
    timestamp_column:
        Timestamp column name.
    config:
        Training configuration dict.
    random_state:
        Random seed for reproducibility.
    feature_subset:
        Optional list of feature column names to use (overrides all features).

    Returns
    -------
    List[TrainingResult]
    """
    features, target, timestamps = prepare_training_matrices(
        dataframe=dataframe,
        target_column=target_column,
        timestamp_column=timestamp_column,
        feature_subset=feature_subset,
    )
    x_train, x_test, y_train, y_test, ts_train, ts_test = split_train_test_by_time(
        features=features,
        target=target,
        timestamps=timestamps,
        test_size=float(config.get("test_size", 0.2)),
    )

    models = get_selected_models(config=config, random_state=random_state)
    folds = int(config.get("cross_validation_folds", 5))
    tuning_cfg: Dict[str, Any] = config.get("tuning", {})  # type: ignore[assignment]
    results: List[TrainingResult] = []
    feature_names = features.columns.tolist()

    for model_name, model in models.items():
        best_params: Dict[str, Any] = {}

        # --- optional tuning ---
        model_tune = tuning_cfg.get(model_name, {})
        if model_tune.get("enabled", False):
            n_iter = int(model_tune.get("n_iter", 10))
            model, best_params = tune_hyperparameters(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                n_iter=n_iter,
                cv_folds=folds,
                random_state=random_state,
            )
        else:
            model.fit(x_train, y_train)

        # --- test predictions ---
        test_preds = model.predict(x_test)
        test_pred_df = pd.DataFrame({
            "timestamp": ts_test.values,
            "actual": y_test.values,
            "predicted": test_preds,
        })

        # --- CV predictions (on train set) ---
        cv_pred_df = generate_time_series_cv_predictions(
            model=model,
            features=x_train.reset_index(drop=True),
            target=y_train.reset_index(drop=True),
            timestamps=ts_train.reset_index(drop=True),
            folds=folds,
        )

        metrics = calculate_regression_metrics(
            y_true=y_test,
            y_pred=pd.Series(test_preds, index=y_test.index),
            cv_predictions=cv_pred_df,
        )

        full_preds_df = pd.DataFrame({
            "timestamp": timestamps.values,
            "actual": target.values,
            "predicted": model.predict(features),
        })

        importance, coefs, intercept = extract_feature_importance(model, feature_names)

        results.append(TrainingResult(
            model_name=model_name,
            model=model,
            metrics=metrics,
            test_predictions=test_pred_df.reset_index(drop=True),
            full_predictions=full_preds_df.reset_index(drop=True),
            feature_names=feature_names,
            feature_importance=importance,
            coefficients=coefs,
            intercept_value=intercept,
            best_params=best_params if best_params else None,
        ))

    return results
