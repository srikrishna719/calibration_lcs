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
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit

from evaluation.metrics import calculate_regression_metrics
from models.model_registry import get_selected_models
from modules.diagnostics import compute_coefficient_table

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None


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
    "ols_regression": {},
    "multiple_linear_regression": {},
}


STATSMODELS_LINEAR_MODELS = {"ols_regression", "multiple_linear_regression"}


class StatsmodelsOLSRegressor:
    """Small sklearn-like wrapper around statsmodels OLS results."""

    def __init__(self) -> None:
        self.result_: Any = None
        self.feature_names_: List[str] = []
        self.design_columns_: List[str] = []

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return {}

    def set_params(self, **params: object) -> "StatsmodelsOLSRegressor":
        return self

    def _prepare_design(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        if self.feature_names_:
            X_df = X_df.reindex(columns=self.feature_names_, fill_value=0)
        design = sm.add_constant(X_df, has_constant="add")
        for column in self.design_columns_:
            if column not in design.columns:
                design[column] = 1.0 if column == "const" else 0.0
        if self.design_columns_:
            design = design[self.design_columns_]
        return design

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatsmodelsOLSRegressor":
        if sm is None:
            raise ImportError("statsmodels is required for OLS Regression.")
        X_df = pd.DataFrame(X).copy()
        self.feature_names_ = [str(column) for column in X_df.columns]
        design = sm.add_constant(X_df, has_constant="add")
        self.design_columns_ = [str(column) for column in design.columns]
        self.result_ = sm.OLS(y, design, missing="drop").fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("StatsmodelsOLSRegressor must be fitted before prediction.")
        design = self._prepare_design(pd.DataFrame(X))
        return np.asarray(self.result_.predict(design), dtype=float)

    @property
    def coef_(self) -> np.ndarray:
        if self.result_ is None:
            return np.array([])
        params = self.result_.params.drop(labels=["const"], errors="ignore")
        return np.asarray(params, dtype=float)

    @property
    def intercept_(self) -> float:
        if self.result_ is None:
            return float("nan")
        return float(self.result_.params.get("const", 0.0))


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
    standard_errors: Optional[Dict[str, float]] = None
    t_statistics: Optional[Dict[str, float]] = None
    p_values: Optional[Dict[str, float]] = None
    coefficient_table: Optional[pd.DataFrame] = None
    validation_method: str = "timeseriessplit"
    validation_predictions: Optional[pd.DataFrame] = None
    residuals: Optional[pd.Series] = None


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
    if target_column not in ordered.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the modelling dataset.")

    timestamps = ordered[timestamp_column]
    target = pd.to_numeric(ordered[target_column], errors="coerce")
    features = (
        ordered.drop(columns=[target_column, timestamp_column], errors="ignore")
        .select_dtypes(include=[np.number])
        .apply(pd.to_numeric, errors="coerce")
    )

    if feature_subset:
        valid_subset = [c for c in feature_subset if c in features.columns]
        if valid_subset:
            features = features[valid_subset]

    modelling_matrix = features.copy()
    modelling_matrix[target_column] = target
    modelling_matrix[timestamp_column] = timestamps
    modelling_matrix = modelling_matrix.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[target_column, *features.columns],
    )

    if modelling_matrix.empty or len(modelling_matrix) < 3:
        raise ValueError("At least three complete modelling rows are required for training.")

    features = modelling_matrix[features.columns]
    target = modelling_matrix[target_column]
    timestamps = modelling_matrix[timestamp_column]
    return features.reset_index(drop=True), target.reset_index(drop=True), timestamps.reset_index(drop=True)


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


def _canonical_validation_method(method: object) -> str:
    """Return a stable validation method key."""
    normalized = str(method or "timeseriessplit").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if normalized in {"kfold", "crossvalidation", "kfoldcrossvalidation"}:
        return "kfold"
    if normalized in {"holdout", "holdoutvalidation", "trainsettestsplit", "traintestsplit"}:
        return "holdout"
    if normalized in {"timeseriessplit", "timeseries", "tscv"}:
        return "timeseriessplit"
    raise ValueError("Validation method must be K-Fold, TimeSeriesSplit, or Holdout Validation.")


def _safe_folds(n_samples: int, requested_folds: int) -> int:
    """Clamp fold count to the available sample size."""
    if n_samples < 3:
        raise ValueError("At least three rows are required for cross-validation.")
    return max(2, min(int(requested_folds), n_samples - 1))


def _clone_model(model: RegressorMixin | StatsmodelsOLSRegressor) -> RegressorMixin | StatsmodelsOLSRegressor:
    """Clone sklearn or local statsmodels-wrapper estimators."""
    if isinstance(model, StatsmodelsOLSRegressor):
        return StatsmodelsOLSRegressor()
    return model.__class__(**model.get_params())


def _make_cv_splitter(method: str, n_samples: int, folds: int, random_state: int):
    """Create the requested sklearn CV splitter."""
    safe_folds = _safe_folds(n_samples, folds)
    if method == "kfold":
        return KFold(n_splits=safe_folds, shuffle=True, random_state=random_state)
    return TimeSeriesSplit(n_splits=safe_folds)


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
    validation_method: str = "timeseriessplit",
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

    cv = _make_cv_splitter(validation_method, len(x_train), cv_folds, random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
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

def generate_validation_predictions(
    model: RegressorMixin | StatsmodelsOLSRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    folds: int,
    method: str,
    random_state: int,
) -> pd.DataFrame:
    """Generate out-of-fold predictions using the requested validation strategy."""
    splitter = _make_cv_splitter(method, len(features), folds, random_state)
    predictions: List[pd.DataFrame] = []

    for train_idx, test_idx in splitter.split(features):
        fold_model = _clone_model(model)
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
    validation_method = _canonical_validation_method(config.get("validation_method", "timeseriessplit"))
    tuning_cfg: Dict[str, Any] = config.get("tuning", {})  # type: ignore[assignment]
    results: List[TrainingResult] = []
    feature_names = features.columns.tolist()

    for model_name, model in models.items():
        best_params: Dict[str, Any] = {}
        if model_name in STATSMODELS_LINEAR_MODELS:
            model = StatsmodelsOLSRegressor()

        # --- optional tuning ---
        model_tune = tuning_cfg.get(model_name, {})
        if model_tune.get("enabled", False) and model_name not in STATSMODELS_LINEAR_MODELS:
            n_iter = int(model_tune.get("n_iter", 10))
            model, best_params = tune_hyperparameters(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                n_iter=n_iter,
                cv_folds=folds,
                random_state=random_state,
                validation_method=validation_method,
            )

        # --- validation predictions and metrics ---
        if validation_method == "holdout":
            model.fit(x_train, y_train)
            test_preds = model.predict(x_test)
            validation_pred_df = pd.DataFrame({
                "timestamp": ts_test.values,
                "actual": y_test.values,
                "predicted": test_preds,
            })
            metrics = calculate_regression_metrics(
                y_true=y_test,
                y_pred=pd.Series(test_preds, index=y_test.index),
            )
        else:
            validation_pred_df = generate_validation_predictions(
                model=model,
                features=features,
                target=target,
                timestamps=timestamps,
                folds=folds,
                method=validation_method,
                random_state=random_state,
            )
            metrics = calculate_regression_metrics(
                y_true=validation_pred_df["actual"],
                y_pred=validation_pred_df["predicted"],
            )
            model.fit(features, target)

        metrics["validation_method"] = validation_method

        full_preds_df = pd.DataFrame({
            "timestamp": timestamps.values,
            "actual": target.values,
            "predicted": model.predict(features),
        })

        importance, coefs, intercept = extract_feature_importance(model, feature_names)
        coefficient_table = None
        standard_errors = None
        t_statistics = None
        p_values = None
        if isinstance(model, StatsmodelsOLSRegressor) and model.result_ is not None:
            coefficient_table = compute_coefficient_table(model.result_)
            if coefficient_table is not None and not coefficient_table.empty:
                standard_errors = dict(zip(
                    coefficient_table["Variable"].astype(str),
                    coefficient_table["Std Error"].astype(float),
                ))
                t_statistics = dict(zip(
                    coefficient_table["Variable"].astype(str),
                    coefficient_table["t-statistic"].astype(float),
                ))
                p_values = dict(zip(
                    coefficient_table["Variable"].astype(str),
                    coefficient_table["p-value"].astype(float),
                ))

        residuals = pd.Series(
            np.asarray(validation_pred_df["actual"], dtype=float)
            - np.asarray(validation_pred_df["predicted"], dtype=float)
        )

        results.append(TrainingResult(
            model_name=model_name,
            model=model,
            metrics=metrics,
            test_predictions=validation_pred_df.reset_index(drop=True),
            full_predictions=full_preds_df.reset_index(drop=True),
            feature_names=feature_names,
            feature_importance=importance,
            coefficients=coefs,
            intercept_value=intercept,
            best_params=best_params if best_params else None,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            p_values=p_values,
            coefficient_table=coefficient_table,
            validation_method=validation_method,
            validation_predictions=validation_pred_df.reset_index(drop=True),
            residuals=residuals,
        ))

    return results
