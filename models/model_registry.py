"""Registry of supported calibration models."""

from __future__ import annotations

from typing import Dict

from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


def build_model_registry(random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Build the default registry of available regression models.

    Parameters
    ----------
    random_state:
        Seed applied to stochastic estimators.

    Returns
    -------
    Dict[str, RegressorMixin]
        Mapping from model names to estimator instances.
    """
    registry: Dict[str, RegressorMixin] = {
        "linear_regression": LinearRegression(),
        "multilinear_regression": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(random_state=random_state),
        "elastic_net": ElasticNet(random_state=random_state),
        "random_forest": RandomForestRegressor(random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "extra_trees": ExtraTreesRegressor(random_state=random_state),
    }

    if XGBRegressor is not None:
        registry["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
        )

    return registry


def get_selected_models(config: Dict[str, object], random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Return the configured subset of models with parameter overrides applied.

    Parameters
    ----------
    config:
        Training configuration.
    random_state:
        Seed applied to stochastic estimators.

    Returns
    -------
    Dict[str, RegressorMixin]
        Requested model instances.
    """
    registry = build_model_registry(random_state=random_state)
    selected_names = [str(name) for name in config.get("selected_models", list(registry.keys()))]
    model_params = config.get("model_params", {})
    selected_models: Dict[str, RegressorMixin] = {}

    for name in selected_names:
        if name not in registry:
            continue
        model = registry[name]
        params = model_params.get(name, {})
        if params:
            model.set_params(**params)
        selected_models[name] = model

    if not selected_models:
        raise ValueError("No valid models were selected for training.")

    return selected_models
