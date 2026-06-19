"""Registry of supported calibration models."""

from __future__ import annotations

from typing import Dict

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


def build_model_registry(random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Build the default registry of supported models."""
    registry: Dict[str, RegressorMixin] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(random_state=random_state, max_iter=10000),
        "random_forest": RandomForestRegressor(random_state=random_state),
    }
    if XGBRegressor is not None:
        registry["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
        )
    return registry


def get_selected_models(config: Dict[str, object], random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Return configured models with parameter overrides applied."""
    registry = build_model_registry(random_state=random_state)
    selected_names = [str(name) for name in config.get("selected_models", registry.keys())]
    model_params = config.get("model_params", {})
    selected_models: Dict[str, RegressorMixin] = {}

    for name in selected_names:
        if name not in registry:
            continue
        model = registry[name]
        params = model_params.get(name, {})
        if params:
            # Coerce string-encoded numbers back to their native types
            # (can happen after JSON round-trip with default=str serializer)
            coerced: Dict[str, object] = {}
            for k, v in params.items():
                if isinstance(v, str):
                    try:
                        coerced[k] = int(v)
                        continue
                    except ValueError:
                        pass
                    try:
                        coerced[k] = float(v)
                        continue
                    except ValueError:
                        pass
                coerced[k] = v
            model.set_params(**coerced)
        selected_models[name] = model

    if not selected_models:
        raise ValueError("No valid models were selected for training.")
    return selected_models
