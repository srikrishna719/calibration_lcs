"""Registry of supported calibration models.

Organises models into Statistical and Machine Learning groups.
Supports OLS Regression (statsmodels), Multiple Linear Regression,
Ridge, Lasso, Random Forest, and XGBoost.
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


# Model grouping for UI display
MODEL_GROUPS: Dict[str, List[str]] = {
    "Statistical Models": [
        "ols_regression",
        "multiple_linear_regression",
        "ridge",
        "lasso",
    ],
    "Machine Learning Models": [
        "random_forest",
        "xgboost",
    ],
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "ols_regression": "OLS Regression",
    "multiple_linear_regression": "Multiple Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Model metadata for UI display
MODEL_INFO: Dict[str, Dict[str, str]] = {
    "ols_regression": {
        "name": "OLS Regression",
        "description": "Ordinary Least Squares via statsmodels — provides p-values, standard errors, t-statistics, and full regression diagnostics.",
        "strengths": "Full statistical inference (p-values, confidence intervals), coefficient interpretability, assumption testing",
        "limitations": "Assumes linearity, no regularisation, sensitive to multicollinearity",
    },
    "multiple_linear_regression": {
        "name": "Multiple Linear Regression",
        "description": "Standard linear regression via scikit-learn. Fits a linear relationship by minimising sum of squared errors.",
        "strengths": "Fast, interpretable, good baseline. No hyperparameters to tune.",
        "limitations": "No regularisation, sensitive to correlated features, no p-values without statsmodels",
    },
    "ridge": {
        "name": "Ridge Regression",
        "description": "L2-regularised linear regression. Shrinks coefficients towards zero without eliminating any.",
        "strengths": "Handles multicollinearity, stable with many correlated features (lag/rolling)",
        "limitations": "All features retained (no sparsity), requires alpha tuning",
    },
    "lasso": {
        "name": "Lasso Regression",
        "description": "L1-regularised linear regression. Can drive coefficients to exactly zero for automatic feature selection.",
        "strengths": "Built-in feature selection, sparse models, interpretable",
        "limitations": "May arbitrarily choose between correlated features, requires alpha tuning",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of independent decision trees, each trained on bootstrap samples with random feature subsets.",
        "strengths": "Handles non-linearity, no scaling needed, robust to outliers, feature importance",
        "limitations": "Less interpretable than linear models, can overfit on small datasets",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient-boosted decision trees with built-in L1+L2 regularisation. Trees built sequentially to correct errors.",
        "strengths": "Often best accuracy on tabular data, handles missing values, built-in regularisation",
        "limitations": "Requires careful tuning, less interpretable, needs xgboost package installed",
    },
}


def build_model_registry(random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Build the default registry of supported models.

    Note: OLS regression is handled separately via statsmodels in train.py.
    This registry contains sklearn-compatible models only.
    """
    registry: Dict[str, RegressorMixin] = {
        "ols_regression": LinearRegression(),  # placeholder — actual OLS via statsmodels in train.py
        "multiple_linear_regression": LinearRegression(),
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
