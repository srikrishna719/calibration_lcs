"""Statistical diagnostics for calibration models."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:  # pragma: no cover
    sm = None
    variance_inflation_factor = None


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute variance inflation factor for each numeric predictor."""
    if sm is None or variance_inflation_factor is None:
        raise ImportError("statsmodels is required for VIF computation.")

    numeric_X = (
        X.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0)
    )
    columns = numeric_X.columns.tolist()

    if not columns:
        return pd.DataFrame(columns=["Variable", "VIF"])
    if len(columns) == 1 or numeric_X.empty:
        return pd.DataFrame({"Variable": columns, "VIF": [1.0] * len(columns)})

    X_with_const = sm.add_constant(numeric_X, has_constant="add")
    rows: list[dict[str, float | str]] = []
    for index, column in enumerate(X_with_const.columns):
        if column == "const":
            continue
        try:
            vif_value = variance_inflation_factor(X_with_const.to_numpy(dtype=float), index)
            vif = round(float(vif_value), 4)
        except Exception:
            vif = float("nan")
        rows.append({"Variable": str(column), "VIF": vif})

    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def _result_names(ols_result: Any, feature_names: Optional[Iterable[str]] = None) -> list[str]:
    """Infer coefficient names from statsmodels results or fallback feature names."""
    params = getattr(ols_result, "params", [])
    param_count = len(params)

    if hasattr(params, "index"):
        return [str(name) for name in params.index.tolist()]

    model = getattr(ols_result, "model", None)
    exog_names = getattr(model, "exog_names", None)
    if exog_names:
        return [str(name) for name in exog_names]

    names = list(feature_names or [])
    if len(names) == param_count - 1:
        names = ["const", *names]
    if len(names) != param_count:
        names = [f"x{i}" for i in range(param_count)]
    return [str(name) for name in names]


def _as_named_series(values: Any, names: list[str]) -> pd.Series:
    """Convert statsmodels arrays/Series into a named pandas Series."""
    if isinstance(values, pd.Series):
        return values
    arr = np.asarray(values, dtype=float).ravel()
    return pd.Series(arr, index=names[: len(arr)])


def compute_coefficient_table(
    ols_result: Any,
    feature_names: Optional[Iterable[str]] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
) -> Optional[pd.DataFrame]:
    """Build a coefficient table from a statsmodels OLS result.

    Optional sklearn fallback arguments are kept so existing callers can pass a
    fitted sklearn linear model plus training data and still receive a table.
    """
    if all(hasattr(ols_result, attr) for attr in ("params", "bse", "tvalues", "pvalues")):
        names = _result_names(ols_result, feature_names)
        params = _as_named_series(ols_result.params, names)
        bse = _as_named_series(ols_result.bse, names)
        tvalues = _as_named_series(ols_result.tvalues, names)
        pvalues = _as_named_series(ols_result.pvalues, names)

        rows = []
        for name in params.index:
            rows.append({
                "Variable": str(name),
                "Coefficient": round(float(params.loc[name]), 6),
                "Std Error": round(float(bse.loc[name]), 6),
                "t-statistic": round(float(tvalues.loc[name]), 6),
                "p-value": round(float(pvalues.loc[name]), 6),
            })
        return pd.DataFrame(rows)

    if hasattr(ols_result, "coef_"):
        names = [str(name) for name in (feature_names or [])]
        coefs = np.asarray(ols_result.coef_, dtype=float).ravel()

        if sm is not None and X is not None and y is not None:
            try:
                model_X = X[names] if names else X
                model_X = sm.add_constant(model_X, has_constant="add")
                fitted = sm.OLS(y, model_X, missing="drop").fit()
                return compute_coefficient_table(fitted)
            except Exception:
                pass

        rows = []
        if hasattr(ols_result, "intercept_"):
            intercept = np.asarray(ols_result.intercept_, dtype=float).ravel()
            if intercept.size:
                rows.append({
                    "Variable": "const",
                    "Coefficient": round(float(intercept[0]), 6),
                    "Std Error": float("nan"),
                    "t-statistic": float("nan"),
                    "p-value": float("nan"),
                })
        rows.extend({
            "Variable": name,
            "Coefficient": round(float(coef), 6),
            "Std Error": float("nan"),
            "t-statistic": float("nan"),
            "p-value": float("nan"),
        } for name, coef in zip(names, coefs))
        return pd.DataFrame(rows) if rows else None

    return None


def shapiro_wilk_test(residuals: np.ndarray | pd.Series) -> dict[str, float]:
    """Run a Shapiro-Wilk normality test on residuals."""
    if scipy_stats is None:
        raise ImportError("scipy is required for Shapiro-Wilk normality testing.")

    residuals_arr = np.asarray(residuals, dtype=float)
    residuals_arr = residuals_arr[np.isfinite(residuals_arr)]

    if residuals_arr.size < 3:
        return {"statistic": float("nan"), "p_value": float("nan")}

    if residuals_arr.size > 5000:
        rng = np.random.default_rng(42)
        residuals_arr = rng.choice(residuals_arr, size=5000, replace=False)

    statistic, p_value = scipy_stats.shapiro(residuals_arr)
    return {
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
    }
