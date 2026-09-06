"""The Model Reference Guide shown on the Modelling step.

Static explanatory content -- formulas, hyperparameter tables, guidance --
extracted from ui/app.py, where it accounted for roughly a tenth of the
module while containing no workflow logic.
"""

from __future__ import annotations

import streamlit as st


def render_model_reference_guide() -> None:
    """Render the collapsible per-model reference."""
    # ---- Model Reference Guide ----
    with st.expander("📚 Model Reference Guide — How Each Model Works", expanded=False):
        st.caption(
            "Everything you need to know before training: how each model works, "
            "the formula it uses, what inputs it needs, and every hyperparameter available. "
            "Parameters marked ✅ are exposed in the UI below; ❌ use defaults or Auto-Tuning."
        )
        _mtabs = st.tabs([
            "📐 Linear Regression",
            "🔵 Ridge",
            "🟡 Lasso",
            "🌲 Random Forest",
            "⚡ XGBoost",
        ])

        with _mtabs[0]:
            st.markdown("#### Linear Regression (Ordinary Least Squares)")
            st.markdown(
                "The baseline model. Fits a straight-line relationship between your sensor features and "
                "the reference PM value by **minimising the sum of squared errors**. "
                "No regularisation — every feature gets a coefficient. Fast and fully interpretable."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n")
            st.markdown("**Objective (what it minimises):**")
            st.latex(r"\min_{\boldsymbol{\beta}} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2")
            st.markdown("**What inputs does it need?**")
            st.info(
                "• **Numeric feature columns only** (sensor readings, lag features, rolling means, time features).\n"
                "• Works best when the sensor–reference relationship is roughly **linear** "
                "(e.g. raw PM channel vs reference PM).\n"
                "• Sensitive to **correlated features** (multicollinearity) — if you add many lag/rolling columns, "
                "use Ridge or Lasso instead.\n"
                "• Feature scaling is NOT required, but helps compare coefficient magnitudes."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `fit_intercept` | `True` | ❌ | Fit β₀ (intercept). Almost always True. |\n"
                "| `positive` | `False` | ❌ | Force all coefficients ≥ 0. Rarely needed. |\n\n"
                "> Linear Regression has **no regularisation parameters**. "
                "If the model overfits or collinear features are a concern, switch to Ridge or Lasso."
            )

        with _mtabs[1]:
            st.markdown("#### Ridge Regression (L2 Regularisation)")
            st.markdown(
                "Extends OLS with an **L2 penalty** on coefficient size. "
                "All coefficients are **shrunk towards zero** (but never exactly zero). "
                "Ideal when many correlated features are present — it distributes weight across them rather than "
                "picking one arbitrarily."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \beta_0 + \sum_{j=1}^{n} \beta_j x_j")
            st.markdown("**Objective:**")
            st.latex(
                r"\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 "
                r"+ \alpha \sum_{j=1}^{n} \beta_j^2 \right]"
            )
            st.markdown("α controls the trade-off: **α → 0** = plain OLS · **α → ∞** = all coefficients → 0.")
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns.\n"
                "• **Recommended** when you have lag/rolling features that are correlated with each other.\n"
                "• Feature scaling helps (so the penalty is applied equally across all features).\n"
                "• For PM calibration: good with humidity-corrected features (PM × RH terms)."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `alpha` | `1.0` | ✅ | L2 penalty strength. Try: 0.01, 0.1, 1, 10, 100. |\n"
                "| `fit_intercept` | `True` | ❌ | Whether to fit the intercept β₀. |\n"
                "| `solver` | `'auto'` | ❌ | Algorithm: `'auto'`, `'svd'`, `'cholesky'`, `'lsqr'`. |\n"
                "| `max_iter` | `None` | ❌ | Max iterations for iterative solvers. |\n"
                "| `tol` | `1e-4` | ❌ | Convergence tolerance. |\n\n"
                "> Auto-tuning searches: α ∈ {0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100}."
            )

        with _mtabs[2]:
            st.markdown("#### Lasso Regression (L1 Regularisation)")
            st.markdown(
                "Like Ridge, but uses the **absolute value** of coefficients as the penalty. "
                "The key difference: L1 can drive some coefficients to **exactly zero**, "
                "automatically removing irrelevant features. "
                "Acts as a built-in feature selector — useful when you suspect only a few inputs really matter."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \beta_0 + \sum_{j=1}^{n} \beta_j x_j")
            st.markdown("**Objective:**")
            st.latex(
                r"\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 "
                r"+ \alpha \sum_{j=1}^{n} |\beta_j| \right]"
            )
            st.markdown("Higher α → more zero coefficients → sparser model.")
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns.\n"
                "• Very effective after polynomial expansion: Lasso will automatically discard the "
                "polynomial terms that don't improve fit.\n"
                "• Feature scaling is important so all features compete fairly for the L1 budget.\n"
                "• For PM calibration: start with small α (0.001–0.01) and increase until only meaningful features remain."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `alpha` | `0.01` | ✅ | L1 penalty. Higher = more features zeroed out. |\n"
                "| `fit_intercept` | `True` | ❌ | Whether to fit β₀. |\n"
                "| `max_iter` | `1000` | ❌ | Max iterations for coordinate descent. Increase if you see convergence warnings. |\n"
                "| `tol` | `1e-4` | ❌ | Convergence tolerance. |\n"
                "| `selection` | `'cyclic'` | ❌ | `'cyclic'` (round-robin) or `'random'` update order. |\n\n"
                "> Auto-tuning searches: α ∈ {0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}."
            )

        with _mtabs[3]:
            st.markdown("#### Random Forest Regressor")
            st.markdown(
                "Builds **T independent decision trees**, each trained on a random bootstrap sample of the data "
                "and using only a random subset of features at each split (to decorrelate trees). "
                "The final prediction is the **average** across all trees. "
                "Naturally handles non-linear relationships and requires no feature scaling."
            )
            st.markdown("**Formula:**")
            st.latex(r"\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{x})")
            st.markdown(
                "Each tree *fₜ* is grown on a bootstrap sample using `max_features` features per split. "
                "Variance is reduced by averaging; bias is controlled by tree depth."
            )
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns — **no scaling required**.\n"
                "• Handles correlated lag/rolling features well (random feature subsets decorrelate trees).\n"
                "• Works out-of-the-box with minimal tuning. Increasing n_estimators always helps (up to a point).\n"
                "• For PM calibration: often outperforms linear models when humidity causes non-linear "
                "hygroscopic particle growth effects."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `n_estimators` | `200` | ✅ | Number of trees. More = stable but slower. 100–500 typical. |\n"
                "| `max_depth` | `10` | ✅ | Max tree depth. None = fully grown. 5–15 prevents overfitting. |\n"
                "| `min_samples_split` | `2` | ❌ (auto-tuned) | Min samples required to split a node. Higher = simpler trees. |\n"
                "| `min_samples_leaf` | `1` | ❌ (auto-tuned) | Min samples in any leaf. Higher = smoother predictions. |\n"
                "| `max_features` | `'sqrt'` | ❌ (auto-tuned) | Features per split: `'sqrt'`, `'log2'`, or float (fraction). |\n"
                "| `bootstrap` | `True` | ❌ | Use bootstrap sampling per tree. |\n"
                "| `oob_score` | `False` | ❌ | Compute out-of-bag validation score for free. |\n"
                "| `n_jobs` | `-1` | ❌ | CPU threads (−1 = all cores). |\n\n"
                "> Auto-tuning searches: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features."
            )

        with _mtabs[4]:
            st.markdown("#### XGBoost (Extreme Gradient Boosting)")
            st.markdown(
                "Builds trees **sequentially** — each new tree is trained to correct the errors (residuals) "
                "of all previous trees. Unlike Random Forest (parallel + average), XGBoost **boosts** performance "
                "step by step. Has built-in L1 + L2 regularisation on leaf weights. "
                "Typically the most accurate model on tabular data, but needs careful tuning."
            )
            st.markdown("**Formula (final prediction after K rounds):**")
            st.latex(r"\hat{y}^{(K)} = \sum_{k=1}^{K} \eta \cdot f_k(\mathbf{x})")
            st.markdown("where η = `learning_rate` and each tree minimises:")
            st.latex(
                r"\mathcal{L}^{(k)} = \sum_{i} l\!\left(y_i,\, \hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)\right) + \Omega(f_k)"
            )
            st.markdown("**Regularisation term on each tree:**")
            st.latex(
                r"\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|"
            )
            st.markdown(
                "T = number of leaves · wⱼ = leaf scores · "
                "γ = min gain to split · λ = L2 (`reg_lambda`) · α = L1 (`reg_alpha`)"
            )
            st.markdown("**What inputs does it need?**")
            st.info(
                "• Any numeric feature columns — **no scaling required**.\n"
                "• Handles missing values internally, but we pre-impute in the preprocessing step.\n"
                "• Benefits the most from rich feature engineering (lag, rolling, interaction terms).\n"
                "• Best model for large datasets (> 500 rows) with non-linear sensor behaviour.\n"
                "• Requires `xgboost` package: install with `pip install xgboost`."
            )
            st.markdown("**All Hyperparameters:**")
            st.markdown(
                "| Parameter | Default | Exposed in UI | What it does |\n"
                "|-----------|---------|---------------|--------------|\n"
                "| `n_estimators` | `200` | ✅ | Number of boosting rounds (trees). |\n"
                "| `max_depth` | `4` | ✅ | Max depth per tree. 3–6 is typical; lower = simpler. |\n"
                "| `learning_rate` (η) | `0.05` | ✅ | Step size per round. Lower needs more trees. |\n"
                "| `subsample` | `0.8` | ❌ (auto-tuned) | Fraction of rows sampled per tree. <1 adds randomness. |\n"
                "| `colsample_bytree` | `0.8` | ❌ (auto-tuned) | Fraction of features used per tree. |\n"
                "| `reg_alpha` (α) | `0.0` | ❌ (auto-tuned) | L1 on leaf weights. Promotes sparse leaf scores. |\n"
                "| `reg_lambda` (λ) | `1.0` | ❌ (auto-tuned) | L2 on leaf weights. Smooths predictions. |\n"
                "| `gamma` | `0` | ❌ | Min loss reduction to split a node. Higher = fewer splits. |\n"
                "| `min_child_weight` | `1` | ❌ | Min sum of instance weight in a leaf. Prevents tiny splits. |\n"
                "| `n_jobs` | `-1` | ❌ | CPU threads. |\n\n"
                "> Auto-tuning searches: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda."
            )

