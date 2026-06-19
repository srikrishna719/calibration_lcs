# Air Quality Sensor Calibration Lab v4.0

Research-grade Python application for calibrating low-cost air quality sensor (LCS)
measurements against reference-grade data using a scientifically robust, reproducible
machine learning pipeline.

## Overview

The application supports:

- Uploading reference and sensor CSV datasets
- Running immediately with bundled sample datasets (7 days, 168 rows)
- Configurable preprocessing (4 missing-value strategies, outlier removal for both datasets)
- Time alignment with user-selectable resample frequency and merge strategy
- Cross-correlation-based automatic lag detection and correction
- Comprehensive EDA with distribution, correlation, missing value, and anomaly analysis
- Training and comparing 5 ML calibration models (LR, Ridge, Lasso, RF, XGBoost)
- Polynomial feature expansion (degree 2/3) and pairwise interaction terms
- Variable selection — preview engineered features, multiselect subset for training
- Hyperparameter auto-tuning via RandomizedSearchCV + TimeSeriesSplit
- Calibration-specific metrics: RMSE, MAE, R², MAPE, Bias, Pearson r, Slope, Intercept
- Model explainability (feature importance for tree models, coefficients for linear)
- Multi-model comparison: scatter grid, time-series overlay, metrics bar chart
- Bland-Altman agreement plot (optional toggle)
- Post-calibration analysis with drift detection and rolling error analysis
- In-app editable README/notes tab (Markdown + live preview + download)
- Zenodo-ready export: calibrated CSV, model .pkl, metrics JSON, config YAML, metadata JSON
- **Inline tooltips & captions** on every widget across all pipeline steps
- **📚 Model Reference Guide** — one tab per model with formula, algorithm description, and hyperparameter table
- Full user control over every pipeline parameter

## Structure

```
modules/
  data_loader.py         — CSV load, timestamp parse, timezone normalization
  preprocessing.py       — IQR/z-score outlier removal; ffill/bfill/interpolate/interpolate_ffill imputation
  alignment.py           — Resample, cross-correlation lag detection, inner/nearest merge
  eda.py                 — Distributions, correlation heatmap, missing-value heatmap, time-series, anomalies
  feature_engineering.py — Lag, rolling, time features; polynomial expansion; pairwise interactions
  drift_analysis.py      — Rolling RMSE/MAE/Bias, dynamic drift detection, post-analysis figures
  exporter.py            — CSV/JSON/YAML/pickle serialization with provenance metadata
  plots.py               — Centralised plotting utilities (scatter, Bland-Altman, multi-model grid)
models/
  model_registry.py      — Supported regressors with parameter overrides
  train.py               — Time-aware split, TimeSeriesSplit CV, feature importance, RandomizedSearchCV
  predict.py             — Prediction helpers
evaluation/
  metrics.py             — Full calibration metric suite
  comparator.py          — Ranked leaderboard with all metrics
pipeline/
  run_pipeline.py        — 7-stage orchestrator (load → preprocess → align → EDA → model → post-analysis → export)
ui/
  app.py                 — Premium 8-step Streamlit UI with custom CSS theming
config/
  default.yaml           — Full pipeline configuration
sample_data/
  reference_dataset.csv         — 7-day reference-grade demo dataset (168 rows)
  low_cost_sensor_dataset.csv   — 7-day LCS demo dataset (168 rows)
docs/versions/
  v0_plan.md … v4_plan.md       — Detailed implementation plans per version
```

## Supported Models

| Model | Type | Hyperparameters (UI-exposed) |
|---|---|---|
| Linear Regression | Linear baseline | — |
| Ridge Regression | Regularised linear | `alpha` |
| Lasso Regression | Sparse linear | `alpha` |
| Random Forest | Ensemble (bagging) | `n_estimators`, `max_depth` |
| XGBoost | Ensemble (boosting) | `n_estimators`, `max_depth`, `learning_rate` |

> XGBoost is optional — gracefully skipped if not installed.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run ui/app.py
```

## Workflow (8 Steps)

1. **Upload Data** — Upload CSVs or use bundled samples; configure timestamp/target columns
2. **Preprocessing** — Select missing value strategy (`ffill`, `bfill`, `interpolate`, `interpolate_ffill`), outlier method and threshold; optionally preprocess reference data too
3. **Alignment** — Choose resample frequency, aggregation, merge strategy; view lag detection results
4. **EDA** — Explore distributions, correlations, missing values, time-series, anomalies
5. **Modelling** — Configure features (lag, rolling, time, polynomial, interactions), select models, tune hyperparameters, preview feature set, train all models
6. **Results** — View leaderboard, select model, inspect metrics and explainability charts; compare multiple models side-by-side
7. **Post-Analysis** — Predicted vs actual, residuals, time-series overlay, rolling error drift detection, Bland-Altman agreement plot
8. **Export** — Download calibrated CSV, model .pkl, metrics JSON, config YAML/JSON, metadata JSON

## Calibration Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| R² | Coefficient of determination |
| MAPE | Mean absolute percentage error |
| Bias | Mean error (predicted − actual) |
| Pearson r | Correlation coefficient |
| Slope | OLS fit slope (ideal = 1.0) |
| Intercept | OLS fit intercept (ideal = 0.0) |

## Version History

| Version | Date | Theme |
|---|---|---|
| v4.0 | 2026-06-13 | UX & Transparency — tooltips, Model Reference Guide, multi-model compare, expanded preprocessing |
| v3.0 | 2026-05-15 | Power-user features — polynomial expansion, hyperparameter tuning, Bland-Altman, editable README |
| v2.0 | 2026-04-20 | Research-grade 8-step pipeline, premium dark UI, full feature engineering, Zenodo export |
| v1.0 | pre-2026-04 | Multi-model basic pipeline, structured stages |
| v0.0 | pre-2026-04 | Proof-of-concept single-model prototype |

See [CHANGELOG.md](CHANGELOG.md) for full details.

## Notes

- `xgboost` is optional. If unavailable, the remaining models still work.
- The reference target column is not normalized, so calibrated predictions stay in native units.
- Alignment uses resampling + cross-correlation-based lag detection before merging.
- All exports include provenance metadata for reproducibility.
- The UI provides full control over every pipeline parameter at each step.
- Lag feature count is validated against dataset size to prevent data leakage warnings.
