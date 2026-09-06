# CaliSenseAQ

Python application for calibrating low-cost air quality sensor (LCS)
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
- Residual analysis: predicted vs actual, residual histogram, residual vs fitted, Q-Q plot
- Guards against predictors that cannot be used in practice — reference-instrument
  channels, and derived columns that reconstruct the target or a reference reading
- Explicit handling of repeated timestamps, so a multi-device file is reduced to one
  series deliberately rather than by silently keeping an arbitrary row
- Scaling applied inside the model, so it is fit per training fold and travels with
  the exported `.pkl`
- In-app editable README/notes tab (Markdown + live preview + download)
- Export outputs: compact/full calibrated CSVs, model .pkl, selected/all-model metrics JSON, config YAML, researcher PDF report, and optional Zenodo/demo metadata JSON
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
  leakage.py             — Pairwise detection of predictors encoding the target or a reference channel
  exporter.py            — CSV/JSON/YAML/pickle serialization with provenance metadata
  plots.py               — Plotting utilities (scatter, multi-model grid, residual diagnostics)
models/
  model_registry.py      — Supported regressors with parameter overrides
  train.py               — Time-aware split, TimeSeriesSplit CV, feature importance, RandomizedSearchCV
  predict.py             — Prediction helpers
evaluation/
  metrics.py             — Full calibration metric suite
  comparator.py          — Ranked leaderboard with all metrics
pipeline/
  run_pipeline.py        — 6-stage orchestrator (load → preprocess → align → EDA → model → export)
ui/
  app.py                 — 13-step Streamlit workflow
  theme.py               — Light/dark palettes and the CSS block
  model_guide.py         — Per-model reference content shown on the Modelling step
config/
  default.yaml           — Full pipeline configuration
  validation.py          — Fills missing sections from defaults; rejects unusable values by name
sample_data/
  reference_dataset.csv         — 7-day reference-grade demo dataset (168 rows)
  low_cost_sensor_dataset.csv   — 7-day LCS demo dataset (168 rows)
docs/versions/
  v0_plan.md … v5_plan.md       — Detailed implementation plans per version
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

## Tests

```bash
pytest                      # full suite
pytest -m "not slow"        # skip the multi-model pipeline runs
pytest -m realdata          # only the 2025 co-location tests (skipped if data/ is absent)
```

## Workflow (13 steps)

Steps 10-12 appear in Advanced mode only.

1. **Upload Data** — Upload CSVs or use bundled samples; configure timestamp/target columns; resolve repeated timestamps if the file holds several devices
2. **Preprocessing** — Missing-value strategy, outlier method and threshold, independently for reference and sensor
3. **Alignment** — Resample frequency, aggregation, merge strategy; lag detection results and any non-numeric columns dropped
4. **EDA** — Distributions, correlations, missing values, time-series, anomalies
5. **Variable Selection** — Choose target and predictors; reference channels and target-encoding columns are excluded by default, with an explicit opt-in
6. **Feature Engineering** — Lag, rolling, polynomial, interaction and time features
7. **Normalization** — Choose a scaler and preview its effect; the scaler itself is fit inside the model
8. **Modelling** — Select models, set hyperparameters or enable nested auto-tuning, optionally restrict the feature subset, train
9. **Validation & Results** — Leaderboard, metrics, explainability, multi-model comparison
10. **Statistical Diagnostics** — OLS coefficient table, VIF, Shapiro-Wilk *(Advanced)*
11. **Residual Analysis** — Residual vs fitted, histogram, Q-Q plot *(Advanced)*
12. **README** — In-app editable notes *(Advanced)*
13. **Export** — Calibrated CSV, model `.pkl`, metrics JSON, all-model metrics JSON, researcher PDF, config YAML/JSON, metadata JSON


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
- When a scaler is selected, the exported `model.pkl` is a scikit-learn `Pipeline`
  containing it, so the model applies to raw sensor readings directly.
- Auto-tuning searches inside each validation fold, so tuned metrics stay honest; it
  costs roughly one extra search per fold.
- Residuals are `predicted - actual` throughout.
- MAPE excludes reference values below 1 ug/m3, where a percentage reflects the
  near-zero denominator rather than the model; the excluded share is reported
  alongside it. Judge low-concentration performance by RMSE or MAE.
- Outlier removal drops a row when any screened column flags it, so the loss grows
  with column count. The Preprocessing step breaks down what each column
  contributes; `preprocessing.outlier_columns` narrows the screen.
- A partial config file is completed from the packaged defaults, so it only needs
  the settings you want to override. Unknown top-level sections are reported as
  likely typos rather than silently ignored.
- Alignment uses resampling + cross-correlation-based lag detection before merging.
- All exports include provenance metadata for reproducibility.
- The UI provides full control over every pipeline parameter at each step.
- Lag feature count is validated against dataset size to prevent data leakage warnings.
