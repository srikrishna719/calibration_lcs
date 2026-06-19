# Air Quality Sensor Calibration Lab v2.0

Research-grade Python application for calibrating low-cost air quality sensor (LCS)
measurements against reference-grade data using a scientifically robust, reproducible
machine learning pipeline.

## Overview

The application supports:

- Uploading reference and sensor CSV datasets
- Running immediately with bundled sample datasets (7 days, 168 rows)
- Configurable preprocessing (missing value strategies, outlier removal)
- Time alignment with user-selectable resample frequency and merge strategy
- Cross-correlation-based automatic lag detection and correction
- Comprehensive EDA with distribution, correlation, missing value, and anomaly analysis
- Training and comparing 5 ML calibration models (LR, Ridge, Lasso, RF, XGBoost)
- Calibration-specific metrics: RMSE, MAE, R², MAPE, Bias, Pearson r, Slope, Intercept
- Model explainability (feature importance for tree models, coefficients for linear)
- Post-calibration analysis with drift detection and rolling error analysis
- Zenodo-ready export: calibrated CSV, model .pkl, metrics JSON, config YAML, metadata JSON
- Full user control over every pipeline parameter

## Structure

- `modules/`
  - `data_loader.py` — dataset loading, validation, timezone normalization
  - `preprocessing.py` — missing values (interpolation/ffill), outlier removal (IQR/z-score)
  - `alignment.py` — resampling, lag detection, inner/nearest merge strategies
  - `eda.py` — distributions, correlation heatmap, missing value heatmap, time-series, anomalies
  - `feature_engineering.py` — lag features, rolling mean/std, time features
  - `drift_analysis.py` — rolling error computation, drift detection, residual analysis
  - `exporter.py` — CSV, JSON, YAML, pickle, and metadata serialization
- `models/`
  - `model_registry.py` — supported regressors with parameter overrides
  - `train.py` — time-aware training, cross-validation, feature importance extraction
  - `predict.py` — prediction helpers
- `evaluation/`
  - `metrics.py` — RMSE, MAE, R², MAPE, Bias, Pearson r, Slope, Intercept
  - `comparator.py` — ranked leaderboard with all metrics
- `pipeline/`
  - `run_pipeline.py` — 7-stage orchestrator (load → preprocess → align → EDA → model → post-analysis → export)
- `ui/`
  - `app.py` — premium 8-step Streamlit UI with custom CSS theming
- `config/`
  - `default.yaml` — full pipeline configuration
- `sample_data/`
  - `reference_dataset.csv` — 7-day reference-grade demo dataset (168 rows)
  - `low_cost_sensor_dataset.csv` — 7-day LCS demo dataset (168 rows)

## Supported Models

- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost (optional — gracefully skipped if not installed)

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
2. **Preprocessing** — Select missing value strategy, outlier method and threshold
3. **Alignment** — Choose resample frequency, aggregation, merge strategy; view lag detection
4. **EDA** — Explore distributions, correlations, missing values, time-series, anomalies
5. **Modelling** — Configure features, select models, tune hyperparameters, train all models
6. **Results** — View leaderboard, select model, inspect metrics and explainability charts
7. **Post-Analysis** — Predicted vs actual, residuals, time-series overlay, rolling error drift detection
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

## Notes

- `xgboost` is optional. If unavailable, the remaining models still work.
- The reference target column is not normalized, so calibrated predictions stay in native units.
- Alignment uses resampling + cross-correlation-based lag detection before merging.
- All exports include provenance metadata for reproducibility.
- The UI provides full control over every pipeline parameter at each step.
