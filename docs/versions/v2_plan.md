# v2.0 — Research-Grade 8-Step Pipeline

**Date**: 2026-04-20  
**Status**: Superseded by v3.0  
**Conversation**: `fa8c64b9-c52e-49f4-9f6a-b4daca50a6df`

---

## Goal
Transform the basic calibration app into a **research-grade, reproducible ML pipeline** with a premium 8-step UI, scientifically rigorous evaluation, and Zenodo-ready export.

## Design Principles
- Every pipeline parameter must be user-controllable
- Time-aware ML only (no data leakage)
- Full provenance for reproducibility
- Premium dark-themed Streamlit UI with Plotly

---

## 8-Step Workflow

| Step | Name | Description |
|------|------|-------------|
| 1 | Upload Data | Upload CSVs or use bundled 7-day sample datasets |
| 2 | Preprocessing | Missing-value imputation + IQR/z-score outlier removal |
| 3 | Alignment | Resample + cross-correlation lag detection + inner/nearest merge |
| 4 | EDA | Distributions, correlation heatmap, missing heatmap, time-series, anomalies |
| 5 | Modelling | Feature engineering + train 5 models with TimeSeriesSplit CV |
| 6 | Results | Leaderboard, metric cards, feature importance/coefficients |
| 7 | Post-Analysis | Predicted vs actual, residuals, time-series overlay, rolling error drift |
| 8 | Export | Calibrated CSV, model .pkl, metrics JSON, config YAML/JSON, metadata JSON |

---

## New Modules

### `modules/data_loader.py`
- `load_csv()` — accepts file path, file-like object, or DataFrame
- `validate_dataset()` — checks timestamp column, numeric columns present
- `parse_and_normalize_timestamps()` — timezone-aware UTC normalization, deduplication
- `load_and_validate_dataset()` — end-to-end loader

### `modules/preprocessing.py`
- `PreprocessingSummary` dataclass
- `clean_missing_values()` — `interpolate_ffill` or `ffill`
- `detect_outlier_mask_iqr()` — IQR-based row mask
- `detect_outlier_mask_zscore()` — z-score-based row mask
- `remove_outliers()` — applies mask, returns clean DataFrame
- `preprocess_dataset()` — full preprocessing with summary

### `modules/alignment.py`
- `resample_timeseries()` — pandas resample with configurable aggregation
- `detect_optimal_lag()` — cross-correlation over [-N, +N] lag range
- `apply_lag()` — shift sensor columns by detected lag
- `align_and_merge_datasets()` — full alignment pipeline, returns merged DataFrame + metadata

### `modules/eda.py`
- `summarize_missing_values()` — per-column missing count and %
- `create_missing_value_heatmap()` — Plotly heatmap
- `detect_basic_anomalies()` — z-score row flagging
- `create_distribution_figure()` — histogram + box marginal
- `create_before_after_distributions()` — overlay before/after cleaning
- `create_correlation_heatmap()` — `px.imshow` correlation matrix
- `create_time_series_figure()` — multi-column time-series
- `create_anomaly_figure()` — time-series with anomaly markers
- `generate_eda_outputs()` — orchestrator returning dict of all figures

### `modules/feature_engineering.py`
- `select_feature_columns()` — sensor columns, optional meteorological cols
- `create_lag_features()` — shifted copies
- `create_rolling_features()` — rolling mean + optional std
- `create_time_features()` — hour, day-of-week, day-of-month
- `engineer_features()` — orchestrator

### `modules/drift_analysis.py`
- `compute_rolling_errors()` — rolling RMSE, MAE, Bias
- `detect_drift_periods()` — dynamic threshold = median × multiplier
- `create_rolling_error_figure()` — line chart with drift markers
- `create_residual_histogram()` — px.histogram with box marginal
- `create_predicted_vs_actual_figure()` — scatter + 1:1 line
- `create_residual_vs_predicted_figure()` — scatter + zero hline
- `create_time_series_overlay_figure()` — reference vs calibrated overlay
- `generate_post_analysis_outputs()` — orchestrator

### `modules/exporter.py`
- `export_dataframe_csv_bytes()`
- `export_metrics_json_bytes()`
- `export_model_bytes()` — pickle
- `export_config_json_bytes()`
- `export_config_yaml_bytes()` — PyYAML fallback to JSON
- `build_metadata()` — Zenodo-ready provenance dict
- `export_metadata_json_bytes()`

### `models/model_registry.py`
- `build_model_registry()` — LR, Ridge, Lasso, RF, XGBoost (optional)
- `get_selected_models()` — filter + apply param overrides from config

### `models/train.py`
- `TrainingResult` dataclass — model, metrics, test/full predictions, feature importance, coefficients, intercept
- `extract_feature_importance()` — tree importance or linear coefficients
- `prepare_training_matrices()` — features / target / timestamps split
- `split_train_test_by_time()` — chronological leakage-safe split
- `generate_time_series_cv_predictions()` — `TimeSeriesSplit` OOF predictions
- `train_models()` — full training loop

### `models/predict.py`
- `predict_with_model()` — inference returning timestamp + prediction DataFrame

### `evaluation/metrics.py`
- `mean_absolute_percentage_error()` — ignores zero targets
- `bias()` — mean(predicted − actual)
- `pearson_r()` — numpy corrcoef
- `fit_slope_intercept()` — `np.polyfit` OLS
- `calculate_regression_metrics()` — full suite + optional CV metrics

### `evaluation/comparator.py`
- `create_leaderboard()` — sorted DataFrame with rank column
- `select_best_model()` — retrieves top-ranked TrainingResult

### `pipeline/run_pipeline.py`
- `load_config()` — YAML/JSON config loader
- `load_input_data()` — Stage 1
- `run_preprocessing_stage()` — Stage 2
- `run_alignment_stage()` — Stage 3
- `run_eda_stage()` — Stage 4
- `run_modeling_stage()` — Stage 5 (feature engineering + training)
- `run_post_analysis_stage()` — Stage 6
- `build_export_bundle()` — Stage 7
- `run_full_pipeline()` — non-interactive end-to-end runner

---

## UI (`ui/app.py`)

### Styling
- Google Fonts Inter
- Dark gradient header (`#1e1b4b → #312e81 → #4338ca`)
- Step progress bar (dots: done=green, active=gradient, pending=dark)
- Metric cards with color coding (good/warn/bad)
- Section cards, info pills
- Custom sidebar styling

### Caching
All heavy computation wrapped in `@st.cache_data`:
- `cached_load_sample()`
- `cached_config_from_text()`
- `cached_load_input()`
- `cached_preprocessing()`
- `cached_alignment()`
- `cached_eda()`
- `cached_modeling()`

### Session State
- `current_step`, `config`, `input_label`
- `data_outputs`, `preprocessing_outputs`, `alignment_outputs`
- `eda_outputs`, `modeling_outputs`, `selected_model_name`
- `post_analysis_outputs`, `export_bundle`

---

## Configuration (`config/default.yaml`)

```yaml
app:
  random_state: 42
data:
  timestamp_column: timestamp
  target_column: pm25
  timezone: UTC
  sensor_prefix: sensor
  reference_prefix: reference
preprocessing:
  missing_strategy: interpolate_ffill
  outlier_method: iqr
  outlier_threshold: 1.5
alignment:
  resample_rule: 1h
  aggregation: mean
  max_lag_steps: 3
  lag_column: auto
  merge_strategy: inner
feature_engineering:
  enabled: true
  lag_steps: [1, 2, 3]
  rolling_windows: [3, 6]
  rolling_std: true
  add_time_features: true
  optional_columns:
    - sensor_temperature
    - sensor_humidity
    - sensor_voc
training:
  test_size: 0.2
  cross_validation_folds: 5
  selected_models: [linear_regression, ridge, lasso, random_forest, xgboost]
evaluation:
  sort_by: rmse
  ascending: true
drift_analysis:
  rolling_window: 6
  drift_threshold: 1.5
```

---

## Sample Data
- `sample_data/reference_dataset.csv` — 168 rows, 7-day hourly reference-grade PM2.5
- `sample_data/low_cost_sensor_dataset.csv` — 168 rows, 7-day hourly LCS PM2.5

---

## Dependencies (`requirements.txt`)
```
pandas
numpy
scikit-learn
xgboost
plotly
streamlit
PyYAML
scipy
joblib
```
