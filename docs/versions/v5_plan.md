# v5.0 — Research-Oriented Calibration Workbench

**Date**: 2026-06-19 → 2026-07-02  
**Status**: Current  
**Conversations**: `e5102506-240b-4fc0-9a2d-857346013c0e` (infrastructure + chunks 1–3), `7e08a884-3e03-4efa-b4b7-ae2f32a79f41` (chunks 4–5 + UI finalization), `b5f81dcb-b4e8-4590-b63d-f49080d3c7bd` (code review + optimization)

---

## Goal

Major restructuring from a 9-step pipeline (v4) to a **13-step research-grade calibration workbench** with emphasis on scientific transparency, reproducibility, user control, and statistical interpretability. Every chart and table gets download buttons; every model decision is tracked in a reproducible `project_run.json`.

---

## User Requirements (original)

> The application should prioritize:
> - Scientific transparency
> - Reproducibility
> - User control
> - Downloadable outputs
> - Statistical interpretability
> - Calibration model development
>
> Do NOT implement: transferability analysis, LOSO/LOCO validation, hotspot/spatial analysis, drift monitoring systems, forecasting, AutoML, auto feature/predictor selection.

### Resolved Design Questions

| Question | Answer |
|----------|--------|
| Modelling Objective toggle | Metadata-only — saved in `project_run.json`, no auto-behavior |
| Model Summary Report format | **PDF** via `fpdf2` |
| EDA Scatter Plot Matrix | **User-selectable columns** via multiselect |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| 13-step pipeline (not 12) | README kept as read-only Step 13 per user request |
| OLS via statsmodels, not sklearn | Full statistical inference: p-values, std errors, t-stats, confidence intervals |
| `StatsmodelsOLSRegressor` wrapper | Provides sklearn-like `.fit()/.predict()/.coef_/.intercept_` API for consistency with training loop |
| 3 validation methods | TimeSeriesSplit (default), K-Fold, Holdout — researcher controls the validation strategy |
| Normalization as dedicated Step 7 | Explicit, reversible, with before/after summary — researchers need to see the effect on feature distributions |
| VIF + Shapiro-Wilk in Step 10 | Standard regression diagnostics that must be separate from model results |
| `_reset_downstream()` helper | Consolidates 5 repeated state-reset loops into named tuples |
| `fpdf2` for PDF reports | Lightweight, no system dependencies (unlike reportlab), adequate for tabular model summaries |
| `kaleido` for PNG export | Plotly's recommended headless renderer for chart-to-image conversion |
| `width='stretch'` migration | Streamlit deprecated `use_container_width` after 2025-12-31; migrated all 50+ occurrences |

---

## 13-Step Pipeline

| Step | Name | New? | Description |
|------|------|------|-------------|
| 1 | 📤 Upload Data | — | Upload CSVs or use bundled sample; optional column dropping |
| 2 | 🧹 Preprocessing | — | Independent sensor + reference cleaning (6 imputation strategies) |
| 3 | 🔗 Alignment | — | Resample + cross-correlation lag + merge |
| 4 | 📊 EDA | — | Distributions, correlation, missing heatmap, time-series, anomalies |
| 5 | 🎯 Variable Selection | **NEW** | Select target/features from merged dataset, preview modelling matrix |
| 6 | 🧬 Feature Engineering | **NEW** | Lag, rolling, polynomial, interaction, time features — moved out of old Step 5 |
| 7 | 📏 Normalization | **NEW** | StandardScaler / MinMaxScaler / RobustScaler with before/after stats |
| 8 | 🤖 Modelling | Modified | Model groups (Statistical / ML), objective toggle, multi-model training |
| 9 | ✅ Validation & Results | Modified | Sortable leaderboard, metric cards, feature importance, multi-model comparison |
| 10 | ⚠️ Statistical Diagnostics | **NEW** | Coefficient table (p-values), VIF, Shapiro-Wilk normality test |
| 11 | 📈 Residual Analysis | Modified | Pred vs Actual, Residual vs Fitted, Histogram, QQ Plot (rolling error removed) |
| 12 | 💾 Export | Modified | + `project_run.json`, + PDF model report, 9 download buttons |
| 13 | 📖 README | Modified | Read-only (editor removed), download-only |

---

## Changes by File

### NEW `modules/download_helpers.py`

Reusable download-button helpers for every displayed table and chart.

**Functions:**
- `_csv_bytes(df)` — serialize DataFrame to UTF-8 CSV bytes
- `_plotly_png_bytes(fig)` — render Plotly figure to PNG via kaleido
- `render_df_download(df, key, filename)` — compact CSV download button
- `render_chart_download(fig, source_df, key, filename_prefix)` — PNG + source data CSV buttons

---

### NEW `modules/normalization.py`

Dataset normalization with 3 sklearn scalers.

**Functions:**
- `_canonical_method(method)` — normalize user strings to scaler keys (with alias map)
- `_valid_numeric_columns(df, columns)` — filter to existing numeric columns
- `normalize_dataset(df, columns, method)` — apply scaler, optional `return_scaler`
- `_rounded_stat(series, stat_name)` — safe rounded statistic computation
- `get_normalization_summary(df_before, df_after, columns)` — before/after mean/std/min/max table

**Supported scalers:** `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `none`

---

### NEW `modules/diagnostics.py`

Statistical diagnostics for calibration models.

**Functions:**
- `compute_vif(X)` — Variance Inflation Factor via `statsmodels.stats.outliers_influence`
- `_result_names(ols_result, feature_names)` — infer coefficient names from statsmodels results
- `_as_named_series(values, names)` — convert arrays to named pandas Series
- `compute_coefficient_table(ols_result)` — build table with Variable, Coefficient, Std Error, t-statistic, p-value; falls back to sklearn coef_-only table if statsmodels unavailable
- `shapiro_wilk_test(residuals)` — Shapiro-Wilk normality test (samples to 5000 if needed)

---

### MODIFIED `modules/preprocessing.py`

#### `PreprocessingSummary` dataclass
- Added `percentage_removed: float = 0.0`

#### `clean_missing_values()`
- Added `"none"` strategy — no treatment (pass through)
- Added `"drop"` strategy — drop all rows with NaN
- Added `"interpolate_bfill"` strategy — linear interpolation + backward fill
- Total: 7 strategies (`none`, `drop`, `ffill`, `bfill`, `interpolate`, `interpolate_ffill`, `interpolate_bfill`)

---

### MODIFIED `modules/drift_analysis.py`

#### New: `create_qq_plot(predictions_df)`
- QQ plot of residuals against normal distribution
- Uses `scipy.stats.probplot` for theoretical quantiles
- Plots sample quantiles vs normal reference line
- Graceful fallback if scipy unavailable or insufficient data

---

### MODIFIED `modules/exporter.py`

Complete rewrite for v5.0:

#### New: `build_project_run(config, result, export_ts)`
- Generates `project_run.json` with full provenance: app version, modelling objective, validation method, features used, preprocessing config, normalization config, metrics, and export timestamp

#### New: `build_model_summary_pdf(result, config, leaderboard)`
- PDF model summary report via `fpdf2`
- Includes: header, model details, metrics table, feature importance, configuration summary
- Returns bytes for download

#### Modified: `build_metadata()`
- Updated software version to `"Calibration Lab v5.0"`

#### Modified: `SafeEncoder`
- JSON encoder handling numpy types, pandas objects, and non-serializable objects

---

### MODIFIED `models/model_registry.py`

#### Model Groups
```python
MODEL_GROUPS = {
    "Statistical Models": ["ols_regression", "multiple_linear_regression", "ridge", "lasso"],
    "Machine Learning Models": ["random_forest", "xgboost"],
}
```

#### New: `MODEL_DISPLAY_NAMES`, `MODEL_INFO`
- Display names for all 6 models
- Model metadata: name, description, strengths, limitations

#### Modified: `build_model_registry()`
- Added `ols_regression` placeholder (actual OLS via statsmodels in train.py)
- Added `multiple_linear_regression` (sklearn LinearRegression)

---

### MODIFIED `models/train.py`

#### New: `StatsmodelsOLSRegressor`
sklearn-like wrapper around `statsmodels.OLS`:
- `.fit(X, y)` → `sm.OLS(y, sm.add_constant(X)).fit()`
- `.predict(X)` → handles design matrix with const column
- `.coef_`, `.intercept_` properties
- Full `result_` attribute for accessing p-values, standard errors, t-statistics

#### New: `STATSMODELS_LINEAR_MODELS`
Set `{"ols_regression", "multiple_linear_regression"}` — routes these to `StatsmodelsOLSRegressor`

#### Modified: `TrainingResult` dataclass
New fields:
- `standard_errors: Optional[Dict[str, float]]`
- `t_statistics: Optional[Dict[str, float]]`
- `p_values: Optional[Dict[str, float]]`
- `coefficient_table: Optional[pd.DataFrame]`
- `validation_method: str`
- `validation_predictions: Optional[pd.DataFrame]`
- `residuals: Optional[pd.Series]`

#### New: `_canonical_validation_method(method)`
Normalizes user input to `"timeseriessplit"`, `"kfold"`, or `"holdout"`

#### New: `_clone_model(model)`, `_make_cv_splitter(method, n_samples, folds, random_state)`
Helpers for cloning models and creating CV splitters

#### Modified: `generate_validation_predictions()`
- Now supports all 3 validation methods (was TimeSeriesSplit only)
- `method` and `random_state` parameters added

#### Removed: `generate_time_series_cv_predictions()` *(v5.0.1 cleanup)*
- Backward-compat wrapper removed — never imported anywhere

#### Modified: `train_models()`
- Routes `ols_regression`/`multiple_linear_regression` to `StatsmodelsOLSRegressor`
- Captures coefficient table, standard errors, t-statistics, p-values from statsmodels result
- Computes residuals from validation predictions
- Supports `validation_method` config key

---

### MODIFIED `pipeline/run_pipeline.py`

#### New: `train_on_prepared_dataset()`
- Standalone function for training on a pre-prepared modelling dataset
- Used by the 13-step interactive UI where feature engineering, normalization, and variable selection happen in separate steps

#### Modified: `build_export_bundle()`
- Added `project_run.json` generation
- Added PDF model summary report
- Bundle now includes 9 downloadable artefacts

---

### MODIFIED `config/default.yaml`

New/updated keys:
```yaml
alignment:
  max_lag_steps: 0          # Changed from 3 — no auto-lag by default

normalization:              # NEW section
  method: none

training:
  validation_method: timeseriessplit    # NEW
  modelling_objective: ""              # NEW (metadata-only)
  selected_models:
    - ols_regression           # NEW
    - multiple_linear_regression # NEW
    - ridge
    - lasso
    - random_forest
    - xgboost
```

---

### MODIFIED `ui/app.py`

Major restructuring — the single largest change in v5.0.

#### Pipeline Architecture
- `STEPS` expanded from 9 → 13 items
- `STEP_KEYS` expanded correspondingly
- Dispatch table maps all 13 steps to render functions

#### Session State (new keys)
- `variable_selection_outputs` — modelling matrix from Step 5
- `feature_engineering_outputs` — featured dataset from Step 6
- `normalization_outputs` — normalized dataset + scaler from Step 7
- `normalization_method` — selected normalization method
- `residual_analysis_outputs` — model name for Step 11

#### New: `_reset_downstream(*keys)` *(v5.0.1 cleanup)*
Consolidates 5 inline state-reset loops into a single helper function with named reset groups:
- `_DOWNSTREAM_FROM_UPLOAD`
- `_DOWNSTREAM_FROM_PREPROCESSING`
- `_DOWNSTREAM_FROM_ALIGNMENT`
- `_DOWNSTREAM_FROM_MODELING`

#### New render functions
- `render_variable_selection()` — Step 5: target/feature column selection with dataset preview
- `render_feature_engineering()` — Step 6: all feature engineering controls, preview button
- `render_normalization()` — Step 7: scaler selection, apply, before/after summary
- `render_statistical_diagnostics()` — Step 10: coefficient table, VIF, Shapiro-Wilk in 3 tabs

#### Modified render functions

| Function | Changes |
|----------|---------|
| `render_modelling()` | Model groups display, objective toggle, validation method selector, feature subset multiselect |
| `render_results()` | Sortable leaderboard (sort-by dropdown + ascending toggle), enhanced metric cards |
| `render_residual_analysis()` | Renamed from `render_post_analysis`; 4 charts (Pred vs Actual, Residual vs Fitted, Histogram, QQ); rolling error/time-series removed |
| `render_export()` | 9 download buttons in 3×3 grid; project_run.json + PDF report |
| `render_readme()` | Read-only rendering (editor removed); download Markdown button only |

#### New helper functions
- `_model_label(key)` — human-readable model name
- `_format_metric_dataframe(df)` — round metrics to 2dp, p-values to 4dp
- `_best_row_style(row)` — highlight best model in leaderboard
- `_display_chart_with_downloads(fig, source_df, key, prefix)` — chart + download buttons
- `_normalization_summary_tables(summary)` — split before/after into separate DataFrames
- `_parse_positive_int_list(text)` — parse comma-separated integers for lag steps
- `_highlight_best(row)` — comparison table cell highlighting

#### Streamlit API Migration *(v5.0.1)*
- Replaced 50+ instances of `use_container_width=True` → `width='stretch'`
- Replaced 3 instances in `modules/download_helpers.py`
- Removed unused `Tuple` import from `typing`

---

### MODIFIED `requirements.txt`

```diff
+statsmodels
+kaleido
+fpdf2
```

---

## New in `config/default.yaml`

```yaml
normalization:
  method: none              # StandardScaler | MinMaxScaler | RobustScaler | none

training:
  validation_method: timeseriessplit   # timeseriessplit | kfold | holdout
  modelling_objective: ""             # "Interpretability Focused" | "Prediction Accuracy Focused" | ""
  selected_models:
    - ols_regression
    - multiple_linear_regression
    - ridge
    - lasso
    - random_forest
    - xgboost
```

---

## Backward Compatibility

- All new config keys have safe defaults (`none`, `timeseriessplit`, `""`)
- Old 9-step config files will work — new steps simply use defaults
- `StatsmodelsOLSRegressor` is transparent to the training loop (sklearn-like API)

---

## Verification

### App Startup
- `streamlit run ui/app.py` launches on `http://localhost:8501` ✅
- All 13 steps visible in sidebar ✅
- Version badge: "v5.0 · Calibration Lab" ✅

### Deprecation Warnings
- Zero `use_container_width` warnings after migration ✅
- Console output clean after page interactions ✅

### Pipeline Walk-through
- Upload → Load & Validate → success ✅
- Preprocessing → Run → success ✅
- Alignment → Run → success ✅
- All 13 steps navigate without `IndexError` ✅

### Syntax & Imports
- `py_compile.compile(app.py)` — Syntax OK ✅
- All new modules import successfully ✅

---

## Files Modified / Created

| File | Type | Summary |
|------|------|---------|
| `ui/app.py` | Modified | 13-step pipeline, 4 new render functions, helpers, download buttons, API migration |
| `modules/download_helpers.py` | **NEW** | Reusable chart/table download buttons |
| `modules/normalization.py` | **NEW** | 3 sklearn scalers + summary generation |
| `modules/diagnostics.py` | **NEW** | VIF, coefficient table, Shapiro-Wilk |
| `modules/preprocessing.py` | Modified | 3 new imputation strategies, percentage_removed |
| `modules/drift_analysis.py` | Modified | QQ plot function |
| `modules/exporter.py` | Modified | project_run.json, PDF report, SafeEncoder |
| `models/model_registry.py` | Modified | OLS, MLR, model groups, model metadata |
| `models/train.py` | Modified | StatsmodelsOLSRegressor, 3 validation methods, diagnostics capture |
| `pipeline/run_pipeline.py` | Modified | `train_on_prepared_dataset`, expanded export bundle |
| `config/default.yaml` | Modified | normalization, validation_method, modelling_objective |
| `requirements.txt` | Modified | +statsmodels, +kaleido, +fpdf2 |

---

*Previous version plan: [v4_plan.md](v4_plan.md)*
