# Calibration Lab — Version Changelog

> **Repository**: `d:\calibration_app`  
> **Project**: Air Quality Sensor Calibration Lab  
> **Purpose**: Research-grade ML pipeline for calibrating low-cost air quality sensors against reference-grade measurements.

---

## v5.0 — Research-Oriented Calibration Workbench *(2026-06-19 → 2026-07-02)*

**Theme**: Major pipeline restructuring — 13-step research-grade workbench with OLS regression (statsmodels), dedicated normalization/diagnostics steps, chart-level downloads, PDF reports, and full reproducibility via `project_run.json`.

### New Features

| # | Feature | Files |
|---|---------|-------|
| 1 | **13-step pipeline** — expanded from 9 steps with dedicated Variable Selection, Feature Engineering, Normalization, and Statistical Diagnostics steps | `ui/app.py` |
| 2 | **OLS Regression via statsmodels** — full p-values, standard errors, t-statistics, coefficient table; `StatsmodelsOLSRegressor` sklearn-like wrapper | `models/train.py`, `models/model_registry.py` |
| 3 | **3 validation methods** — TimeSeriesSplit (default), K-Fold, Holdout — researcher controls strategy | `models/train.py`, `ui/app.py` |
| 4 | **Normalization step** — StandardScaler, MinMaxScaler, RobustScaler with before/after summary table | `modules/normalization.py` (NEW), `ui/app.py` |
| 5 | **Statistical Diagnostics step** — VIF table, OLS coefficient table, Shapiro-Wilk normality test in 3 tabs | `modules/diagnostics.py` (NEW), `ui/app.py` |
| 6 | **Download buttons on every chart and table** — PNG + source CSV for charts, CSV for tables | `modules/download_helpers.py` (NEW), `ui/app.py` |
| 7 | **QQ Plot** in Residual Analysis — residuals vs normal distribution with reference line | `modules/drift_analysis.py`, `ui/app.py` |
| 8 | **`project_run.json` export** — full provenance metadata for reproducibility | `modules/exporter.py`, `pipeline/run_pipeline.py` |
| 9 | **PDF Model Summary Report** — downloadable report via `fpdf2` with metrics, features, config | `modules/exporter.py` |
| 10 | **Sortable leaderboard** — sort-by dropdown + ascending toggle with smart defaults | `ui/app.py` |
| 11 | **Model groups** — Statistical Models (OLS, MLR, Ridge, Lasso) and ML Models (RF, XGBoost) organized in UI | `models/model_registry.py`, `ui/app.py` |
| 12 | **Modelling objective toggle** — Interpretability vs Prediction Accuracy (metadata-only, saved in exports) | `ui/app.py`, `modules/exporter.py` |
| 13 | **3 new imputation strategies** — `none`, `drop`, `interpolate_bfill` added to preprocessing | `modules/preprocessing.py` |
| 14 | **README as read-only** — editor removed, download-only | `ui/app.py` |

### New Files
- `modules/download_helpers.py` — chart/table download buttons
- `modules/normalization.py` — dataset normalization with sklearn scalers
- `modules/diagnostics.py` — VIF, coefficient table, Shapiro-Wilk

### v5.0.1 — Code Review & Optimization *(2026-07-02)*
- **Deprecated API migration**: replaced 50+ `use_container_width=True` → `width='stretch'` across `app.py` and `download_helpers.py`
- **State reset consolidation**: `_reset_downstream()` helper replaces 5 inline loops
- **Dead code removal**: unused `generate_time_series_cv_predictions()` removed from `train.py`
- **Import cleanup**: removed unused `Tuple` from typing imports
- **Zero deprecation warnings** in console after migration

### Breaking Changes
- `STEPS` array expanded from 9 → 13 items (dispatch table updated)
- `TrainingResult` dataclass gains 7 new fields
- `build_export_bundle()` returns expanded dict with `project_run_json` and `model_summary_pdf`
- `use_container_width` parameter removed (now `width='stretch'`)

### Dependencies Added
- `statsmodels` — OLS regression, VIF computation
- `kaleido` — Plotly PNG export
- `fpdf2` — PDF report generation

---

## v4.0 — UX & Transparency Release *(2026-06-13)*

**Theme**: Scientific transparency, user guidance, and usability — inline help at every widget, full model explanations with formulas, expanded preprocessing options, smarter lag warnings, multi-model comparison, and several UI bug fixes.

### New Features

| # | Feature | Files |
|---|---------|-------|
| 1 | **Inline tooltips & captions on all widgets** across Steps 1–5 — every input has a `help=` popup explaining purpose, valid values, and air-quality-specific guidance | `ui/app.py` |
| 2 | **📚 Model Reference Guide expander** (Step 5) — one tab per model showing algorithm description, LaTeX formula, what inputs it needs, and a full hyperparameter table distinguishing UI-exposed (✅) from hidden/auto-tuned (❌) params | `ui/app.py` |
| 3 | **Multi-model comparison** (Step 5, post-training) — multiselect N models; 3 tabs: scatter grid (2-per-row), time-series overlay of all models, metrics table with best-value highlighting + bar chart | `ui/app.py` |
| 4 | **Expanded missing-value strategies** — added `bfill` (backward-fill) and `interpolate` (interpolation only) alongside existing `ffill` and `interpolate_ffill`; all 4 exposed in the preprocessing UI | `modules/preprocessing.py`, `ui/app.py` |
| 5 | **Lag features UX fix** — expanded caption explaining "N steps ago" concept with hourly examples, dataset-size warning (fires when `max_lag × 20 > n_rows`), per-strategy tooltips | `ui/app.py` Step 5 |
| 6 | **Upload page tooltip coverage** — file uploader, config fields (timestamp col, target col, timezone), sample data checkbox all annotated with format guidance | `ui/app.py` Step 1 |

### Bug Fixes

| # | Bug | Fix |
|---|-----|-----|
| 1 | `KeyError: 'model'` in comparison metrics bar chart | Corrected column name to `'model_name'` (matches `comparator.py`) |
| 2 | Modelling "Next ➡️" button looped back to same page | Button was indented inside `if out is not None:` block; moved to function body level with `disabled=True` guard |
| 3 | Upload page icon text overlap | CSS: zeroed font-size of Material Icons fallback `<span>`, replaced with `⬆` via `::before` |

### Smoke Test (2026-06-13)
All **11 pipeline checks passed** (0 failures):
- All 4 imputation strategies produce zero NaNs
- Stages 1–5 (Load → Preprocess → Align → EDA → Model) complete without error
- Leaderboard `model_name` column present and filterable
- Training on 165-row merged dataset: best model = lasso

---

## v3.0 — Research-Grade Extended Pipeline *(2026-05-15)*


**Theme**: Power-user features — polynomial feature expansion, hyperparameter tuning, multi-model visual comparisons, Bland-Altman, editable README.

### New Features
| # | Feature | Files |
|---|---------|-------|
| 1 | Reference-data preprocessing toggle (outlier opt-in) | `preprocessing.py`, `run_pipeline.py`, `app.py` Step 2 |
| 2 | Polynomial features (degree 2/3) on user-chosen column subset | `feature_engineering.py`, `app.py` Step 5 |
| 3 | Pairwise interaction terms (col_i × col_j) | `feature_engineering.py`, `app.py` Step 5 |
| 4 | Variable selection — preview engineered features, multiselect subset | `train.py`, `app.py` Step 5 |
| 5 | Hyperparameter auto-tuning (RandomizedSearchCV + TimeSeriesSplit) per model | `train.py`, `app.py` Step 5 |
| 6 | Extended time/date features: unix, julian, calendar date, cyclical sin/cos, season, day name, is_weekend | `feature_engineering.py`, `app.py` Step 5 |
| 7 | Enhanced scatter: 1:1 ideal line + OLS regression fit + Pearson r annotation | `plots.py` (NEW), `app.py` Steps 6 & 7 |
| 8 | Multi-model comparison: scatter grid, time-series overlay, metrics bar chart | `plots.py`, `app.py` Step 6 |
| 9 | Bland-Altman agreement plot (optional toggle) | `plots.py`, `app.py` Step 7 |
| 10 | In-app editable README/notes tab (Markdown + live preview + download) | `app.py` Step 9 |

### New File
- `modules/plots.py` — centralised plotting utilities

### Breaking Changes
- `cached_modeling()` now takes `feature_subset_json` as third argument
- `run_modeling_stage()` now accepts `feature_subset` parameter
- `train_models()` now accepts `feature_subset` and `config['training']['tuning']`
- `TrainingResult` dataclass gains `best_params` field
- `create_time_features()` now accepts a `config` dict for flags

---

## v2.0 — Research-Grade 8-Step Pipeline *(2026-04-20)*

**Theme**: Full ML pipeline transformation — scientifically rigorous, reproducible, premium UI.

### New Features
- **8-step Streamlit workflow** (Upload → Preprocessing → Alignment → EDA → Modelling → Results → Post-Analysis → Export)
- **Premium dark UI**: custom CSS, Inter font, glassmorphism metric cards, gradient header, step progress bar
- **5 ML models**: Linear Regression, Ridge, Lasso, Random Forest, XGBoost (optional)
- **8 calibration metrics**: RMSE, MAE, R², MAPE, Bias, Pearson r, OLS Slope, OLS Intercept
- **Time-aware ML**: chronological train/test split + `TimeSeriesSplit` cross-validation (no data leakage)
- **Cross-correlation lag detection**: automatic optimal lag estimation between sensor and reference
- **EDA module**: distributions, correlation heatmap, missing-value heatmap, time-series, anomaly detection
- **Feature engineering**: lag features, rolling mean/std, basic time features (hour, DOW, DOM)
- **Model explainability**: feature importance (tree models), coefficients (linear models)
- **Drift analysis**: rolling RMSE/MAE/Bias, dynamic threshold drift detection
- **Post-calibration analysis**: Predicted vs Actual, residual plot, time-series overlay, rolling error, residual histogram
- **Zenodo-ready export**: calibrated CSV, model .pkl, metrics JSON, config YAML/JSON, metadata JSON
- **Full config control**: `config/default.yaml` + UI overrides at every step
- **Bundled sample data**: 7-day reference + LCS datasets (168 rows each)

### Architecture
```
modules/
  data_loader.py       — CSV load, timestamp parse, timezone normalization
  preprocessing.py     — IQR/z-score outlier removal, interpolate/ffill imputation
  alignment.py         — Resample, lag detect, inner/nearest merge
  eda.py               — EDA figures and anomaly detection
  feature_engineering.py — Lag, rolling, time features
  drift_analysis.py    — Rolling error, drift detection, post-analysis figures
  exporter.py          — CSV/JSON/YAML/pickle serialization
models/
  model_registry.py    — Supported regressors with param overrides
  train.py             — Time-aware split, TimeSeriesSplit CV, feature importance
  predict.py           — Prediction helpers
evaluation/
  metrics.py           — Full calibration metric suite
  comparator.py        — Ranked leaderboard
pipeline/
  run_pipeline.py      — 7-stage orchestrator
ui/
  app.py               — Premium 8-step Streamlit UI
```

---

## v1.0 — Multi-Model Basic Pipeline *(pre-2026-04-20)*

**Theme**: Added multiple model support and structured pipeline stages over v0.

### Features Added over v0
- Multiple scikit-learn models (Linear Regression, Ridge, Lasso)
- Basic train/test split with evaluation
- RMSE, MAE, R² metrics
- CSV upload + download
- Simple time alignment (resample)
- Basic missing value handling
- Minimal Streamlit UI with tabs

---

## v0.0 — Original Prototype *(pre-2026-04-20)*

**Theme**: Proof-of-concept single-model calibration app.

### Features
- Upload one reference CSV and one sensor CSV
- Simple Linear Regression calibration
- Basic RMSE metric
- Download calibrated CSV
- Minimal Streamlit interface

---

*Each version's detailed implementation plan is documented in `docs/versions/`.*
