# v3.0 — Research-Grade Extended Pipeline

**Date**: 2026-05-15  
**Status**: Previous *(superseded by [v4.0](v4_plan.md))*  
**Conversation**: `6a4b7ab9-7a9d-42b5-8284-ac10104e2b47`

---

## Goal
Add 9 major research-grade enhancements requested by the user to transform the v2.0 pipeline into a fully flexible, scientifically powerful calibration workbench.

---

## User Requirements (original)

> 1. option to add preprocess for reference data
> 2. adding Higher order (degree) and also interaction terms in input
> 3. options to choose input variables for the model
> 4. Hyperparameter tuning option
> 5. scatter plots
> 6. comparing plots with multiple models
> 7. multiple variables - unix date, julian date, date, categorical terms for days, seasons...
> 8. 1:1 lines & linear fit lines
> 9. readme tab to give external instructions

### Design Decisions Confirmed by User
- Hyperparameter tuning: user controls n_iter freely (no hard cap)
- Polynomial features: user-chosen column subset only (avoid feature explosion)
- Bland-Altman plot: optional toggle (on/off)
- README tab: editable in all phases (persistent session state)

---

## Changes by File

### NEW `modules/plots.py`

Central plotting utilities for calibration science.

**Functions:**

#### `_ols_fit(x, y)`
Internal helper. Returns `(slope, intercept, x_range, y_fit)` from `np.polyfit`. Returns `None` if < 2 valid points.

#### `_pearson(x, y)`
Internal helper. Returns Pearson r via `np.corrcoef`.

#### `create_scatter_with_fit(predictions_df, title, model_name)`
- Scatter points coloured by predicted value (Viridis colorscale)
- **1:1 ideal line** (dashed gold) — perfect calibration benchmark
- **OLS regression fit line** (green) — actual fit with slope/intercept in legend
- **Pearson r annotation** (top-left, styled box)
- Returns `go.Figure` with `plotly_dark` template

#### `create_bland_altman_plot(predictions_df, model_name)`
- x-axis: mean of (actual, predicted)
- y-axis: difference (predicted − actual)
- Horizontal lines: mean bias, +1.96σ LoA, −1.96σ LoA
- Annotations showing exact threshold values

#### `create_multi_model_scatter(training_results, max_cols=3)`
- Subplot grid (`make_subplots`), one panel per model
- Each panel: scatter + 1:1 line + OLS fit + Pearson r annotation
- Shared layout with `plotly_dark`

#### `create_multi_model_timeseries(training_results)`
- All model predictions overlaid on a single time-series chart
- Reference (actual) shown in white, models in palette colours

#### `create_multi_model_metrics_bar(leaderboard)`
- Grouped bar chart, one subplot per metric (RMSE, MAE, R², Pearson r)
- Each bar group: one bar per model

---

### MODIFIED `modules/feature_engineering.py`

#### New: `create_polynomial_features(dataframe, columns, degree)`
- Uses `sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False)`
- Expands only the user-chosen `columns`
- Appends only *new* columns (does not duplicate originals)
- Degree 1 → no-op

#### New: `create_interaction_terms(dataframe, columns)`
- Creates `col_i × col_j` for every pair in `columns`
- Column name format: `col_i_x_col_j`
- Skips duplicate column names

#### Modified: `create_time_features(dataframe, timestamp_column, config=None)`
Extended with config-controlled flags:

| Flag key | Feature added | Default |
|----------|--------------|---------|
| `hour_of_day` | 0-23 integer | True |
| `day_of_week` | 0=Mon…6=Sun | True |
| `day_of_month` | 1-31 | True |
| `unix_timestamp` | seconds since epoch | False |
| `julian_date` | day-of-year 1-366 | False |
| `calendar_date` | integer days since 1970-01-01 | False |
| `cyclical_hour` | sin + cos of hour/24 | False |
| `cyclical_dow` | sin + cos of DOW/7 | False |
| `cyclical_doy` | sin + cos of DOY/365 | False |
| `season` | DJF/MAM/JJA/SON → 0/1/2/3 | False |
| `day_name` | Mon-Sun numeric + is_weekend | False |

#### Modified: `engineer_features()`
New config keys honoured:
- `polynomial_degree` (int, default 1)
- `polynomial_columns` (list of str, default [])
- `interaction_columns` (list of str, default [])
- `time_feature_flags` (dict of booleans, default {})

---

### MODIFIED `models/train.py`

#### New: `_PARAM_GRIDS` (module-level dict)
Default hyperparameter search spaces:

```python
random_forest:
  n_estimators: [50, 100, 200, 300, 500]
  max_depth: [3, 5, 8, 10, 15, None]
  min_samples_split: [2, 5, 10]
  min_samples_leaf: [1, 2, 4]
  max_features: ["sqrt", "log2", 0.5, 0.8]

xgboost:
  n_estimators: [50, 100, 200, 300]
  max_depth: [3, 4, 5, 6, 8]
  learning_rate: [0.01, 0.05, 0.1, 0.2]
  subsample: [0.7, 0.8, 0.9, 1.0]
  colsample_bytree: [0.7, 0.8, 0.9, 1.0]
  reg_alpha: [0.0, 0.01, 0.1, 1.0]
  reg_lambda: [0.5, 1.0, 2.0]

ridge:
  alpha: [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

lasso:
  alpha: [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
```

#### Modified: `TrainingResult` dataclass
- Added `best_params: Optional[Dict[str, Any]] = None`

#### New: `tune_hyperparameters(model_name, model, x_train, y_train, n_iter, cv_folds, random_state)`
- Uses `RandomizedSearchCV(estimator, param_grid, n_iter, cv=TimeSeriesSplit, scoring="neg_root_mean_squared_error", n_jobs=-1)`
- Returns `(best_estimator, best_params_dict)`
- Models with no param grid (e.g., LinearRegression) → returns model unchanged

#### Modified: `prepare_training_matrices()`
- Added `feature_subset: Optional[List[str]]` parameter
- Filters feature columns to the subset if provided

#### Modified: `train_models()`
- Added `feature_subset: Optional[List[str]]` parameter (passed to `prepare_training_matrices`)
- Reads `config["training"]["tuning"]` dict for per-model tuning config:
  ```yaml
  tuning:
    random_forest:
      enabled: true
      n_iter: 20
    xgboost:
      enabled: false
      n_iter: 10
  ```
- If `tuning[model_name]["enabled"]` is True → calls `tune_hyperparameters()` instead of `.fit()`
- Stores `best_params` in `TrainingResult`

---

### MODIFIED `pipeline/run_pipeline.py`

#### Modified: `run_preprocessing_stage()`
- Reads `config["preprocessing"]["apply_to_reference"]` (bool, default False)
- If False: creates `ref_prep_cfg` with `outlier_method = "none"` for reference
- If True: uses same preprocessing config for both datasets
- Missing-value imputation always applied to both

#### Modified: `run_modeling_stage()`
- Added `feature_subset: list | None = None` parameter
- Passes `feature_subset` through to `train_models()`

---

### MODIFIED `ui/app.py`

#### Session State (new keys)
- `featured_preview`: `pd.DataFrame | None` — result of feature preview
- `selected_features`: `List[str] | None` — user-selected feature subset
- `readme_content`: `str | None` — editable README content

#### STEPS (added)
- `"📖 README"` as 9th step

#### `cached_modeling()` signature change
```python
# Before
cached_modeling(merged_df, cfg_text)
# After
cached_modeling(merged_df, cfg_text, feature_subset_json="null")
```

#### Step 2 — Preprocessing (additions)
- Renamed expander to "Sensor Data Preprocessing"
- New expander: "Reference Data Preprocessing"
  - Checkbox: "Apply outlier removal to reference data" (default: off)
  - Contextual info pill showing active method if enabled
  - Green success message when off (recommended default)

#### Step 5 — Modelling (additions)
Six new/modified expanders:

| Expander | Content |
|----------|---------|
| ⚙️ Lag & Rolling Features | (unchanged from v2) |
| ⚙️ Polynomial & Interaction Features | degree selectbox [1/2/3] + poly_cols multiselect + interaction_cols multiselect |
| ⚙️ Time & Date Features | enable toggle + 8 optional feature checkboxes in 3 columns |
| ⚙️ Training Settings | (unchanged from v2) |
| ⚙️ Manual Hyperparameters | (unchanged from v2) |
| 🔧 Auto-Tuning | per-model enable checkbox + n_iter number_input |

New "Variable Selection" section:
- "Preview Features" button → runs `engineer_features()`, stores result in session state
- Shows feature count pill + collapsible preview table
- Multiselect of all available columns (empty = use all)

Modified training button:
- Passes `subset_json = json.dumps(st.session_state.selected_features)` to `cached_modeling()`
- Shows "Best Hyperparameters Found" expander if any model was tuned

#### Step 6 — Results (additions)
Changed from flat layout to 4-tab layout:

| Tab | Content |
|-----|---------|
| 📏 Metrics | Metric cards (RMSE/MAE/R²/MAPE + Bias/Pearson r/Slope/Intercept) + Full leaderboard |
| 🔍 Explainability | Feature importance bar / coefficients bar (unchanged) |
| 📊 Scatter | `create_scatter_with_fit()` for selected model |
| 🔀 Multi-Model | Sub-tabs: Scatter Grid / Time-Series Overlay / Metrics Bar |

#### Step 7 — Post-Analysis (additions)
- Added Bland-Altman checkbox in Drift Settings expander
- Tab "Pred vs Actual" now uses `create_scatter_with_fit()` instead of basic scatter
- Optional 6th tab "⚖️ Bland-Altman" when checkbox is on
- `show_ba` variable controls tab visibility dynamically

#### Step 9 — README (new)
`render_readme()` function:
- Initialises from `PROJECT_ROOT/README.md` if available, else from `_DEFAULT_README` constant
- Split-pane layout (col1=edit, col2=preview)
- `st.text_area` (600px height) for editing — persists in `st.session_state.readme_content`
- Download button: "💾 Download as Markdown" → `calibration_lab_notes.md`
- Reset button: clears `readme_content` from session state and reruns

`_DEFAULT_README` constant includes:
- Quick-start guide
- Feature engineering options table
- Calibration metrics table
- Tips & notes
- Empty "My Notes" section

---

## New in `config/default.yaml` (conceptual additions)
These keys are not in the YAML file but are injected by the UI and stored in session config:

```yaml
preprocessing:
  apply_to_reference: false   # NEW

feature_engineering:
  polynomial_degree: 1        # NEW
  polynomial_columns: []      # NEW
  interaction_columns: []     # NEW
  time_feature_flags:         # NEW
    unix_timestamp: false
    julian_date: false
    calendar_date: false
    cyclical_hour: false
    cyclical_dow: false
    cyclical_doy: false
    season: false
    day_name: false

training:
  tuning:                     # NEW
    random_forest:
      enabled: false
      n_iter: 10
    xgboost:
      enabled: false
      n_iter: 10
    ridge:
      enabled: false
      n_iter: 10
    lasso:
      enabled: false
      n_iter: 10
```

---

## Backward Compatibility
- All new config keys are opt-in with safe defaults
- App works identically to v2.0 if no new features are enabled
- `cached_modeling()` default `feature_subset_json="null"` preserves old cache behaviour

---

## Verification
- All 5 modified/new files pass `ast.parse()` syntax check ✅
- All new modules import successfully ✅  
- `streamlit run ui/app.py` launches on `http://localhost:8501` ✅
