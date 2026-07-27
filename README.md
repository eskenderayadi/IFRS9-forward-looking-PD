# Forward-Looking Credit Risk (IFRS 9) — DR Projection with Macroeconomic Scenarios

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Code style: Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()

This repository contains a **forward-looking (prospective)** credit-risk project aligned with **IFRS 9** principles.  
It demonstrates how to integrate **macroeconomic scenarios** (GDP growth, unemployment, interest rates, inflation, credit spread, and their lags) into **Probability of Default (PD)** / **Default Rate (DR)** projections.

The project is designed as a practical template for:
- building scenario-driven credit-risk forecasts,
- testing several statistical / machine-learning approaches,
- comparing model performance on historical data,
- and presenting results through a simple **Streamlit** interface.

> **Used Models**: **OLS** (with automatic subset selection), **Beta regression**, and **Decision Tree** (two-step importance selection).

> **Best Model**: **Decision Tree** with an **Adjusted R² of 0.943** on the training set.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Architecture & Design](#architecture--design)
- [Methodology](#methodology)
- [Data Requirements](#data-requirements)
- [Technical Stack](#technical-stack)
- [How to Run the Project](#how-to-run-the-project)
- [How the Streamlit App Works](#how-the-streamlit-app-works)
- [Model Details](#model-details)
- [Outputs and Expected Artifacts](#outputs-and-expected-artifacts)
- [Model Evaluation](#model-evaluation)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Reproducibility Notes](#reproducibility-notes)
- [Development & Contribution](#development--contribution)
- [Performance & Scaling](#performance--scaling)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Model Card](#model-card)

---

## Project Overview

The goal of this project is to project credit default rates using a **forward-looking framework** inspired by **IFRS 9 expected credit loss** methodology. Rather than relying only on historical averages, this approach integrates **macroeconomic drivers** to generate **scenario-based projections** that reflect current and expected economic conditions.

In practice, the workflow is:

1. **Collect** historical default / DR data and macroeconomic indicators,
2. **Align** both datasets on a common time axis,
3. **Engineer** lagged and transformed features,
4. **Train** candidate models (OLS, Beta, Decision Tree),
5. **Compare** their predictive performance,
6. **Generate** scenario-based projections for future periods,
7. **Validate** and document results for governance and reporting.

### Why Forward-Looking?

Under **IFRS 9**, credit-risk measurements should reflect **reasonable and supportable forward-looking information**. This means:
- Projections should not be purely backward-looking.
- They should incorporate expected changes in macroeconomic conditions.
- Multiple scenarios should be considered (baseline, adverse, optimistic).
- Results should be explainable to auditors and regulators.

---

## Business Context

Under **IFRS 9**, credit-risk measurements should reflect **reasonable and supportable forward-looking information**. This means that projections should not be purely backward-looking; they should incorporate expected changes in macroeconomic conditions.

This repository demonstrates a simplified but realistic version of that idea by linking Default Rate (DR) dynamics to macroeconomic drivers such as:

- **GDP growth** — overall economic activity and employment capacity,
- **Unemployment rate** — household income stability and defaults,
- **Policy rate** — cost of credit and refinancing ability,
- **Inflation (CPI)** — purchasing power and pricing dynamics,
- **Credit spread** — risk premium and borrowing costs,
- **Lagged versions** of the above — momentum and delayed effects.

The result is a **transparent** and **explainable** framework that can be adapted to:
- Different portfolios (corporate, retail, mortgage),
- Rating segments (AAA, BBB, high-yield),
- Geographic regions or sectors,
- Internal risk policies and governance requirements.

---

## Key Features

- **Scenario-based forecasting**: Baseline, adverse, and optimistic macro paths with customizable assumptions.
- **Multiple model families**: 
  - **Linear (OLS)** with automatic subset selection and p-value filtering,
  - **Bounded-response (Beta regression)** for 0–1 bounded targets,
  - **Tree-based (Decision Tree)** with two-step variable importance selection.
- **Feature engineering**: Lagging, scaling, transformation of macroeconomic inputs; automatic lag detection.
- **Backtesting support**: Evaluate forecasts against historical observations; in-sample and out-of-sample metrics.
- **Interactive dashboard**: Explore assumptions, adjust scenarios, visualize projected DR paths in real time.
- **Exportable results**: Save projected outputs, model coefficients, and diagnostics for downstream analysis.
- **Gaussian link transformation**: Convert model outputs (historic_z) to DR proxy via inverse-normal mapping.
- **Automatic model selection**: OLS with p-value filtering and adjusted R² ranking.
- **Robustness checks**: Cross-validate on time-series splits; check residual stability.

---

## Repository Structure

```
IFRS9-DR-PROJECTION/
├── README.md                    # This file — comprehensive documentation
├── MODEL_CARD.md                # Detailed model documentation and governance
├── LICENSE                      # MIT license
├── pyproject.toml               # Project metadata and dependencies (uv)
├── uv.lock                      # Pinned dependencies (uv lock file)
│
├── main.ipynb                   # Main Jupyter notebook for exploration, training, validation
├── streamlit_app.py             # Interactive Streamlit dashboard
│
├── notebook data/               # Data directory for notebook-based analysis
│   └── (place your training data here)
│
└── streamlit data/              # Data directory for Streamlit app
    └── (place your app data here)
```

### Future Structure (Recommended)

As the project grows, consider organizing code into modules:

```
IFRS9-DR-PROJECTION/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loading and validation
│   │   ├── preprocessor.py      # Cleaning, alignment, imputation
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ols_model.py          # OLS with subset selection
│   │   ├── beta_model.py         # Beta regression
│   │   ├── tree_model.py         # Decision tree with importance selection
│   │   └── base.py               # Abstract model class
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics (MAE, RMSE, R², calibration)
│   │   ├── plotting.py           # Visualization helpers
│   │   └── config.py             # Global configuration
│   └── validation/
│       ├── __init__.py
│       ├── backtesting.py        # Time-series cross-validation
│       └── diagnostics.py        # Residual analysis, stability checks
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_metrics.py
├── artifacts/                    # Model exports, scalers, checkpoints
│   └── (generated at runtime)
├── data/                         # Input datasets
│   ├── macroeconomic_indicators.xlsx
│   └── historical_pd.xlsx
└── notebooks/
    └── exploration.ipynb
```

---

## Architecture & Design

### Data Flow

```
Raw Data (Excel/CSV)
    ↓
[Data Loader] → Validation & Type Checking
    ↓
[Preprocessor] → Alignment, Imputation, Lag Generation
    ↓
[Feature Engineering] → Scaling, Transformation
    ↓
[Train/Test Split] (chronological)
    ↓
┌─────────────────────────────────────┐
│  Model Selection & Training        │
├─────────────────────────────────────┤
│ • OLS (with automatic subset)       │
│ • Beta Regression (rescaled Y)      │
│ • Decision Tree (Top-5 importance)  │
└─────────────────────────────────────┘
    ↓
[Evaluation] → R², MAE, RMSE, Calibration
    ↓
[Projection] → Apply to future scenarios
    ↓
[Visualization] → Streamlit Dashboard
    ↓
[Export] → CSV, JSON, or artifacts
```

### Model Pipeline

Each model follows a standardized pipeline:

1. **Fit**: Train on selected features and target,
2. **Predict (train)**: Generate in-sample predictions,
3. **Predict (future)**: Project on out-of-sample data,
4. **Transform**: Convert `historic_z` → `DR` via Gaussian link,
5. **Evaluate**: Compute metrics, residuals, stability.

---

## Methodology

### 1) Data Preparation

The model starts by preparing historical macroeconomic and credit-risk series:

- **Handling missing values** → interpolation, forward-fill, or drop (configurable),
- **Aligning frequencies** → resample monthly, quarterly, or annual data to common granularity,
- **Lag generation** → create L1, L2, L3, ... versions of macro variables,
- **Scaling** → standardize numeric inputs (optional StandardScaler),
- **Feature engineering** → compute changes, spreads, rolling averages, trend indicators,
- **Look-ahead bias prevention** → shift macro variables forward to avoid leakage.

### 2) Estimation

Several candidate models are fitted to the prepared dataset:

#### OLS with Automatic Subset Selection

- **Candidate pool**: All numeric features (excluding target and DR),
- **Algorithm**: Enumerate all subsets (2^k combinations),
- **Filter**: Retain subsets with **all p-values ≤ 0.05**,
- **Selection**: Choose highest **Adjusted R²** among filtered subsets,
- **Fallback**: If no subset passes p-value filter, fit full model,
- **Output**: Coefficients, p-values, Adjusted R², fitted predictions.

**Rationale**: Automatic subset selection reduces overfitting and identifies only statistically significant drivers. Manual alternative: use domain expertise or penalized regression (LASSO/Ridge).

#### Beta Regression

- **Data prep**: Rescale target Y to (0, 1) using min/max normalization with ε-clipping,
- **Model**: GLM with **Beta family** (if available) or standalone **betareg** module,
- **Features**: Same as OLS-selected subset (for consistency),
- **Bounded output**: Naturally constrains predictions to [0, 1] → mimics DR bounds,
- **Back-transform**: Rescale predictions back to original Y scale,
- **Metric**: R² and Adjusted R² on back-transformed predictions.

**Rationale**: Default Rates are naturally bounded [0, 1]. Beta regression respects this constraint, improving calibration.

#### Decision Tree (Two-Step)

1. **Shallow tree (exploration)**: 
   - Fit DecisionTree with `max_depth=3` on **all** candidate features,
   - Extract feature importances (Gini / variance reduction),
2. **Top-5 selection**: 
   - Retain the **5 most important** features,
   - If fewer than 5 features have positive importance, keep all or default to top available,
3. **Compact tree (final model)**:
   - Fit final DecisionTree (`max_depth=3`) on Top-5 only,
   - Extract final importances, predictions, and R² (train).

**Rationale**: Reduces overfitting by focusing on key interactions; interpretable through feature importance and tree structure.

### 3) Scenario Generation

Future macroeconomic paths are provided under different scenarios:

- **Baseline**: Central economic view (e.g., IMF or central bank forecast),
- **Adverse**: Stressed / deteriorating conditions (e.g., recession, unemployment spike),
- **Optimistic**: Favorable conditions (e.g., strong recovery, rate cuts).

Macro variables are propagated through each fitted model to obtain projected DR values. Scenarios can be:
- **Predefined** (loaded from files or config),
- **Ad-hoc** (user adjusts sliders in Streamlit dashboard),
- **Probabilistic** (draw from distributions, then average).

### 4) Validation & Backtesting

Model performance is assessed using multiple approaches:

- **In-sample metrics**: Adjusted R², MAE, RMSE on training data,
- **Out-of-sample evaluation** (if future data available): Predict on hold-out periods,
- **Time-series cross-validation**: Rolling / expanding windows to account for time dependency,
- **Directional accuracy**: Pct of time model correctly predicts direction (increase/decrease),
- **Stability checks**: PSI (Population Stability Index), CSI (Characteristic Stability Index) across time,
- **Residual diagnostics**: Plot residuals, check normality (Shapiro-Wilk), autocorrelation (Durbin-Watson).

### 5) Decisioning & Reporting

Resulting projections support:

- **Expected Credit Loss (ECL)** calculations under multiple scenarios,
- **Sensitivity analysis** → how do results change with 1% GDP shock?
- **Scenario comparison** → which scenario poses highest risk?
- **Portfolio monitoring** → track vs. forecast; flag deviations,
- **Management reporting** → Board, audit, and regulator communications.

---

## Data Requirements

The repository expects two broad data categories:

### 1) Macroeconomic Indicators

**Example file**: `data/macroeconomic_indicators.xlsx`

**Possible columns**:
- `date` (or `Period`, `Trimestre`, `Quarter`) — date index,
- `gdp_growth` — YoY or QoQ growth rate (%),
- `unemployment_rate` — unemployment rate (%),
- `policy_rate` — central bank policy rate (%),
- `cpi_inflation` — year-on-year inflation (%),
- `credit_spread` — sovereign or corporate spread (bps),
- Additional: employment, real estate indices, commodity prices, etc.

**Frequency**: Monthly, quarterly, or annual. Recommend **quarterly** for default rate alignment.

**Data quality**:
- No extreme outliers (check for data entry errors),
- Sufficient history (minimum 3–5 years recommended),
- Current and forecast versions (for scenario projection).

### 2) Historical Default / DR Data

**Example file**: `data/historical_pd.xlsx`

**Possible columns**:
- `date` (or `Period`, `Trimestre`) — date index,
- `segment` (optional) — portfolio, rating, or product segment,
- `pd_obs` or `default_rate_obs` — observed PD / DR (0–1),
- `historic_z` — latent or intermediate credit risk metric (not strictly 0–1),
- `volume` — count of obligors in segment (optional).

**Frequency**: Align with macro indicators (typically quarterly).

**Data quality**:
- Ensure consistent segment definitions over time,
- Handle portfolio transitions (new obligors, maturities),
- Validate against source systems (core data warehouse, rating model outputs).

### Important Notes

- **Ensure time frequency is consistent** or properly converted (use resampling if needed),
- **Check for look-ahead bias** → shift macro variables forward by 1+ periods to avoid leakage,
- **If multiple segments exist** → align data by portfolio, vintage, or rating band,
- **Validate data types** before training (numeric vs. categorical, date parsing),
- **Document data lineage** → source, extraction query, refresh frequency, assumptions.

### Example Data Format

```csv
Trimestre,gdp_growth,unemployment_rate,policy_rate,cpi_inflation,credit_spread,historic_z,DR
2022-Q1,2.5,3.8,0.25,1.8,50,0.15,0.08
2022-Q2,1.8,3.9,0.5,2.3,60,0.18,0.09
2022-Q3,1.2,3.7,1.5,3.1,75,0.22,0.11
2022-Q4,0.5,3.6,2.0,3.5,100,0.28,0.14
2023-Q1,-0.2,4.1,2.5,4.0,120,0.35,0.18
...
```

---

## Technical Stack

### Core Dependencies

- **Python 3.11+** — language runtime,
- **pandas 3.0.3+** — data manipulation and alignment,
- **numpy 2.4.6+** — numerical operations,
- **scipy 1.17.1+** — statistical functions (norm.ppf, norm.cdf),
- **scikit-learn 1.9.0+** — DecisionTreeRegressor, preprocessing,
- **statsmodels 0.14.6+** — OLS, GLM (Beta family), diagnostics,
- **streamlit 1.59.0+** — interactive dashboard,
- **plotly 6.8.0+** — interactive charts.

### Optional / Advanced

- **jupyter** — for notebook-based exploration,
- **openpyxl** — for reading `.xlsx` files (included via pandas),
- **pytest** — for unit tests,
- **black**, **flake8** — for code formatting and linting,
- **mypy** — for type checking.

### Environment Management

- **uv** — fast Python package manager (recommended),
- Alternative: `pip` with `requirements.txt` or `poetry`.

---

## How to Run the Project

### Prerequisites

- **Python 3.11+** installed,
- **uv** installed (or equivalent: `pip`, `poetry`, `conda`),
- Access to required input data files (CSV/XLSX format).

### Setup

#### Option 1: Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/eskenderayadi/IFRS9-DR-PROJECTION.git
cd IFRS9-DR-PROJECTION

# Sync dependencies (creates .venv)
uv sync

# Activate virtual environment
source .venv/bin/activate          # macOS / Linux
# or
.venv\Scripts\activate             # Windows

# Place input data in ./streamlit\ data/ or ./notebook\ data/
```

#### Option 2: Using `pip`

```bash
# Clone repository
git clone https://github.com/eskenderayadi/IFRS9-DR-PROJECTION.git
cd IFRS9-DR-PROJECTION

# Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# or
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt   # (create from pyproject.toml if needed)
```

### Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

This opens a browser window at `http://localhost:8501/`.

**Typical user flow**:
1. Upload **training data** (CSV/XLSX with macro + DR variables),
2. Optionally upload **future data** for projections,
3. Select features (X) and target (Y, default: `historic_z`),
4. Choose models (OLS, Beta, Decision Tree),
5. Click **"Lancer l'estimation"** (Launch estimation),
6. Inspect results, plots, and exported files.

### Notebook Workflow

For step-by-step analysis and exploration:

```bash
jupyter notebook main.ipynb
```

Then:
1. Open `main.ipynb`,
2. Run cells in order,
3. Inspect preprocessing outputs, model diagnostics, feature importance,
4. Save final artifacts to `artifacts/` (model.joblib, scaler.joblib, etc.),
5. Later, launch Streamlit app to visualize interactively.

### Configuration

Edit these sections in **`streamlit_app.py`** to customize:

```python
# Default target variable
target_var = st.sidebar.selectbox("Variable cible (Y)", num_cols,
                                   index=num_cols.index("historic_z") if "historic_z" in num_cols else 0)

# Default models to run
methods = st.sidebar.multiselect("Modèles",
                                  ["OLS", "BetaRegression", "DecisionTree"],
                                  default=["OLS", "BetaRegression", "DecisionTree"])

# Enable/disable automatic OLS subset selection
auto = st.sidebar.checkbox("Sous-ensemble OLS automatique", True)

# Default mean_abs_DR (if not in data)
mean_abs_DR = st.sidebar.number_input("mean_abs_DR (si DR absent)", value=0.05, step=0.001)
```

---

## How the Streamlit App Works

The **Streamlit application** is a lightweight interactive risk dashboard. It allows you to:

### 1) Data Upload
- **Training data** (required): CSV or XLSX with macro indicators, target, and optional DR,
- **Future data** (optional): Same columns for out-of-sample projection.

### 2) Variable Selection
- **Target (Y)**: Default to `historic_z`; change via sidebar,
- **Features (X)**: Multi-select from numeric columns,
- **Exclude**: Target, DR (automatically excluded).

### 3) Model Selection
- **OLS**: With or without automatic subset selection,
- **Beta Regression**: If statsmodels BetaFamily available (gracefully skipped if not),
- **Decision Tree**: Two-step importance selection (explore all → Top-5 → compact tree).

### 4) Training & Inference
- Click **"Lancer l'estimation"** to fit all selected models,
- Displays coefficient tables, p-values, Adjusted R², feature importances,
- Generates predictions (train + future if available).

### 5) Time Axis Selection
- Choose a column (e.g., `Trimestre`, `Quarter`, `date`) as x-axis,
- Or use row index `(index)`.

### 6) Visualization (4 Charts)
- **In-sample historic_z**: Real vs. predicted (OLS, Beta, Tree),
- **Out-of-sample historic_z** (if future data provided),
- **In-sample DR**: Real vs. predicted (transformed via Gaussian link),
- **Out-of-sample DR** (if future data provided).

### 7) Export & Download
- Charts are interactive (hover, zoom, pan, download as PNG),
- Models are printed to console / app log,
- Consider adding CSV export of predictions (future enhancement).

### User Experience Flow

```
Launch Streamlit
    ↓
[1] Upload training file (CSV/XLSX)
    ↓
[2] (Optional) Upload future file
    ↓
[3] View data preview
    ↓
[4] Select features (X) and target (Y)
    ↓
[5] Select models (OLS/Beta/Tree)
    ↓
[6] Choose time axis column
    ↓
[7] Click "Lancer l'estimation"
    ↓
[8] View results:
    • Model coefficients & p-values
    • Feature importances (Tree)
    • R² metrics
    ↓
[9] Inspect 4 interactive charts
    ↓
[10] Adjust parameters & re-run
    ↓
[11] Export results (download, screenshot, or manual copy)
```

---

## Model Details

### OLS with Automatic Subset Selection

**Objective**: Find the minimal set of significant predictors (highest Adjusted R²).

**Algorithm**:
```
for k in range(1, num_features + 1):
    for each combination of k features:
        fit OLS(X_subset, y)
        if all p-values <= 0.05:
            candidate = (Adjusted R², X_subset, model)
            if better than previous best:
                best = candidate
if best found:
    return best model & selected features
else:
    return full OLS model
```

**Pros**:
- Automatic variable selection,
- Removes insignificant variables,
- Interpretable coefficients,
- Good for small feature sets (<15 features).

**Cons**:
- Computationally expensive for large feature sets (2^k subsets),
- Risk of data snooping / multiple testing bias,
- Unstable on high-correlation features.

**Hyperparameters** (in code):
- P-value threshold: `0.05` (hardcoded; consider making configurable),
- Metric: `Adjusted R²` (penalizes added variables).

### Beta Regression

**Objective**: Model bounded responses (0, 1) with Beta distribution.

**Data prep**:
```python
y_min, y_max = y_train.min(), y_train.max()
y_beta = ((y_train - y_min) / (y_max - y_min)).clip(eps, 1 - eps)
```

**Model**: GLM with Beta family (if available) or standalone betareg module.

**Back-transform**:
```python
y_pred = y_beta_scaled * (y_max - y_min) + y_min
```

**Pros**:
- Respects 0–1 bounds on predictions,
- Better calibration for rates/proportions,
- Accounts for heteroskedasticity.

**Cons**:
- Requires sufficient dispersion in y,
- Statsmodels support may be incomplete; fallback library required,
- More complex interpretation.

**Output metrics**: R² and Adjusted R² on back-transformed predictions.

### Decision Tree (Two-Step)

**Step 1: Exploration tree** (all features)
```python
dt_full = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
dt_full.fit(X_train[all_features], y_train)
importances = dt_full.feature_importances_
```

**Step 2: Top-5 selection**
```python
vars_dt = importances.head(5).index.tolist()
```

**Step 3: Compact tree** (Top-5 only)
```python
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
dt_model.fit(X_train[vars_dt], y_train)
r2_dt = r2_score(y_train, dt_model.predict(X_train[vars_dt]))
```

**Pros**:
- Captures non-linear relationships,
- Automatic interaction detection,
- Feature importance directly interpretable,
- Robust to scaling.

**Cons**:
- Prone to overfitting; limited depth and sample size constraints,
- Can be unstable (small data changes → large tree changes),
- Not as precise as ensemble methods (RF, GBM).

**Hyperparameters** (configurable):
- `max_depth=3` — limits complexity,
- `min_samples_split=2` — minimum samples per split,
- Consider: `min_samples_leaf`, `criterion` (mse, friedmand_mse, mae).

### DR Transformation (Gaussian Link)

**Purpose**: Convert model output (`historic_z`) to a Default Rate proxy (`DR`).

**Mapping**:
```python
# Compute constant from training data
m = mean_abs(DR_train)                # unconditional DR level
c = norm.ppf(m)                       # inverse-normal CDF

# Apply Gaussian link
DR = norm.cdf(c - historic_z)
```

**Rationale**:
- Converts unbounded `historic_z` to bounded [0, 1] `DR`,
- Inverse relationship: higher `historic_z` → lower `DR` (risk),
- Centers around empirical mean DR,
- Standard approach in credit risk modeling.

**Example**:
```
If m = 0.05 (5% unconditional DR):
  c = norm.ppf(0.05) ≈ -1.645
  
If historic_z = 0:
  DR = norm.cdf(-1.645 - 0) = norm.cdf(-1.645) ≈ 0.05 ✓
  
If historic_z = -1 (better conditions):
  DR = norm.cdf(-1.645 + 1) = norm.cdf(-0.645) ≈ 0.26 (higher DR)
  
If historic_z = +1 (worse conditions):
  DR = norm.cdf(-1.645 - 1) = norm.cdf(-2.645) ≈ 0.004 (lower DR) ✓
```

---

## Outputs and Expected Artifacts

Depending on how you structure the project, you may produce:

### Model Artifacts
- `ols_model.joblib` — fitted OLS estimator + scaler,
- `beta_model.joblib` — fitted Beta GLM,
- `dt_model.joblib` — fitted DecisionTree,
- `scaler.joblib` — StandardScaler for feature normalization.

### Data Artifacts
- `X_train_selected.csv` — training features (after subset selection),
- `y_train_transformed.csv` — target after any transformation,
- `predictions_train.csv` — model predictions on training set,
- `predictions_future.csv` — out-of-sample projections.

### Evaluation Artifacts
- `metrics_summary.json` — R², Adjusted R², MAE, RMSE, calibration metrics,
- `residual_diagnostics.csv` — residuals, standardized residuals, diagnostics,
- `feature_importance.csv` — from OLS (coefficients) and Tree (Gini/variance reduction).

### Visualization Artifacts
- `in_sample_historic_z.html` — interactive Plotly chart,
- `out_of_sample_historic_z.html`,
- `in_sample_dr.html`,
- `out_of_sample_dr.html`,
- `residual_plots.png` — matplotlib QQ plot, ACF, etc.

### Governance Artifacts
- `MODEL_CARD.md` — detailed model documentation (already in repo),
- `DATA_CARD.md` — data lineage, sources, validation,
- `CHANGELOG.md` — version history, changes, retraining events,
- `validation_report.md` — backtesting results, stability checks.

A good practice is to:
- Keep generated artifacts separated from source code,
- Store in `artifacts/` or `outputs/` directory,
- Document naming conventions for each file type,
- Version artifacts with timestamps or git hashes.

---

## Model Evaluation

The current repository reports:

- **OLS**: Automatic subset selection, Adjusted R² metric,
- **Beta Regression**: Adjusted R² after back-transformation,
- **Decision Tree**: R² on train set, feature importances (Top-5).

### Recommended Extended Metrics

For a more complete assessment, consider reporting:

#### 1) Train / Validation / Test Decomposition
```
Train Set: 70% of data (fit models)
Validation Set: 15% of data (tune hyperparameters, select models)
Test Set: 15% of data (final evaluation, unbiased estimate)
```

#### 2) Error Metrics
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAE = mean_absolute_error(y_true, y_pred)
RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
R2 = r2_score(y_true, y_pred)
Adj_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)  # p = num features
```

#### 3) Directional Accuracy
```
Pct_correct_direction = mean((y_pred > median) == (y_true > median))
```

#### 4) Calibration & Stability
```
PSI (Population Stability Index) — compare train vs. test distributions
CSI (Characteristic Stability Index) — for each feature separately
Traffic Light Approach: PSI < 0.05 (Green), 0.05-0.1 (Yellow), >0.1 (Red)
```

#### 5) Time-Series Cross-Validation
```
Because credit-risk data is time-dependent:
• Use expanding or rolling windows (not random splits),
• Fit on historical data; validate on next quarter,
• Track metric degradation over forecast horizon.
```

#### 6) Residual Analysis
```
• Normality test (Shapiro-Wilk p-value),
• Autocorrelation (Durbin-Watson, ACF plot),
• Heteroskedasticity (Breusch-Pagan test),
• Outlier detection (studentized residuals > ±3σ).
```

#### 7) Scenario Robustness
```
• Stress test: apply extreme macro shocks,
• Check prediction stability: does model blow up?
• Compare directional forecasts across scenarios.
```

---

## Assumptions and Limitations

This project is a useful prototype, but it still relies on simplifying assumptions:

### Data Assumptions
- **Macroeconomic inputs are available and reliable** — source from official (central bank, IMF, World Bank) or validated providers,
- **Historical defaults are accurately recorded** — data governance and system validation required,
- **Sufficient history** (minimum 3–5 years) to estimate robust parameters,
- **Stationarity** (or manageable non-stationarity) — macro variables do not have permanent regime breaks.

### Modeling Assumptions
- **Linear or tree relationships** between macro variables and DR — true relationships may be more complex,
- **Gaussian link** for DR transformation — empirical link may differ,
- **Homoskedasticity** (equal error variance) — OLS assumes this; may not hold,
- **No multicollinearity** — high correlations between predictors bias estimates,
- **No omitted variables** — missing key drivers biases forecasts.

### Model Performance Assumptions
- **Performance may vary** across portfolios and time periods (e.g., retail vs. corporate, pre/post-crisis),
- **A single model may not generalize equally** across all economic regimes,
- **Out-of-sample performance may degrade** as economic conditions shift.

### Governance & Production Use
- **Current implementation is NOT validated** for production use without:
  - Governance controls (approval, documentation, audit),
  - Model-risk oversight (validation, ongoing monitoring, challenger models),
  - Strong data lineage and audit trail,
  - Explainability checks (feature importance, sensibility of coefficients),
  - Independent third-party validation.
- **IFRS 9 implementations typically require**:
  - Documented expert judgment (when to override model),
  - Scenario weighting (how to combine baseline/adverse/optimistic into ECL),
  - Governance policy (model governance, escalation, backtesting frequency),
  - Regulatory approval (if used for capital or public reporting).

### Practical Limitations
- **Small dataset risk** — parameter estimates unstable,
- **Structural breaks** — model fails if regime fundamentally changes,
- **Correlation vs. causation** — macro variables may be proxies, not true drivers,
- **Forecasting horizon** — model may be valid 1 quarter ahead but invalid 3 years out,
- **Scenario specification** — expert judgment required; wrong scenarios → wrong projections.

---

## Reproducibility Notes

To reproduce the project reliably:

### Code & Environment
- **Pin package versions** in `pyproject.toml` (done via `uv.lock`),
- **Keep notebooks & scripts** under version control (Git),
- **Set random seeds** for reproducibility (in `main.ipynb` and `streamlit_app.py`):
  ```python
  import numpy as np
  np.random.seed(42)
  ```

### Data & Configuration
- **Document data sources** and refresh dates (e.g., "IMF WEO Oct 2025"),
- **Store preprocessing assumptions** alongside artifacts (e.g., "lag = 1, dropped Q1 2020 outlier"),
- **Keep scenario definitions explicit** and versioned (baseline.json, adverse.json, etc.),
- **Record exact notebook / script** used to generate results (e.g., "main.ipynb commit abc123"),

### Version Control Best Practices
```bash
git log --oneline --all
git tag -a v1.0.0 -m "Baseline model release"
git show v1.0.0:main.ipynb  # retrieve exact version
```

### Enhanced Reproducibility

For even stronger reproducibility, consider adding:

#### 1) `requirements.txt` or lock file
```bash
uv pip freeze > requirements.txt
```

#### 2) Centralized configuration (`config.py`)
```python
# config.py
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.7
OLS_P_VALUE_THRESHOLD = 0.05
TREE_MAX_DEPTH = 3
LAG_PERIODS = [1, 2, 3]
MACRO_VARIABLES = ["gdp_growth", "unemployment_rate", ...]
TARGET_VARIABLE = "historic_z"
```

#### 3) Dedicated training script
```bash
python -m src.train \
  --data-file data/historical_data.xlsx \
  --config config.py \
  --output-dir artifacts/v1.0.0
```

#### 4) Unit tests
```bash
pytest tests/ -v
```

#### 5) Continuous integration (GitHub Actions / GitLab CI)
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

---

## Development & Contribution

### Code Style

- **Language**: Python 3.11+,
- **Format**: Follow PEP 8 with line length 100,
- **Linter**: `flake8` (optional),
- **Formatter**: `black` (optional).

```bash
black --line-length=100 src/
flake8 src/ --max-line-length=100
```

### Testing

Implement unit tests for:
- Data loading & preprocessing,
- Model training & prediction,
- Metric calculations,
- Feature engineering functions.

```bash
pytest tests/ -v --cov=src
```

### Adding Features

1. **Create a branch**: `git checkout -b feature/my-feature`,
2. **Implement & test**: Add code, tests, and documentation,
3. **Pull request**: Describe changes, request review,
4. **Merge**: After approval, merge to `main`.

### Contribution Guidelines

- **Report bugs** via GitHub Issues with reproducible examples,
- **Suggest features** with use-case rationale,
- **Submit PRs** with clear commit messages and tests,
- **Update documentation** (README, MODEL_CARD) for significant changes.

---

## Performance & Scaling

### Computational Complexity

| Model | Training | Prediction | Notes |
|-------|----------|-----------|-------|
| **OLS** | O(n * 2^k) | O(k) | Subset enumeration expensive for k > 15 |
| **Beta** | O(n * k^2) | O(k) | Iterative optimizer (fewer iterations than OLS) |
| **Tree** | O(n * k * log n) | O(log n) | Tree traversal; parallelizable |

### Memory Usage
- **Typical dataset** (1000 rows, 50 features): ~10 MB,
- **Large dataset** (1M rows, 50 features): ~2.5 GB,
- **Model artifacts**: 1–10 MB (joblib serialization).

### Optimization Tips

1. **Feature selection upstream**: Reduce k before subset enumeration,
2. **Parallel subset enumeration**: Use `multiprocessing` for OLS,
3. **Streaming inference**: For real-time scoring, pre-load models and batch predictions,
4. **Caching**: Store preprocessed data to avoid re-computation,
5. **GPU acceleration** (future): Use CuPy, Rapids for large-scale data.

### Deployment Scenarios

**Scenario 1: Batch Processing** (monthly/quarterly)
```bash
python -m src.train --data data/latest.xlsx --output artifacts/
# Upload predictions to data warehouse
```

**Scenario 2: Real-Time API** (Streamlit + FastAPI)
```bash
streamlit run streamlit_app.py &
python -m fastapi run src/api.py
```

**Scenario 3: Scheduled Job** (Docker + cron)
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "src.train"]
```

---

## FAQ & Troubleshooting

### Q: My data has missing values. What should I do?

**A**: The app drops rows with NAs in selected features. Alternatives:
- Impute using mean, median, forward-fill, or interpolation,
- Use sklearn's `SimpleImputer` or `KNNImputer`,
- Implement in `preprocessor.py` and re-run.

### Q: Can I use categorical variables (e.g., sector)?

**A**: Currently, the app auto-selects numeric columns only. To use categorical:
- One-hot encode them before uploading,
- Extend `streamlit_app.py` to handle categoricals,
- Consider regularized regression (LASSO) to avoid overfitting.

### Q: Why is my OLS subset selection returning the full feature set?

**A**: If no subset passes the p-value filter (all p ≤ 0.05), the app defaults to full OLS. Reasons:
- Many features highly correlated → high p-values,
- Small sample size → estimates unstable,
- **Fix**: Reduce features manually or lower p-value threshold (edit code).

### Q: How do I interpret tree feature importance?

**A**: Importance = fraction of variance reduction attributed to each feature. Higher = more useful. Sum of importances ≈ 1.

### Q: Can I export predictions?

**A**: Currently, predictions are printed to the Streamlit console and plotted. **Enhancement**: Add download button:
```python
predictions_df = pd.DataFrame({...})
st.download_button("Download Predictions", predictions_df.to_csv(index=False), "predictions.csv")
```

### Q: What if I want to use ensemble methods (Random Forest, XGBoost)?

**A**: Extend `src/models/` with new model classes. Example:
```python
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(BaseModel):
    def __init__(self, **hyperparams):
        self.model = RandomForestRegressor(**hyperparams)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
```

### Q: How do I run backtests on multiple time periods?

**A**: Implement time-series cross-validation in `validation/backtesting.py`:
```python
def time_series_cv(df, train_size=0.7, step=1):
    n = len(df)
    for end in range(n - int(n * (1 - train_size)), n, step):
        train = df[:end - int(n * (1 - train_size))]
        test = df[end - int(n * (1 - train_size)):end]
        yield train, test
```

### Q: My model predictions are outside [0, 1]. Why?

**A**: Only Beta regression and the DR transformation guarantee [0, 1] bounds. OLS and raw Decision Tree may exceed bounds. **Fix**:
- Use Beta regression,
- Clip predictions: `np.clip(pred, 0, 1)`,
- Use logistic / probit link (future enhancement).

### Q: How sensitive is my forecast to macro assumptions?

**A**: Conduct sensitivity analysis:
1. Train model on historical data,
2. Create scenarios: base, +1% GDP, -2% unemployment, etc.,
3. Predict under each scenario,
4. Compare results (% change in DR vs. base).

### Q: Can I use this model for individual-level credit decisions?

**A**: **No**. This model is portfolio-level only. It cannot:
- Price individual loans,
- Rank obligor creditworthiness,
- Detect fraud or non-payment intent.

Use with expert judgment for portfolio management only.

---

## License

Released under the **MIT License**. See `LICENSE` for details.

**Summary**:
- You may use, modify, and distribute this software freely,
- You must include a copy of the license,
- The software is provided "as-is" without warranty.

---

## Acknowledgements

This project was prepared as part of the **Nexialog Consulting Challenge**. Special thanks to:

- **Mr. Salem** for the opportunity and mentorship,
- **Nexialog Consulting** for providing business context and data requirements,
- **Open-source community** for pandas, scikit-learn, statsmodels, and Streamlit.

---

## Model Card

See [`MODEL_CARD.md`](MODEL_CARD.md) for detailed documentation, including:

- **Purpose & Scope**: Intended use and limitations,
- **Data & Inputs**: Input format and requirements,
- **Preprocessing**: Feature engineering and transformations,
- **Models**: Algorithm details for OLS, Beta, Decision Tree,
- **Training & Projection**: End-to-end workflow,
- **Metrics**: Evaluation approach,
- **Assumptions & Limitations**: Key caveats,
- **Governance & Documentation**: Best practices,
- **Fairness & Responsible Use**: Ethical considerations,
- **Reproducibility**: Versioning and artifact management,
- **Changelog**: Version history.

---

## Related Resources

- **IFRS 9 Overview**: [IFRS Foundation](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)
- **ECL Methodology**: [EBA IFRS 9 GL](https://www.eba.europa.eu/documents/10180/2020421/EBA+GL+2017+14+GL+on+PD+estimation+comparators+and+LGD+estimation.pdf)
- **Credit Risk Modeling**: Bluhm, Overbeck, Wagner (2003)
- **Macro-Financial Linkages**: IMF GFSR reports

---

**Last Updated**: 2025-08-12  
**Repository Owner**: @eskenderayadi  
**Questions?** Open an issue or contact the repository owner.
