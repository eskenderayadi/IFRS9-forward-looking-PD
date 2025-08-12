# Model Card — Forward-Looking PD Toolkit (OLS • Beta • Decision Tree)

**Project**: Nexialog Challenge — Forward-looking Credit Risk (IFRS 9)  
**Component**: Streamlit app for model comparison and DR projection  
**Version**: 1.0.0 (2025-08-12)  
**Owner**: _Your Name_

---

## 1) Purpose & Scope

This app provides a compact **forward-looking PD modeling toolkit** with three modeling options:
- **OLS** with automatic subset selection (p ≤ 0.05) and best adjusted R²,
- **BetaRegression** (when available) on a rescaled target,
- **Decision Tree** with a two-step variable selection (importance → Top-5) then a compact tree.

It supports **training** on a base dataset and **projection** on a future dataset, with **in/out plots** for both the raw target (`historic_z`) and the **derived PD proxy** (`DR`).

Intended use:
- Exploratory modeling, feature screening, and scenario exploration for credit risk teams.
- Teaching/demo of forward-looking logic where **historic risk factor** → **PD proxy** mapping is shown.

Not intended for:
- Direct production ECL estimation without validation, governance, and integration with enterprise data pipelines.
- Underwriting or individual-level decisions.

---

## 2) Data & Inputs

### 2.1. Datasets
- **Training file** (CSV/XLSX): contains numeric features (X), the target `historic_z`, and optionally `DR`.
- **Future file** (CSV/XLSX, optional): same features for out-of-sample projections.

### 2.2. Feature types
- The app auto-detects **numeric** columns as candidates for X.
- The **target** defaults to `historic_z` (you can change it in the sidebar).

### 2.3. Time axis
- Choose a column (e.g., `Trimestre`) or use the **row index** as the x-axis for plots.

### 2.4. Mean absolute DR
- If `DR` is present in training data, its **mean absolute value** is computed automatically.
- If absent, the user can set **`mean_abs_DR`** in the sidebar (default 0.05).

---

## 3) Preprocessing & Target Engineering

### 3.1. Target transformation to DR
The app converts predicted `historic_z` to a PD proxy `DR` via a **Gaussian link**:

- Let `m = mean_abs_DR` (from training `DR` or user input).
- Compute `c = Φ⁻¹(m)` where `Φ` is the standard normal CDF.
- Map `historic_z` → `DR` as:  
  **DR = Φ(c − historic_z)**

This yields a DR-like measure that increases when `historic_z` decreases (and vice versa), with `m` anchoring the unconditional level.

### 3.2. Missing values
- Rows with missing values in the selected X are dropped during model training.

---

## 4) Models

### 4.1. OLS (with automatic subset selection)
- If **auto** is enabled, the app searches all subsets of candidate X:
  - Keep subsets with **all p-values ≤ 0.05** (excluding constant).
  - Among those, select the **highest adjusted R²**.
  - If none pass the p-value filter, fit OLS on the full X set.
- Reported metric: **adjusted R²** on the training sample.

### 4.2. BetaRegression (optional)
- If Statsmodels **Beta family** or **betareg** back-up is available, the target is rescaled to (0,1) using min/max of `historic_z`:  
  `y_beta = clip((y − min)/(max − min), ε, 1−ε)` with ε=1e−6.
- A GLM with Beta family (or betareg formula) is fitted on the same OLS-selected variables.
- Predictions are **rescaled back** to the original `historic_z` scale.
- Reported metric: R² on train and **adjusted R²**.

> If the BetaRegression dependency is not available, the option is disabled gracefully.

### 4.3. Decision Tree (two-step)
1. Fit a **shallow tree** (max_depth=3) on **all** candidate X to compute importances.  
2. Keep the **Top-5** most important variables.  
3. Fit a **compact tree** (max_depth=3) on these Top-5.  
4. Report **R² (train)** and a table of feature importances.

---

## 5) Training, Projection & Plots

- **Training**: fit on selected X and target (`historic_z`) from the training dataset.
- **Projection** (optional): if a future dataset is uploaded, produce **out-of-sample** predictions for each model.
- **Plots**:
  - `historic_z` **in-sample** (and out-of-sample if future data is provided).
  - `DR` **in-sample** (and out-of-sample), using the Gaussian link described above.

---

## 6) Metrics

- **Adjusted R²** for OLS and BetaRegression (train).
- **R² (train)** for Decision Tree.
- Additional metrics (MAE/RMSE, calibration) can be added in future versions.

> Note: on small datasets or highly collinear X, adjusted R² can be unstable; always complement with cross-validation.

---

## 7) Assumptions & Limitations

- **Distributional assumption**: the DR mapping assumes a **Gaussian link**; if the empirical link is different, calibration may be off.
- **Subset selection bias**: p-value filtering on many subsets risks **data snooping**; mitigate with out-of-sample validation.
- **Tree depth**: shallow trees are interpretable but may **underfit**; ensembles (RF/GBM) can be considered later.
- **Feature scaling**: OLS/Beta may benefit from explicit scaling; currently the app relies on raw numeric inputs.
- **Missing data**: rows with NA in X are dropped; consider imputation to avoid bias.

---

## 8) Governance & Documentation

- Keep a **data card** (sources, extraction queries, timestamps, versions).
- Save **model artifacts** per release (seed, hyperparameters, selected variables).
- Maintain an **evaluation report** with in/out plots, residual diagnostics, and stability checks (PSI/CSI).
- Record **acceptance criteria** (e.g., adjusted R² thresholds, stability constraints).

---

## 9) Fairness & Responsible Use

- Do not include protected attributes or proxies in X.
- Check **segment-level** behavior (e.g., by geography/sector) for disparate impact.
- Ensure predictions are used at **portfolio level** unless governance approves granular use.
- Provide business users with **plain-language** documentation of assumptions and limitations.

---

## 10) Reproducibility

- Record the exact **feature set** selected by OLS, the **Top-5** from the tree, and any **BetaRegression** flags.
- Pin the Python package versions via `requirements.txt` and a `pip freeze` snapshot.
- Keep scripts/notebooks and **random seeds** under version control.

---

## 11) How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- Upload your **training** dataset (required) and **future** dataset (optional).
- Choose the target (`historic_z` by default) and select models (OLS/Beta/DecisionTree).
- For DR plots, either provide `DR` in training or set `mean_abs_DR` in the sidebar.
- Export results/figures directly from the UI (or extend the app to save artifacts).

---

## 12) Changelog

- **1.0.0** — Initial Model Card for Streamlit app; documents OLS subset selection, BetaRegression rescaling, Decision Tree Top-5 selection, DR mapping via Gaussian link.
