# Forward-Looking Credit Risk (IFRS 9) — DR Projection with Macroeconomic Scenarios

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains a **forward-looking (prospective)** credit-risk project aligned with **IFRS 9** principles.  
It demonstrates how to integrate **macroeconomic scenarios** (GDP growth, unemployment, interest rates, inflation, credit spread, and their lags) into **Probability of Default (PD)** / **Default Rate (DR)** projections over 12-month and lifetime horizons.

The project is designed as a practical template for:
- building scenario-driven credit-risk forecasts,
- testing several statistical / machine-learning approaches,
- comparing model performance on historical data,
- and presenting results through a simple **Streamlit** interface.

> Used Models: **OLS**, **Beta regression**, and **Decision Tree**.

> Best Model: **Decision Tree** with an **Adjusted R² of 0.943** on the train set.

---

## Table of contents

- [Project overview](#project-overview)
- [Business context](#business-context)
- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Methodology](#methodology)
- [Data requirements](#data-requirements)
- [How to run the project](#how-to-run-the-project)
- [How the Streamlit app works](#how-the-streamlit-app-works)
- [Outputs and expected artifacts](#outputs-and-expected-artifacts)
- [Model evaluation](#model-evaluation)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Reproducibility notes](#reproducibility-notes)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Model card](#model-card)

---

## Project overview

The goal of this project is to project credit default rates using a **forward-looking framework** inspired by **IFRS 9 expected credit loss** methodology. Rather than relying only on historical averages, the project incorporates macroeconomic assumptions and scenario analysis to estimate how risk may evolve under different economic conditions.

In practice, the workflow is:
1. collect historical default / DR data and macroeconomic indicators,
2. align both datasets on a common time axis,
3. engineer lagged and transformed features,
4. train candidate models,
5. compare their predictive performance,
6. and generate scenario-based projections for future periods.

---

## Business context

Under IFRS 9, credit-risk measurements should reflect **reasonable and supportable forward-looking information**. This means that projections should not be purely backward-looking: they should incorporate the expected path of economic variables that influence portfolio quality.

This repository demonstrates a simplified version of that idea by linking DR dynamics to macroeconomic drivers such as:
- GDP growth,
- unemployment rate,
- policy / interest rates,
- inflation,
- credit spread,
- and lagged versions of those variables.

The result is a transparent and explainable framework that can be adapted to different portfolios, rating segments, or internal risk policies.

---

## Key features

- **Scenario-based forecasting**: baseline, adverse, and optimistic macro paths.
- **Multiple model families**: linear, bounded-response, and tree-based approaches.
- **Feature engineering**: lagging, scaling, and transformation of macroeconomic inputs.
- **Backtesting support**: evaluate forecasts against historical observations.
- **Interactive dashboard**: explore assumptions and visualize projected DR paths.
- **Exportable results**: save projected outputs for downstream analysis.

---

## Repository structure

The exact structure may evolve, but the project typically includes:

- `main.ipynb` — main notebook used for exploration, training, and validation.
- `streamlit_app.py` — interactive app for scenario selection and visualization.
- `artifacts/` — saved model objects such as scalers and trained estimators.
- `data/` — macroeconomic inputs and historical default / DR datasets.
- `MODEL_CARD.md` — detailed documentation about the model and its use.
- `LICENSE` — license terms.

If you add more modules later, consider splitting the code into:
- `src/data/` for preprocessing,
- `src/features/` for feature engineering,
- `src/models/` for model training,
- `src/visualization/` for charts,
- `src/utils/` for shared helpers.

---

## Methodology

### 1) Data preparation

The model starts by preparing historical macroeconomic and credit-risk series:
- handling missing values,
- aligning frequencies (monthly, quarterly, etc.),
- generating lagged variables,
- scaling numeric inputs,
- and engineering additional predictors such as changes, spreads, or trend indicators.

### 2) Estimation

Several candidate models can be fitted to the prepared dataset:
- **OLS** as a baseline linear model,
- **Beta regression** when the response is bounded between 0 and 1,
- **Decision Tree** for non-linear relationships and interaction effects.

### 3) Scenario generation

Future macroeconomic paths are then provided under different scenarios:
- **Baseline**: central economic view,
- **Adverse**: stressed / deteriorating conditions,
- **Optimistic**: favorable conditions.

These scenarios are propagated through the model to obtain projected DR values.

### 4) Validation

Model performance should be assessed using historical holdout periods or rolling windows. Useful metrics include:
- **MAE**
- **RMSE**
- **R² / Adjusted R²**
- **Directional accuracy**
- **Stability across time**

### 5) Decisioning and reporting

The resulting projections can be used to support:
- expected credit loss calculations,
- sensitivity analysis,
- scenario comparison,
- portfolio-level monitoring,
- and management reporting.

---

## Data requirements

The repository expects two broad data categories:

### 1) Macroeconomic indicators
Example file: `data/file_name.xlxs`

Possible columns:
- `date`
- `gdp_growth`
- `unemployment_rate`
- `policy_rate`
- `cpi_inflation`
- `credit_spread`

### 2) Historical default / DR data
Example file: `data/historical_pd.xlsx`

Possible columns:
- `date`
- `segment`
- `pd_obs`

### Important notes
- Ensure the time frequency is consistent or properly converted.
- Check whether macro variables must be shifted by one or more periods to avoid look-ahead bias.
- If multiple segments exist, align the data by portfolio, vintage, or rating band as needed.
- Validate data types and missing-value handling before training.

---

## How to run the project

### Prerequisites
- Python 3.10 or later
- `uv` installed, or an equivalent Python environment manager
- access to the required input data files

### Setup

```bash
# Clone the repository
 git clone https://github.com/eskenderayadi/IFRS9-DR-PROJECTION.git nexialog
 cd nexialog

# Create and activate the virtual environment
 uv sync
 source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Place input data in ./data if needed
# Update the configuration in streamlit_app.py to match your file names and paths

# Launch the application
 streamlit run streamlit_app.py
```

### Notebook workflow

If you prefer to reproduce the analysis step by step:
1. open `main.ipynb`,
2. run all cells in order,
3. inspect preprocessing and model-training outputs,
4. save the final artifacts to `artifacts/`,
5. then launch the Streamlit app.

---

## How the Streamlit app works

The Streamlit application is intended as a lightweight risk dashboard. It allows you to:
- choose a scenario (**baseline**, **adverse**, **optimistic**),
- select the projection horizon,
- adjust macro shocks such as changes in unemployment, rates, or GDP,
- compare projected DR values against historical data,
- and download the output as a CSV file.

Typical user flow:
1. load model artifacts,
2. choose scenario assumptions,
3. generate projections,
4. inspect the chart/table output,
5. export results for reporting.

---

## Outputs and expected artifacts

Depending on how you structure the project, you may produce:
- trained model files such as `model.joblib`,
- preprocessing objects such as `scaler.joblib`,
- projected DR tables,
- backtesting summaries,
- evaluation plots,
- feature-importance charts,
- and exported CSV results.

A good practice is to keep generated artifacts separated from source code and to document the naming convention used for each file.

---

## Model evaluation

The current repository reports:
- **OLS**, **Beta**, and **Decision Tree** as tested approaches,
- **Decision Tree** as the best-performing model,
- **Adjusted R² = 0.943** on the training set.

For a more complete assessment, consider reporting:
- train / validation / test metrics,
- time-based out-of-sample performance,
- error distributions,
- calibration checks,
- and scenario robustness.

Because credit-risk data is often time-dependent, random train-test splits are usually less appropriate than chronological splits.

---

## Assumptions and limitations

This project is a useful prototype, but it still relies on simplifying assumptions:
- macroeconomic inputs are assumed to be available and sufficiently reliable,
- model performance may vary across portfolios and time periods,
- a single model may not generalize equally well across all economic regimes,
- and the current implementation should be validated against governance, audit, and model-risk requirements before production use.

In addition, IFRS 9 implementations typically require:
- documented expert judgment,
- scenario weighting,
- governance controls,
- explainability checks,
- and strong data lineage.

---

## Reproducibility notes

To reproduce the project reliably:
- pin package versions where possible,
- document the data source and refresh date,
- store preprocessing assumptions alongside model artifacts,
- keep scenario definitions explicit and versioned,
- and record the exact notebook / script used to generate results.

If you want stronger reproducibility, consider adding:
- a `requirements.txt` or lock file,
- a `config.py` or YAML configuration,
- seed control for any randomized procedures,
- and a dedicated training script outside the notebook.

---

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgements

This project was prepared as part of the **Nexialog Consulting Challenge**. Special thanks to Mr. Salem for the opportunity and support.

---

## Model card

See [MODEL_CARD.md](MODEL_CARD.md) for detailed documentation, including model purpose, assumptions, limitations, and usage guidance.
