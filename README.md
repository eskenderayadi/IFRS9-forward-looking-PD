# Forward-Looking Credit Risk (IFRS 9) — DR Projection with Macroeconomic Scenarios

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains a **forward-looking (prospective)** credit-risk project aligned with **IFRS 9** principles.  
It demonstrates how to integrate **macroeconomic scenarios** (GDP growth, unemployment, interest rates, and their lags) into **Probability of Default (PD)** projections over 12-month and lifetime horizons.

> Used Models : OLS, Bêta, and Decision Tree.

> Best Model : Decision Tree with an **Adjusted R² : 0.943** (on train data).

---

## 📦 Quickstart

```bash
#1) Clone repository with git:
git clone https://github.com/eskenderayadi/IFRS9-DR-PROJECTION.git nexialog
cd nexialog

# 2) Create and activate a virtual env (recommended)
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) (Optional) add data
# Put your macro and default/cohort CSVs in ./data and update streamlit_app.py config

# 4) Run the Streamlit app
streamlit run streamlit_app.py
```

---

## 📁 Data expectations (example)

- `data/file_name.xlxs`: monthly/quarterly macro indicators with columns like:
  - `date, gdp_growth, unemployment_rate, policy_rate, cpi_inflation, credit_spread`
- `data/historical_pd.xlsx`: historical DRs (by portfolio/segment/vintage) with columns like:
  - `date, segment, pd_obs`

You can join and align these on `date` (taking care of frequency alignment and lags).

---

## 🧪 Methodology (overview)

1. **Preparation**: Impute, lag, and scale macro variables; engineer changes and levels.
2. **Estimation**: Fit a DR model (e.g., logit with macro drivers + seasonality / cohort effects).
3. **Scenarioing**: Feed **forward** macro paths (baseline/adverse/optimistic) to produce DR projections.
4. **Validation**: Backtest on a rolling window; compare **MAE/RMSE** and **directional accuracy**.
5. **Decisioning**: Export DR paths for ECL components; sensitivity analysis by driver.

> The exact modeling choices are flexible—adapt to your data and policy.

---

## 🖥️ Streamlit app — features

- Select scenario (**baseline / adverse / optimistic**) and horizon.
- Adjust shocks (Δunemployment, Δrates, ΔGDP, etc.).
- Visualize **projected DR** vs. historicals and download the results as CSV.

---

## 📚 Reproduce

- Run the notebook `main.ipynb` end-to-end.
- Train/save your model artifacts to `artifacts/` (e.g., `model.joblib`, `scaler.joblib`).
- The Streamlit app will load them at startup.

---

## 📝 License

Released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgements

This project was prepared as part of the **Nexialog Consulting Challenge**, so great thanks to Mr. Salem who gave us this great opportunity.
---

**Model Card:** see [MODEL_CARD.md](MODEL_CARD.md) for detailed documentation.
