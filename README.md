# Forward-Looking Credit Risk (IFRS 9) — PD Projection with Macroeconomic Scenarios

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-app-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains a **forward-looking (prospective)** credit-risk project aligned with **IFRS 9** principles.  
It demonstrates how to integrate **macroeconomic scenarios** (GDP growth, unemployment, interest rates, etc.) into **Probability of Default (PD)** projections over 12-month and lifetime horizons.

> Built with **Python** and **Streamlit**. Includes a research **notebook** and an **interactive app** to explore scenarios, visualize PD paths, and compare stress vs. baseline outcomes.

---

## 🚀 Highlights

- **IFRS 9 forward-looking** treatment with scenario-conditioned PD.
- **Feature pipeline** for macro variables → PD drivers (lagging, differencing, scaling).
- **Stat/ML models** (e.g., logistic regression / gradient boosting / GAM—choose your flavor).
- **Scenario engine** to run **baseline / adverse / optimistic** cases and sensitivity sweeps.
- **Streamlit app** for interactive exploration and visualization.
- **Reproducible notebook**: `main.ipynb` (provided).

---

## 🧭 Project structure

```
.
├─ streamlit_app.py          # Interactive app (forward-looking PD scenarios)
├─ main.ipynb                # Research notebook (feature eng., modeling, evaluation)
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ MODEL_CARD.md             # Detailed model documentation
├─ streamlit data/
└─ notebook data/                    
```

> You’ve already provided `main_nexialog.ipynb`. Keep it at repo root or in `notebooks/` and adjust paths accordingly.

---

## 📦 Quickstart

```bash
# 1) Create and activate a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) add data
# Put your macro and default/cohort CSVs in ./data and update streamlit_app.py config

# 4) Run the Streamlit app
streamlit run streamlit_app.py

# 5) Explore the notebook
jupyter lab  # or jupyter notebook
```

---

## 📁 Data expectations (example)

- `data/file_name.xlxs`: monthly/quarterly macro indicators with columns like:
  - `date, gdp_growth, unemployment_rate, policy_rate, cpi_inflation, credit_spread`
- `data/historical_pd.xlsx`: historical PDs (by portfolio/segment/vintage) with columns like:
  - `date, segment, pd_obs`

You can join and align these on `date` (taking care of frequency alignment and lags).

---

## 🧪 Methodology (overview)

1. **Preparation**: Impute, lag, and scale macro variables; engineer changes and levels.
2. **Estimation**: Fit a PD model (e.g., logit with macro drivers + seasonality / cohort effects).
3. **Scenarioing**: Feed **forward** macro paths (baseline/adverse/optimistic) to produce PD projections.
4. **Validation**: Backtest on a rolling window; compare **MAE/RMSE** and **directional accuracy**.
5. **Decisioning**: Export PD paths for ECL components; sensitivity analysis by driver.

> The exact modeling choices are flexible—adapt to your data and policy.

---

## 🖥️ Streamlit app — features

- Select scenario (**baseline / adverse / optimistic**) and horizon.
- Adjust shocks (Δunemployment, Δrates, ΔGDP, etc.).
- Visualize **projected PD** vs. historicals and download the results as CSV.

---

## 📚 Reproduce

- Run the notebook `main_nexialog.ipynb` end-to-end.
- Train/save your model artifacts to `artifacts/` (e.g., `model.joblib`, `scaler.joblib`).
- The Streamlit app will load them at startup.

---

## 🔒 Notes on governance

- Keep a **model card** (assumptions, data lineage, limitations).
- Track **scenario definitions** and **versioned parameters**.
- Document **limitations** (extrapolation risk, regime changes, data gaps).

---

## 📝 License

Released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgements

This project was prepared as part of the **Nexialog Consulting Challenge** and focuses on **forward-looking IFRS 9 PD** modeling with macroeconomic scenarios.

---

**Model Card:** see [MODEL_CARD.md](MODEL_CARD.md) for detailed documentation.
