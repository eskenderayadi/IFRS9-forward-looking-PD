# Nexialog Project

This project analyzes macroeconomic data to model and predict 2020-2022 quarterly default rates (**DR**) by modelling and regressing 2010 to 2019 quarterly DRs. It compares three statistical models:

- **Ordinary Least Squares (OLS) regression**
- **Beta Regression** (useful when fitting U or inversed U data forms)
- **Decision Tree Regressor**

Results on the sample dataset are summarized below:

| Model | Adjusted R² (example) |
|-------|-----------------------|
| OLS | 0.975 |
| Beta Regression | 0.939 |
| Decision Tree (train set) | 0.927 |

These values come from the `main_nexialog.ipynb` notebook where several regression variants with lags are tested. The following lines illustrate the best OLS result:

```text
      Dep. Variable:             historic_z   R-squared:      0.979
      Model:                            OLS   Adj. R-squared: 0.975
```

## Using the Streamlit app

The [`streamlit_app.py`](streamlit_app.py) file provides a web interface to run these models without code.
Here are the main steps:

1. **Load the data**
   - Import a training dataset (CSV or Excel) and optionally a future dataset for forecasts.
2. **Choose the variables**
   - Select the target variable (default `historic_z`) and the explanatory variables.
   - Optionally enable automatic best subset selection for OLS.
3. **Select the models**
   - Tick the models to estimate (OLS, Beta Regression, Decision Tree). If Beta Regression is unavailable, a warning is displayed.
4. **Display the results**
   - Each model shows its coefficients and adjusted R².
   - Interactive charts compare observed and predicted values for `historic_z` and `DR` on the training and projection samples.

To launch the app:
```bash
streamlit run streamlit_app.py
```

You will then have a full dashboard to test different variable configurations and instantly visualize their impact on the forecasts.
