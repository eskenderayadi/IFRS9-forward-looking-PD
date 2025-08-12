# streamlit_app.py
# ------------------------------------------------------------------
#  pip install streamlit pandas numpy scipy statsmodels scikit-learn plotly
# ------------------------------------------------------------------
"""
Streamlit : OLS • BetaRegression • Decision Tree (+ inversion historic_z → DR)

• OLS : meilleur sous-ensemble p ≤ 0.05  
• BetaRegression : mêmes variables  
• DecisionTree :  
    1. arbre “large” sur **toutes** les variables X  
    2. on garde les 5 variables les plus importantes  
    3. arbre “compact” (paramètres par défaut) sur ces Top-5  
• 4 graphiques : in/out pour historic_z et DR
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import plotly.graph_objects as go
from scipy.stats import norm

# ---------- compat BetaRegression ----------
try:
    from statsmodels.genmod.families import Beta as BetaFamily
    HAS_BETA = True
except ImportError:
    HAS_BETA = False
    try:
        from statsmodels.othermod.betareg import BetaModel
    except ImportError:
        BetaModel = None

EPS = 1e-6
st.set_page_config(page_title="OLS / Beta / DecisionTree", layout="wide")
st.title("Analyse automatique : OLS • Beta • Decision Tree")

# ------------------------------------------------------------------ 1. Upload
st.sidebar.header("1. Données")
base_file   = st.sidebar.file_uploader("• Jeu d'entraînement", ("csv", "xlsx"), key="base")
future_file = st.sidebar.file_uploader("• Jeu futur (optionnel)", ("csv", "xlsx"), key="future")
if not base_file: st.stop()

read_any  = lambda f: pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
df_base   = read_any(base_file)
df_future = read_any(future_file) if future_file else pd.DataFrame()

st.subheader("Aperçu – entraînement"); st.dataframe(df_base.head())
if not df_future.empty:
    st.subheader("Aperçu – Projections"); st.dataframe(df_future.head())

# ------------------------------------------------------------------ 1 bis. mean_abs_DR
if "DR" in df_base.columns:
    mean_abs_DR = float(np.abs(df_base["DR"]).mean())
else:
    mean_abs_DR = st.sidebar.number_input("mean_abs_DR (si DR absent)",
                                          min_value=0.0, max_value=1.0,
                                          value=0.05, step=0.001)
const_ppf = norm.ppf(mean_abs_DR)
hz_to_dr  = lambda hz: norm.cdf(const_ppf - hz)

# ------------------------------------------------------------------ 2. Variables
num_cols   = df_base.select_dtypes(include=np.number).columns.tolist()
target_var = st.sidebar.selectbox("Variable cible (Y)", num_cols,
                                  index=num_cols.index("historic_z")
                                  if "historic_z" in num_cols else 0)

feature_candidates = [c for c in num_cols if c not in {target_var, "DR"}]
feature_vars       = st.sidebar.multiselect("Variables explicatives (X)",
                                            feature_candidates,
                                            default=feature_candidates[:3])

methods = st.sidebar.multiselect("Modèles",
                                 ["OLS", "BetaRegression", "DecisionTree"],
                                 default=["OLS", "BetaRegression", "DecisionTree"])
if "BetaRegression" in methods and not (HAS_BETA or BetaModel):
    st.sidebar.warning("BetaRegression indisponible (statsmodels).")

auto = st.sidebar.checkbox("Sous-ensemble OLS automatique", True)

if not st.sidebar.button("Lancer l'estimation"): st.stop()

# ------------------------------------------------------------------ 3. Matrices
X_train = df_base[feature_vars].dropna()
y_train = df_base.loc[X_train.index, target_var]
DR_real = df_base.loc[X_train.index, "DR"] if "DR" in df_base.columns else None
fit_ols = lambda X, y: sm.OLS(y, sm.add_constant(X)).fit()

# ------------------------------------------------------------------ 4. OLS
ols_model, ols_vars, r2_adj_ols = None, feature_vars, None
if "OLS" in methods:
    if auto:
        best = None
        for k in range(1, len(feature_vars)+1):
            for sub in combinations(feature_vars, k):
                mdl = fit_ols(X_train[list(sub)], y_train)
                if (mdl.pvalues.drop("const", errors="ignore") < 0.05).all():
                    cand = (mdl.rsquared_adj, sub, mdl)
                    if best is None or cand[0] > best[0]: best = cand
        if best: _, ols_vars, ols_model = best
        else:    ols_model = fit_ols(X_train, y_train)
    else:
        ols_model = fit_ols(X_train, y_train)
    if isinstance(ols_vars, tuple): ols_vars = list(ols_vars)
    r2_adj_ols = ols_model.rsquared_adj

# ------------------------------------------------------------------ 5. BetaRegression
beta_model, beta_pred_train, r2_adj_beta = None, None, None
if "BetaRegression" in methods and (HAS_BETA or BetaModel):
    ymin, ymax = y_train.min(), y_train.max()
    if ymin < ymax:
        y_beta = ((y_train - ymin)/(ymax - ymin)).clip(EPS, 1-EPS)
        Xb = sm.add_constant(X_train[ols_vars])
        try:
            if HAS_BETA:
                beta_model  = sm.GLM(y_beta, Xb, family=BetaFamily()).fit()
                beta_scaled = beta_model.predict(Xb)
            else:
                frm  = "y_beta ~ " + " + ".join(ols_vars)
                data = pd.concat([y_beta.rename("y_beta"), X_train[ols_vars]], axis=1)
                beta_model  = BetaModel.from_formula(frm, data).fit()
                beta_scaled = beta_model.predict()
            beta_pred_train = beta_scaled*(ymax - ymin)+ymin
            r2   = r2_score(y_train, beta_pred_train)
            r2_adj_beta = 1 - (1 - r2)*(len(y_train)-1)/(len(y_train)-len(ols_vars)-1)
        except AssertionError:
            beta_model = None

# ------------------------------------------------------------------ 6. Decision Tree : large ➜ top-5 ➜ compact
dt_model, dt_pred_train, r2_dt = None, None, None
if "DecisionTree" in methods:
    # 1. Arbre large (toutes variables)
    dt_full = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    dt_full.fit(X_train[feature_vars], y_train)
    importances = pd.Series(dt_full.feature_importances_, index=feature_vars)
    importances = importances[importances > 0].sort_values(ascending=False)
    # 2. Top-5
    vars_dt = importances.head(5).index.tolist() or feature_vars[:5]
    # 3. Arbre compact
    dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    dt_model.fit(X_train[vars_dt], y_train)
    dt_pred_train = dt_model.predict(X_train[vars_dt])
    r2_dt = r2_score(y_train, dt_pred_train)
    dt_tbl = (pd.DataFrame({"Variable": vars_dt,
                            "Importance": dt_model.feature_importances_})
              .sort_values("Importance", ascending=False)
              .round(4))

# ------------------------------------------------------------------ 7. Tableaux
st.subheader("Résultats modèles")

def show_coef_table(model, title, r2adj):
    tbl = (pd.DataFrame({"Coef": model.params, "p-val": model.pvalues})
           .round(4).reset_index().rename(columns={"index": "Variable"}))
    st.markdown(f"**{title}**")
    st.table(tbl)
    st.markdown(f"**R² ajusté : {r2adj:.3f}**\n")

if ols_model:   show_coef_table(ols_model, "OLS", r2_adj_ols)
if beta_model:  show_coef_table(beta_model, "BetaRegression", r2_adj_beta)
if dt_model:
    st.markdown("**Decision Tree — importance des variables (Top-5)**")
    st.table(dt_tbl)
    st.markdown(f"**R² (train) : {r2_dt:.3f}**\n")

# ------------------------------------------------------------------ 8. Axe temps
non_num = [c for c in df_base.columns if c not in num_cols]
opts = ["(index)"] + non_num
time_col = st.selectbox("Axe temps", opts,
                        index=opts.index("Trimestre") if "Trimestre" in opts else 0)

if time_col == "(index)":
    x_train  = X_train.index
    x_future = range(x_train.max()+1, x_train.max()+1+len(df_future))
else:
    x_train  = df_base.loc[X_train.index, time_col]
    x_future = (df_future[time_col] if not df_future.empty and time_col in df_future.columns
                else range(len(x_train), len(x_train)+len(df_future)))

# ------------------------------------------------------------------ 9. Prédictions historic_z
pred_ols_train  = ols_model.predict(sm.add_constant(X_train[ols_vars])) if ols_model else None
pred_beta_train = beta_pred_train
pred_dt_train   = dt_pred_train

if not df_future.empty:
    Xf = sm.add_constant(df_future[ols_vars], has_constant="add")
    pred_ols_future = ols_model.predict(Xf) if ols_model else None
    if beta_model:
        beta_scaled_f = beta_model.predict(Xf) if HAS_BETA else beta_model.predict(df_future[ols_vars])
        pred_beta_future = beta_scaled_f*(y_train.max()-y_train.min())+y_train.min()
    else: pred_beta_future = None
    pred_dt_future = dt_model.predict(df_future[vars_dt]) if dt_model else None
else:
    pred_ols_future = pred_beta_future = pred_dt_future = None

# ------------------------------------------------------------------ 10. DR
to_dr = lambda arr: hz_to_dr(arr) if arr is not None else None
pred_ols_DR_train, pred_beta_DR_train, pred_dt_DR_train    = map(to_dr,[pred_ols_train,pred_beta_train,pred_dt_train])
pred_ols_DR_future, pred_beta_DR_future, pred_dt_DR_future = map(to_dr,[pred_ols_future,pred_beta_future,pred_dt_future])

# ------------------------------------------------------------------ 11. Graph helper
def add(fig, x, y, lbl):
    if y is not None:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=lbl))

# ------------------- Graphs historic_z -------------------
fig_hz_in = go.Figure()
add(fig_hz_in, x_train, y_train, "historic_z réel")
add(fig_hz_in, x_train, pred_ols_train, "OLS")
add(fig_hz_in, x_train, pred_beta_train, "Beta")
add(fig_hz_in, x_train, pred_dt_train, "DecisionTree")
fig_hz_in.update_layout(title="In-sample : historic_z", hovermode="x unified")
st.plotly_chart(fig_hz_in, use_container_width=True)

if any(v is not None for v in [pred_ols_future, pred_beta_future, pred_dt_future]):
    fig_hz_out = go.Figure()
    add(fig_hz_out, x_future, pred_ols_future, "OLS")
    add(fig_hz_out, x_future, pred_beta_future, "Beta")
    add(fig_hz_out, x_future, pred_dt_future, "DecisionTree")
    fig_hz_out.update_layout(title="Out-sample : historic_z", hovermode="x unified")
    st.plotly_chart(fig_hz_out, use_container_width=True)

# ------------------- Graphs DR -------------------
fig_dr_in = go.Figure()
add(fig_dr_in, x_train, DR_real, "DR réel")
add(fig_dr_in, x_train, pred_ols_DR_train, "OLS")
add(fig_dr_in, x_train, pred_beta_DR_train, "Beta")
add(fig_dr_in, x_train, pred_dt_DR_train, "DecisionTree")
fig_dr_in.update_layout(title="In-sample : DR", hovermode="x unified")
st.plotly_chart(fig_dr_in, use_container_width=True)

if any(v is not None for v in [pred_ols_DR_future, pred_beta_DR_future, pred_dt_DR_future]):
    fig_dr_out = go.Figure()
    add(fig_dr_out, x_future, pred_ols_DR_future, "OLS")
    add(fig_dr_out, x_future, pred_beta_DR_future, "Beta")
    add(fig_dr_out, x_future, pred_dt_DR_future, "DecisionTree")
    fig_dr_out.update_layout(title="Out-sample : DR", hovermode="x unified")
    st.plotly_chart(fig_dr_out, use_container_width=True)

if __name__ == "__main__":
    pass
