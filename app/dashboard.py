# app/dashboard.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.evaluation.metrics import regression_metrics, directional_metrics
from src.evaluation.backtest import backtest_summary, buy_and_hold_summary


# -----------------------------
# Helpers
# -----------------------------
def equity_curve(simple_returns: np.ndarray) -> np.ndarray:
    """Cumulative simple return curve: cumprod(1+r) - 1."""
    r = np.asarray(simple_returns).reshape(-1)
    if len(r) == 0:
        return r
    return np.cumprod(1.0 + r) - 1.0


def load_preds_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["date_id", "y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. "
                         f"Expected at least {required}.")

    # Enforce date_id as Trading Day Index (integer), NOT datetime.
    # If someone previously exported as epoch-like timestamps, this will fail early (good).
    df["date_id"] = pd.to_numeric(df["date_id"], errors="raise").astype(np.int64)

    # Optional columns
    if "fold" in df.columns:
        df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    if "model" in df.columns:
        df["model"] = df["model"].astype(str)

    # Sort by time index
    df = df.sort_values("date_id").reset_index(drop=True)

    # Ensure numeric
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    # Drop any rows with NaNs in outputs (should be rare; keeps dashboard robust)
    df = df.dropna(subset=["date_id", "y_true", "y_pred"]).reset_index(drop=True)

    return df


def plot_lines(x, y1, y2, label1, label2, xlabel, ylabel, title):
    fig = plt.figure()
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    st.pyplot(fig)


def plot_hist(values, title, xlabel):
    fig = plt.figure()
    plt.hist(values, bins=60)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    st.pyplot(fig)


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Hull Tactical Dashboard", layout="wide")
st.title("Hull Tactical Market Prediction Dashboard")
st.caption("X-axis uses Trading Day Index (date_id) as provided by the dataset. No calendar-date conversion is applied.")


# Sidebar - model selection
st.sidebar.header("Controls")

MODEL_FILES = {
    "Ridge": Path("data/processed/preds_ridge.csv"),
    "XGBoost": Path("data/processed/preds_xgboost.csv"),
    "LSTM": Path("data/processed/preds_lstm.csv"),
    "Transformer": Path("data/processed/preds_transformer.csv"),
    "Ridge_126d": Path("data/processed/preds_ridge_126d.csv"),
    "XGBoost_126d": Path("data/processed/preds_xgboost_126d.csv"),
    "LSTM_126d": Path("data/processed/preds_lstm_126d.csv"),
    "Transformer_126d": Path("data/processed/preds_transformer_126d.csv"),
}

model_display = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))
csv_path = MODEL_FILES[model_display]

if not csv_path.exists():
    st.error(f"Missing prediction file: {csv_path}\n\n"
             f"Run training/export first (e.g., `python -m scripts.run_all`).")
    st.stop()

try:
    df = load_preds_csv(csv_path)
except Exception as e:
    st.error(f"Failed to load {csv_path}.\n\n{e}")
    st.stop()

# Optional fold filter
folds_available = sorted([int(x) for x in df["fold"].dropna().unique()]) if "fold" in df.columns else []
if folds_available:
    fold_choice = st.sidebar.multiselect("Folds (optional)", folds_available, default=folds_available)
    df = df[df["fold"].isin(fold_choice)].copy()

# Range filter by trading day index
min_id, max_id = int(df["date_id"].min()), int(df["date_id"].max())
rng = st.sidebar.slider("Trading Day Index (date_id) range", min_id, max_id, (min_id, max_id))
df = df[(df["date_id"] >= rng[0]) & (df["date_id"] <= rng[1])].copy()

# Smoothing
smooth = st.sidebar.checkbox("Rolling mean smoothing", value=True)
window = st.sidebar.slider("Smoothing window", 5, 60, 20) if smooth else 1

# Strategy threshold
threshold = st.sidebar.slider("Strategy threshold (ŷ > threshold => long)", -0.02, 0.02, 0.0, 0.001)

# Top-K errors
topk = st.sidebar.slider("Show Top-K absolute errors", 5, 50, 15)

# Extract arrays
x = df["date_id"].to_numpy()
y_true = df["y_true"].to_numpy(dtype=float)
y_pred = df["y_pred"].to_numpy(dtype=float)

# Smooth for visualization only
if smooth and window > 1:
    y_true_plot = pd.Series(y_true).rolling(window, min_periods=1).mean().to_numpy()
    y_pred_plot = pd.Series(y_pred).rolling(window, min_periods=1).mean().to_numpy()
else:
    y_true_plot = y_true
    y_pred_plot = y_pred

# Metrics
reg = regression_metrics(y_true, y_pred)
direc = directional_metrics(y_true, y_pred)
bt = backtest_summary(y_true, y_pred, risk_free=0.0, threshold=float(threshold))
bh = buy_and_hold_summary(y_true, risk_free=0.0)

# KPI row
st.subheader("Key Metrics (Selected Window)")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("RMSE", f"{reg['rmse']:.6f}")
c2.metric("MAE", f"{reg['mae']:.6f}")
c3.metric("R²", f"{reg['r2']:.3f}")
c4.metric("DirAcc", f"{direc['directional_accuracy']:.3f}")
c5.metric("F1", f"{direc['f1']:.3f}")
c6.metric("Sharpe (Model)", f"{bt['sharpe']:.3f}")
c7.metric("MaxDD (Model)", f"{bt['max_drawdown']:.3f}")

c8, c9 = st.columns(2)
c8.metric("Sharpe (Buy&Hold)", f"{bh['sharpe']:.3f}")
c9.metric("MaxDD (Buy&Hold)", f"{bh['max_drawdown']:.3f}")

# Plot section
st.subheader("Predicted vs Real Returns")
plot_lines(
    x=x,
    y1=y_true_plot,
    y2=y_pred_plot,
    label1="Real (y_true)",
    label2="Pred (y_pred)",
    xlabel="Trading Day Index (date_id)",
    ylabel="Daily Return",
    title=f"{model_display}: Returns over Time",
)

# Equity curves
st.subheader("Equity Curve Comparison")
signal = (y_pred > float(threshold)).astype(float)
r_model = signal * y_true
r_bh = y_true

eq_model = equity_curve(r_model)
eq_bh = equity_curve(r_bh)

fig = plt.figure()
plt.plot(x, eq_model, label=f"{model_display} (Long-or-Cash)")
plt.plot(x, eq_bh, label="Buy-and-Hold")
plt.xlabel("Trading Day Index (date_id)")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns (Simple Return Compounding)")
plt.legend()
st.pyplot(fig)

st.caption(
    f"Strategy rule: long if ŷ > {threshold:.3f}, else cash. "
    f"Model Sharpe={bt['sharpe']:.3f}, MaxDD={bt['max_drawdown']:.3f}. "
    f"Buy&Hold Sharpe={bh['sharpe']:.3f}, MaxDD={bh['max_drawdown']:.3f}."
)

# Error analysis
st.subheader("Error Analysis")

err = y_pred - y_true
abs_err = np.abs(err)

colA, colB = st.columns(2)
with colA:
    plot_hist(err, title="Prediction Error Distribution (y_pred - y_true)", xlabel="error")
with colB:
    plot_hist(abs_err, title="Absolute Error Distribution |y_pred - y_true|", xlabel="abs(error)")

# Top-K worst points table
st.subheader(f"Top-{topk} Largest Absolute Errors (Selected Window)")
tmp = df.copy()
tmp["error"] = err
tmp["abs_error"] = abs_err
tmp = tmp.sort_values("abs_error", ascending=False).head(int(topk))

show_cols = ["date_id", "y_true", "y_pred", "error", "abs_error"]
if "fold" in tmp.columns:
    show_cols.insert(1, "fold")
st.dataframe(tmp[show_cols], use_container_width=True)
