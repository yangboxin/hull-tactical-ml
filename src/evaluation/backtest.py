# src/evaluation/backtest.py
from __future__ import annotations

import numpy as np


def _equity_curve(returns: np.ndarray, use_log: bool = False) -> np.ndarray:
    """
    Build an equity curve from per-period returns.
    If use_log=True, treat returns as log returns (sum).
    Otherwise treat as simple returns (cumprod(1+r)).
    """
    r = np.asarray(returns).reshape(-1)

    if use_log:
        return np.cumsum(r)
    return np.cumprod(1.0 + r) - 1.0


def max_drawdown(returns: np.ndarray, use_log: bool = False) -> float:
    """
    Maximum drawdown of an equity curve built from returns.
    Returns negative number (e.g., -0.12 means -12% max drawdown).
    """
    eq = _equity_curve(returns, use_log=use_log)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min()) if len(dd) else 0.0


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, eps: float = 1e-12) -> float:
    """
    Sharpe ratio using per-period returns.
    risk_free is per-period risk-free (default 0 for simplicity).
    """
    r = np.asarray(returns).reshape(-1)
    if len(r) == 0:
        return 0.0
    excess = r - risk_free
    mu = float(np.mean(excess))
    sd = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    return mu / (sd + eps)


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, eps: float = 1e-12) -> float:
    """
    Sortino ratio using downside deviation.
    """
    r = np.asarray(returns).reshape(-1)
    if len(r) == 0:
        return 0.0
    excess = r - risk_free
    downside = excess[excess < 0]
    dd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    mu = float(np.mean(excess))
    return mu / (dd + eps)


def long_or_cash_strategy_returns(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Simple strategy:
      if predicted return > threshold -> long (signal=1)
      else -> cash (signal=0)

    strategy_return_t = signal_t * true_return_t
    """
    y_true = np.asarray(y_true_returns).reshape(-1)
    y_pred = np.asarray(y_pred_returns).reshape(-1)

    signal = (y_pred > threshold).astype(float)
    return signal * y_true


def backtest_summary(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray,
    risk_free: float = 0.0,
    threshold: float = 0.0,
) -> dict:
    """
    Convenience wrapper that computes strategy returns + key portfolio metrics.
    """
    strat_r = long_or_cash_strategy_returns(y_true_returns, y_pred_returns, threshold=threshold)
    return {
        "sharpe": sharpe_ratio(strat_r, risk_free=risk_free),
        "sortino": sortino_ratio(strat_r, risk_free=risk_free),
        "max_drawdown": max_drawdown(strat_r, use_log=False),
        "mean_return": float(np.mean(strat_r)) if len(strat_r) else 0.0,
        "std_return": float(np.std(strat_r, ddof=1)) if len(strat_r) > 1 else 0.0,
        "n_days": int(len(strat_r)),
    }

def buy_and_hold_summary(
    y_true_returns: np.ndarray,
    risk_free: float = 0.0,
) -> dict:
    """
    Buy-and-hold baseline: always long the market.
    """
    r = np.asarray(y_true_returns).reshape(-1)

    return {
        "sharpe": sharpe_ratio(r, risk_free=risk_free),
        "sortino": sortino_ratio(r, risk_free=risk_free),
        "max_drawdown": max_drawdown(r, use_log=False),
        "mean_return": float(np.mean(r)) if len(r) else 0.0,
        "std_return": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
        "n_days": int(len(r)),
    }
