# src/evaluation/backtest.py
from __future__ import annotations

import numpy as np


def equity_curve_from_returns(returns: np.ndarray, use_log: bool = False) -> np.ndarray:
    """
    Build an equity curve from per-period returns.

    If use_log=True, treat `returns` as log returns and return cumulative log equity.
    Else treat `returns` as simple returns and return cumulative simple equity starting at 1.0.

    Returns
    -------
    equity : np.ndarray
        If use_log=False: equity[t] = Π_{i<=t}(1+r_i)
        If use_log=True : equity[t] = Σ_{i<=t} r_i   (log-equity)
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return r

    if use_log:
        return np.cumsum(r)

    return np.cumprod(1.0 + r)


def max_drawdown_from_equity(equity: np.ndarray, use_log: bool = False) -> float:
    """
    Maximum drawdown computed from an equity curve.

    Parameters
    ----------
    equity : np.ndarray
        If use_log=False: equity is simple equity (e.g., starts at 1.0).
        If use_log=True : equity is cumulative log-equity.

    Returns
    -------
    mdd : float
        Negative number (e.g., -0.12 means -12% max drawdown).
    """
    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    if eq.size == 0:
        return 0.0

    if use_log:
        # drawdown in log space: dd = eq - peak(eq)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        return float(dd.min())

    # simple equity: dd = eq/peak - 1
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, 1.0, peak) - 1.0
    return float(dd.min())


def max_drawdown_from_returns(returns: np.ndarray, use_log: bool = False) -> float:
    """
    Convenience wrapper: compute max drawdown from per-period returns.
    """
    eq = equity_curve_from_returns(returns, use_log=use_log)
    return max_drawdown_from_equity(eq, use_log=use_log)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_daily: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Daily Sharpe ratio (not annualized).
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return 0.0

    excess = r - float(risk_free_daily)
    mu = float(np.mean(excess))
    sd = float(np.std(excess, ddof=1)) if excess.size > 1 else 0.0
    return mu / (sd + eps)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_daily: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Daily Sortino ratio using downside deviation (not annualized).
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return 0.0

    excess = r - float(risk_free_daily)
    downside = excess[excess < 0.0]
    dd = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
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
    y_true = np.asarray(y_true_returns, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred_returns, dtype=np.float64).reshape(-1)

    signal = (y_pred > float(threshold)).astype(np.float64)
    return signal * y_true


def compute_positions(y_pred: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Turn predictions into {-1, 0, +1} positions using a symmetric threshold.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    pos = np.zeros_like(y_pred, dtype=np.float64)
    thr = float(threshold)
    pos[y_pred > thr] = 1.0
    pos[y_pred < -thr] = -1.0
    return pos


def backtest_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free: float = 0.0,
    threshold: float = 0.0,
    cost_per_trade: float = 0.0,
) -> dict:
    """
    Long/short strategy based on prediction sign with optional threshold and transaction costs.

    Parameters
    ----------
    y_true : np.ndarray
        Realized next-day returns.
    y_pred : np.ndarray
        Predicted next-day returns.
    risk_free : float
        Daily risk-free rate (same scale as y_true).
    threshold : float
        Symmetric threshold for taking positions; |y_pred| <= threshold => flat.
    cost_per_trade : float
        Cost applied per unit position change (e.g., 0.0001 = 1bp).
        Flip +1->-1 costs 2*cost_per_trade.

    Returns
    -------
    dict with sharpe, max_drawdown, exposure, trades, mean_return, std_return, n_days
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size == 0:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "exposure": 0.0,
            "trades": 0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "n_days": 0,
        }

    pos = compute_positions(y_pred, threshold=threshold)

    # transaction costs charged on absolute position change
    pos_change = np.abs(np.diff(pos, prepend=0.0))
    costs = float(cost_per_trade) * pos_change

    strat_ret = pos * y_true - costs

    eq = equity_curve_from_returns(strat_ret, use_log=False)
    mdd = max_drawdown_from_equity(eq, use_log=False)
    shp = sharpe_ratio(strat_ret, risk_free_daily=float(risk_free))

    exposure = float(np.mean(np.abs(pos)))
    trades = int(np.sum(pos_change > 0))

    return {
        "sharpe": float(shp),
        "max_drawdown": float(mdd),
        "exposure": exposure,
        "trades": trades,
        "mean_return": float(np.mean(strat_ret)),
        "std_return": float(np.std(strat_ret, ddof=1)) if strat_ret.size > 1 else 0.0,
        "n_days": int(strat_ret.size),
    }


def buy_and_hold_summary(
    y_true_returns: np.ndarray,
    risk_free: float = 0.0,
) -> dict:
    """
    Buy-and-hold baseline: always long the market (no costs).
    """
    r = np.asarray(y_true_returns, dtype=np.float64).reshape(-1)

    return {
        "sharpe": sharpe_ratio(r, risk_free_daily=float(risk_free)),
        "sortino": sortino_ratio(r, risk_free_daily=float(risk_free)),
        "max_drawdown": max_drawdown_from_returns(r, use_log=False),
        "mean_return": float(np.mean(r)) if r.size else 0.0,
        "std_return": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
        "n_days": int(r.size),
    }
