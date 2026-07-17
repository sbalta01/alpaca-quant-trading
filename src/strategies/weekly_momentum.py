# src/strategies/weekly_momentum.py
"""
Weekly cross-sectional momentum portfolio (long-only, retail-friendly).

Three independent layers, all computable with nothing but daily closes:
  1. Momentum score : average of z-scored 12-1m and 6-1m returns across the
                      universe (skip the most recent month - short-term reversal).
  2. Sizing         : inverse-volatility weights over the top_k names, with a
                      per-name cap and renormalization.
  3. Regime gate    : scale gross exposure down when the benchmark (SPY) trades
                      below its 200-day moving average. Remainder stays in cash.

Everything uses data up to and including the signal date only - no lookahead.
Pure pandas/numpy: no sklearn/torch dependency, so it is cheap to run in CI.

This module is deliberately independent of the Strategy/BacktestEngine stack:
weights are cross-sectional portfolio weights (not per-symbol 0/1 positions),
so it comes with its own walk-forward backtester (main/backtest_weekly_momentum.py).
"""
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def momentum_scores(
    prices: pd.DataFrame,
    lookbacks: Tuple[int, ...] = (252, 126),
    skip: int = 21,
) -> pd.DataFrame:
    """
    Cross-sectional momentum z-scores.

    prices : wide DataFrame of daily closes (index=date, columns=ticker).
    Returns a DataFrame of scores (same shape); row t uses data up to t.
    """
    scores = []
    for lb in lookbacks:
        mom = prices.shift(skip) / prices.shift(lb) - 1.0
        z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1) + 1e-12, axis=0)
        scores.append(z)
    return sum(scores) / len(scores)


def inverse_vol_weights(
    prices: pd.DataFrame,
    members: Iterable[str],
    vol_window: int = 63,
    weight_cap: float = 0.20,
) -> pd.Series:
    """
    Inverse-volatility weights over `members`, capped per name and renormalized.
    Uses the trailing `vol_window` daily returns ending at prices.index[-1].
    """
    members = list(members)
    rets = prices[members].pct_change().iloc[-vol_window:]
    vol = rets.std()
    inv = 1.0 / (vol + 1e-12)
    w = inv / inv.sum()

    # iterative cap-and-redistribute
    for _ in range(20):
        over = w > weight_cap
        if not over.any():
            break
        excess = float((w[over] - weight_cap).sum())
        w[over] = weight_cap
        under = ~over
        if w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def regime_exposure(
    benchmark_prices: pd.Series,
    ma_window: int = 200,
    low_exposure: float = 0.4,
) -> float:
    """1.0 when the benchmark closes at/above its MA, else `low_exposure`."""
    px = benchmark_prices.dropna()
    if len(px) < ma_window:
        return 1.0
    ma = px.rolling(ma_window).mean().iloc[-1]
    return 1.0 if px.iloc[-1] >= ma else low_exposure


def compute_target_weights(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    top_k: int = 10,
    lookbacks: Tuple[int, ...] = (252, 126),
    skip: int = 21,
    vol_window: int = 63,
    weight_cap: float = 0.20,
    ma_window: int = 200,
    low_exposure: float = 0.4,
) -> pd.Series:
    """
    Target portfolio weights as of prices.index[-1] (weights sum to <= 1;
    the remainder is cash). Uses only data up to and including that date.
    """
    score_today = momentum_scores(prices, lookbacks, skip).iloc[-1].dropna()
    if score_today.empty:
        return pd.Series(dtype=float)
    members = score_today.nlargest(top_k).index
    w = inverse_vol_weights(prices, members, vol_window, weight_cap)
    expo = regime_exposure(benchmark, ma_window, low_exposure)
    return (w * expo).sort_values(ascending=False)


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week present in `index`."""
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    last = s.groupby([iso.year.values, iso.week.values]).last()
    return pd.DatetimeIndex(sorted(last.values))


def run_walkforward(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    top_k: int = 10,
    lookbacks: Tuple[int, ...] = (252, 126),
    skip: int = 21,
    vol_window: int = 63,
    weight_cap: float = 0.20,
    ma_window: int = 200,
    low_exposure: float = 0.4,
    cost_bps: float = 0.0,
    warmup: int = None,
) -> dict:
    """
    Walk-forward weekly backtest.

    At each weekly rebalance date t, weights are computed from data up to t
    and applied to returns starting the NEXT trading day (no lookahead).
    cost_bps is charged on turnover (sum |dw|) at each rebalance; the caller
    can set 0 to assume free trading.

    Returns dict with 'daily_returns', 'weights' (per rebalance date),
    'turnover' (per rebalance date), 'exposure' (per rebalance date).
    """
    warmup = warmup if warmup is not None else max(max(lookbacks) + skip, ma_window) + 5
    if warmup >= len(prices.index):
        raise ValueError(
            f"Not enough history: need > {warmup} rows for warm-up, got {len(prices.index)}")
    daily_rets = prices.pct_change().fillna(0.0)
    rebal_dates = [d for d in weekly_rebalance_dates(prices.index) if d >= prices.index[warmup]]

    w_current = pd.Series(dtype=float)
    port_ret = pd.Series(0.0, index=prices.index)
    weights_hist, turnover_hist, expo_hist = {}, {}, {}

    rebal_set = set(rebal_dates)
    pending_weights = None
    for i, date in enumerate(prices.index):
        # apply yesterday's decision this morning
        if pending_weights is not None:
            w_current = pending_weights
            pending_weights = None
        # accrue today's return with weights held today
        if len(w_current) > 0:
            port_ret.loc[date] = float((daily_rets.loc[date, w_current.index] * w_current).sum())
        # decide at tonight's close if it is a rebalance date
        if date in rebal_set:
            hist = prices.loc[:date]
            w_new = compute_target_weights(
                hist, benchmark.loc[:date], top_k, lookbacks, skip,
                vol_window, weight_cap, ma_window, low_exposure,
            )
            all_names = w_new.index.union(w_current.index)
            turnover = float(
                (w_new.reindex(all_names, fill_value=0.0)
                 - w_current.reindex(all_names, fill_value=0.0)).abs().sum())
            # charge costs on the day the trades execute (next day open ~ close of next day here)
            port_ret.loc[date] -= turnover * cost_bps / 1e4
            weights_hist[date] = w_new
            turnover_hist[date] = turnover
            expo_hist[date] = float(w_new.sum())
            pending_weights = w_new

    return {
        "daily_returns": port_ret,
        "weights": weights_hist,
        "turnover": pd.Series(turnover_hist),
        "exposure": pd.Series(expo_hist),
    }


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Standard risk/return metrics on a daily-return series."""
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    cum = float((1 + r).prod())
    years = n / periods_per_year
    cagr = cum ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(r.std(ddof=1)) * np.sqrt(periods_per_year)
    sharpe = float(r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year) if r.std(ddof=1) > 0 else np.nan
    downside = float(r[r < 0].std(ddof=1))
    sortino = float(r.mean() / downside) * np.sqrt(periods_per_year) if downside > 0 else np.nan
    curve = (1 + r).cumprod()
    max_dd = float((curve / curve.cummax() - 1).min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Total Return": cum - 1, "CAGR": cagr, "Vol (ann)": vol, "Sharpe": sharpe,
        "Sortino": sortino, "Max Drawdown": max_dd, "Calmar": calmar,
    }
