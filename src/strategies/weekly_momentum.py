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
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd


def default_selector(scores: pd.Series, top_k: int) -> pd.Index:
    """Pick the top_k names by score. The default selection rule."""
    return scores.nlargest(top_k).index


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
    selector: Callable[[pd.Series, int], pd.Index] = None,
    exposure_fn: Callable[[pd.DataFrame, pd.Series, pd.Series], float] = None,
) -> pd.Series:
    """
    Target portfolio weights as of prices.index[-1] (weights sum to <= 1;
    the remainder is cash). Uses only data up to and including that date.

    `selector(scores, top_k) -> Index` chooses which names to hold; defaults to
    top_k by score. Swapping it is how the random-k null test and rank
    buffering plug in without forking this function.

    `exposure_fn(prices, benchmark, weights) -> float` sets gross exposure;
    defaults to the 200dma regime gate. `make_vol_target_exposure(...)` builds
    a vol-targeting version (optionally combined with the gate).
    """
    score_today = momentum_scores(prices, lookbacks, skip).iloc[-1].dropna()
    if score_today.empty:
        return pd.Series(dtype=float)
    members = (selector or default_selector)(score_today, top_k)
    w = inverse_vol_weights(prices, members, vol_window, weight_cap)
    if exposure_fn is not None:
        expo = exposure_fn(prices, benchmark, w)
    else:
        expo = regime_exposure(benchmark, ma_window, low_exposure)
    return (w * expo).sort_values(ascending=False)


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week present in `index`."""
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    last = s.groupby([iso.year.values, iso.week.values]).last()
    return pd.DatetimeIndex(sorted(last.values))


def realized_portfolio_vol(
    prices: pd.DataFrame,
    weights: pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized volatility of the (fully-invested) book over the trailing
    `window` daily returns ending at prices.index[-1] - causal by construction.

    Uses the FULL covariance (w' Sigma w), not the sum of individual vols: a
    10-name semiconductor book with pairwise correlations of 0.6-0.8 is ~30%
    more volatile than the diagonal alone suggests, and ignoring that would
    make vol targeting systematically over-lever.
    """
    if len(weights) == 0:
        return np.nan
    w = weights / weights.sum() if weights.sum() > 0 else weights
    rets = prices[w.index].pct_change(fill_method=None).iloc[-window:]
    cov = rets.cov() * periods_per_year
    var = float(w.values @ cov.values @ w.values)
    return np.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan


def vol_target_exposure(
    prices: pd.DataFrame,
    weights: pd.Series,
    target_vol: float = 0.20,
    windows: Tuple[int, ...] = (21, 63),
    max_exposure: float = 1.0,
    min_exposure: float = 0.0,
) -> float:
    """
    Gross exposure that scales the book toward `target_vol` annualized.

    Takes the MAX vol estimate across `windows` (fast 21d + slow 63d): the
    short window reacts quickly to a vol spike (de-risk fast), while the long
    window stays elevated afterwards (re-risk slowly) - mitigating realized-vol
    targeting's known pathology of re-levering straight into a rebound.

    `max_exposure` defaults to 1.0 and should stay there: no leverage. In calm
    regimes the formula will ask for >1x; if you want that upside, raise
    `target_vol`, don't raise the cap.
    """
    if len(weights) == 0:
        return max_exposure
    est = max(realized_portfolio_vol(prices, weights, w) for w in windows)
    if not np.isfinite(est) or est <= 0:
        return max_exposure
    return float(np.clip(target_vol / est, min_exposure, max_exposure))


def make_vol_target_exposure(
    target_vol: float = 0.20,
    windows: Tuple[int, ...] = (21, 63),
    max_exposure: float = 1.0,
    min_exposure: float = 0.0,
    with_regime_gate: bool = True,
    ma_window: int = 200,
    low_exposure: float = 0.4,
):
    """
    Build an `exposure_fn(prices, benchmark, weights) -> float` for
    compute_target_weights / run_walkforward: vol targeting, optionally
    multiplied by the 200dma regime gate (set with_regime_gate=False, or
    low_exposure=1.0, to run vol targeting alone).
    """
    def exposure_fn(prices: pd.DataFrame, benchmark: pd.Series,
                    weights: pd.Series) -> float:
        expo = vol_target_exposure(prices, weights, target_vol, windows,
                                   max_exposure, min_exposure)
        if with_regime_gate:
            expo *= regime_exposure(benchmark, ma_window, low_exposure)
        return float(np.clip(expo, min_exposure, max_exposure))
    return exposure_fn


class BufferedSelector:
    """
    Rank buffering: hold an incumbent until it falls out of the top
    `buffer_mult * top_k`, instead of dropping it the moment it leaves the
    top_k. Weekly re-ranking otherwise churns names that merely slipped from
    rank 10 to 11, which is pure cost for no signal.

    Stateful, so run_walkforward takes a FACTORY - two runs must not share one
    instance or a parameter sweep would contaminate itself.

    buffer_mult=1.0 reduces exactly to `default_selector`.
    """

    def __init__(self, buffer_mult: float = 1.5):
        if buffer_mult < 1.0:
            raise ValueError("buffer_mult must be >= 1.0")
        self.buffer_mult = buffer_mult
        self.held: list = []

    def __call__(self, scores: pd.Series, top_k: int) -> pd.Index:
        k = min(top_k, len(scores))
        ranked = scores.sort_values(ascending=False)
        keep_depth = int(np.ceil(self.buffer_mult * top_k))
        eligible = set(ranked.index[:keep_depth])

        # Incumbents that still exist AND are still within the buffer zone.
        # The `in scores.index` check matters: a delisted holding must not raise.
        survivors = [h for h in self.held if h in eligible]
        survivors = survivors[:k]

        if len(survivors) < k:
            fill = [t for t in ranked.index if t not in survivors]
            survivors += fill[:k - len(survivors)]

        # Re-order by score so the output is deterministic regardless of history.
        self.held = list(ranked.index[ranked.index.isin(survivors)])[:k]
        return pd.Index(self.held)


def weekly_rebalance_dates_on(index: pd.DatetimeIndex, weekday: int = 4) -> pd.DatetimeIndex:
    """
    Last trading day of each ISO week at or before `weekday` (Mon=0 ... Fri=4).

    Sibling of weekly_rebalance_dates (which is Friday-close by construction);
    used by the rebalance-weekday robustness sweep. A genuine edge should not
    depend on which day of the week it is rebalanced.
    """
    sub = index[index.dayofweek <= weekday]
    if len(sub) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(sub, index=sub)
    iso = sub.isocalendar()
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
    selector_factory: Callable[[], Callable[[pd.Series, int], pd.Index]] = None,
    rebal_dates: pd.DatetimeIndex = None,
    min_trade_fraction: float = 0.0,
    exposure_fn: Callable[[pd.DataFrame, pd.Series, pd.Series], float] = None,
) -> dict:
    """
    Walk-forward weekly backtest.

    At each weekly rebalance date t, weights are computed from data up to t
    and applied to returns starting the NEXT trading day (no lookahead).
    cost_bps is charged on turnover (sum |dw|) at each rebalance; the caller
    can set 0 to assume free trading.

    `selector_factory` is called ONCE per run to build the selection rule. It
    is a factory rather than an instance because stateful selectors (e.g.
    BufferedSelector) must not carry state across runs in a parameter sweep.

    `warmup` should be passed explicitly when comparing configs with different
    lookbacks, so every config is evaluated over an identical window.
    `rebal_dates` overrides the default Friday-close schedule.

    `min_trade_fraction`: skip per-name weight changes smaller than this (the
    live executor already skips trades under 0.5% of equity, so the default
    0.0 backtest OVERSTATES turnover relative to what actually trades; pass
    0.005 to mirror deploy_weekly_momentum.MIN_TRADE_FRACTION).

    Returns dict with 'daily_returns', 'weights' (per rebalance date),
    'turnover' (per rebalance date), 'exposure' (per rebalance date).
    """
    warmup = warmup if warmup is not None else max(max(lookbacks) + skip, ma_window) + 5
    if warmup >= len(prices.index):
        raise ValueError(
            f"Not enough history: need > {warmup} rows for warm-up, got {len(prices.index)}")
    selector = (selector_factory or (lambda: default_selector))()
    # `alive` marks names with a real print. A held name that goes dead is
    # liquidated to CASH below rather than being left to "earn" 0% forever
    # (which is what pct_change().fillna(0) would silently imply).
    alive = prices.notna()
    # fill_method=None: never pad across a gap. Padding would manufacture a 0%
    # return for a dead name, which is precisely the zombie behavior `alive`
    # exists to prevent.
    daily_rets = prices.pct_change(fill_method=None).fillna(0.0)
    schedule = weekly_rebalance_dates(prices.index) if rebal_dates is None else rebal_dates
    rebal_dates = [d for d in schedule if d >= prices.index[warmup]]

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
        # drop any holding that has gone dead; its weight becomes cash
        if len(w_current) > 0:
            live = alive.loc[date, w_current.index]
            if not live.all():
                w_current = w_current[live.values]
        # accrue today's return with weights held today
        if len(w_current) > 0:
            port_ret.loc[date] = float((daily_rets.loc[date, w_current.index] * w_current).sum())
        # decide at tonight's close if it is a rebalance date
        if date in rebal_set:
            hist = prices.loc[:date]
            w_new = compute_target_weights(
                hist, benchmark.loc[:date], top_k, lookbacks, skip,
                vol_window, weight_cap, ma_window, low_exposure,
                selector=selector, exposure_fn=exposure_fn,
            )
            # No-trade band: leave sub-threshold deltas at their current weight,
            # exactly as the live executor does. Default 0.0 = trade everything.
            if min_trade_fraction > 0 and len(w_current) > 0:
                names = w_new.index.union(w_current.index)
                tgt = w_new.reindex(names, fill_value=0.0)
                cur = w_current.reindex(names, fill_value=0.0)
                small = (tgt - cur).abs() < min_trade_fraction
                tgt[small] = cur[small]
                w_new = tgt[tgt > 0].sort_values(ascending=False)
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
