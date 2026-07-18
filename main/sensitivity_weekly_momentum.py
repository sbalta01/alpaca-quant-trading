# main/sensitivity_weekly_momentum.py
"""
Diagnostics for the weekly momentum strategy: is the edge real, or is it
concentration luck and a knife-edge parameter choice?

Two experiments:

1. RANDOM-K NULL TEST (--null-test)
   Comparing the strategy to an equal-weight universe is confounded: they
   differ in BOTH signal and concentration, so the comparison cannot isolate
   the signal's contribution. This runs the identical machinery - same
   inverse-vol sizing, same 20% cap, same regime gate, same rebalance dates,
   same costs - but selects top_k names AT RANDOM. The resulting distribution
   is "concentration + inverse-vol + gate, with no signal". Where the real
   strategy falls in that distribution is the honest measure of the signal.

   Interpretation (committed in advance):
     >= 95th percentile  -> the signal is doing real work
     ~ median            -> the strategy is a concentration expression;
                            momentum contributes nothing
     in between          -> weak positive edge; size expectations accordingly

2. PARAMETER SENSITIVITY (--oat)
   One-at-a-time sweeps over every parameter. A real edge degrades gracefully;
   an artifact collapses when you nudge a knob. The rebalance-weekday sweep is
   the most diagnostic of these - a genuine edge is not day-of-week specific.

Usage (from the repo root, with .venv active):

    python main/sensitivity_weekly_momentum.py --null-test
    python main/sensitivity_weekly_momentum.py --oat
    python main/sensitivity_weekly_momentum.py --null-test --oat --cost-bps 10
    python main/sensitivity_weekly_momentum.py --oat --cache panel.pkl   # reuse download
"""
import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.strategies.weekly_momentum import (
    performance_metrics, run_walkforward, weekly_rebalance_dates_on,
)
from main.backtest_weekly_momentum import fetch_close_matrix

# Base configuration = what is currently deployed.
BASE = dict(top_k=10, lookbacks=(252, 126), skip=21, vol_window=63,
            weight_cap=0.20, ma_window=200, low_exposure=0.4)

# One-at-a-time sweep axes.
AXES = {
    "top_k": [5, 8, 10, 12, 15, 20, 25],
    "lookbacks": [(252,), (126,), (252, 126), (252, 126, 63)],
    "skip": [0, 5, 10, 21, 42],
    "vol_window": [21, 42, 63, 126, 252],
    "weight_cap": [0.10, 0.15, 0.20, 0.25, 0.35, 1.0],
    "low_exposure": [0.0, 0.2, 0.4, 0.6, 1.0],
}


def load_panel(years: int = 10, tickers=None, cache_path: str = None):
    """Wide close matrix + benchmarks. Cached to disk so sweeps never re-download."""
    if cache_path:
        try:
            store = pd.read_pickle(cache_path)
            print(f"Loaded cached panel from {cache_path}")
            return store["prices"], store["bench"]
        except (FileNotFoundError, KeyError):
            pass

    if tickers:
        universe = tickers
    else:
        from src.data.data_loader import fetch_nasdaq_100_symbols
        universe = fetch_nasdaq_100_symbols()
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25) + 400)
    print(f"Downloading {len(universe)} tickers...")
    prices = fetch_close_matrix(sorted(set(universe)), start, end)
    bench = fetch_close_matrix(["SPY", "QQQ"], start, end)
    prices = prices.loc[:, prices.notna().sum() > 300]
    prices = prices.reindex(bench.index).ffill()
    print(f"Got {prices.shape[1]} usable tickers, {prices.shape[0]} days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")
    if cache_path:
        pd.to_pickle({"prices": prices, "bench": bench}, cache_path)
        print(f"Cached panel to {cache_path}")
    return prices, bench


def grid_warmup(axes: dict, base: dict) -> int:
    """
    Warm-up pinned to the LARGEST lookback/ma_window anywhere in the sweep.

    Without this, configs with shorter lookbacks would start trading earlier and
    be scored over a different (longer, differently-composed) window - silently
    biasing the comparison toward short lookbacks.
    """
    all_lookbacks = list(axes.get("lookbacks", [])) + [base["lookbacks"]]
    max_lb = max(max(lb) for lb in all_lookbacks)
    max_skip = max(list(axes.get("skip", [])) + [base["skip"]])
    max_ma = max(list(axes.get("ma_window", [])) + [base["ma_window"]])
    return max(max_lb + max_skip, max_ma) + 5


def _run(prices, bench, cfg: dict, cost_bps: float, warmup: int,
         selector_factory=None, rebal_dates=None) -> pd.Series:
    res = run_walkforward(prices, bench["SPY"], cost_bps=cost_bps, warmup=warmup,
                          selector_factory=selector_factory, rebal_dates=rebal_dates,
                          **cfg)
    return res["daily_returns"], res["turnover"]


def _metrics_on(returns: pd.Series, window: pd.DatetimeIndex) -> dict:
    return performance_metrics(returns.reindex(window).fillna(0.0))


def rolling_3y_min_sharpe(returns: pd.Series, window: pd.DatetimeIndex) -> float:
    """Worst rolling 3-year Sharpe - distinguishes a steady edge from one great year."""
    r = returns.reindex(window).fillna(0.0)
    if len(r) < 756 + 20:
        return np.nan
    roll = r.rolling(756)
    sharpe = (roll.mean() / roll.std(ddof=1)) * np.sqrt(252)
    return float(sharpe.min())


# ------------------------------------------------------------------ null test
def _iid_random_factory(seed: int):
    """Fresh random draw every rebalance. Churns ~the whole book weekly."""
    def factory():
        rng = np.random.RandomState(seed)
        def pick(scores: pd.Series, top_k: int) -> pd.Index:
            k = min(top_k, len(scores))
            return pd.Index(rng.choice(scores.index, size=k, replace=False))
        return pick
    return factory


def _sticky_random_factory(seed: int, replace_n: int = 2):
    """
    Random selection that HOLDS its picks, swapping only `replace_n` names per
    rebalance. This is the turnover-matched null: without it, an i.i.d. random
    selector replaces ~the entire book weekly (~90x/yr turnover) and its cost
    drag alone would make any real strategy look skilful.
    """
    def factory():
        rng = np.random.RandomState(seed)
        held: list = []

        def pick(scores: pd.Series, top_k: int) -> pd.Index:
            nonlocal held
            k = min(top_k, len(scores))
            universe = list(scores.index)
            held = [h for h in held if h in scores.index]      # drop departed names
            if len(held) < k:                                   # initial fill
                pool = [u for u in universe if u not in held]
                held += list(rng.choice(pool, size=k - len(held), replace=False))
            else:
                n_swap = min(replace_n, k)
                if n_swap > 0:
                    out = list(rng.choice(held, size=n_swap, replace=False))
                    keep = [h for h in held if h not in out]
                    pool = [u for u in universe if u not in keep]
                    held = keep + list(rng.choice(pool, size=n_swap, replace=False))
            return pd.Index(held[:k])
        return pick
    return factory


def random_k_null(prices, bench, base: dict, n_seeds: int = 200,
                  cost_bps: float = 10.0, warmup: int = None,
                  mode: str = "iid", replace_n: int = 2) -> pd.DataFrame:
    """
    Null distribution: identical pipeline, but names chosen at random.

    mode='iid'    - fresh draw each week (high turnover; NOT turnover-matched)
    mode='sticky' - holds picks, swaps `replace_n` per week (turnover-matched)

    Returns one row per seed with that draw's metrics and realized turnover.
    """
    warmup = warmup if warmup is not None else grid_warmup({}, base)
    build = _iid_random_factory if mode == "iid" else (
        lambda s: _sticky_random_factory(s, replace_n))
    rows = []
    for seed in range(n_seeds):
        rets, turn = _run(prices, bench, base, cost_bps, warmup,
                          selector_factory=build(seed))
        m = performance_metrics(rets.loc[rets.ne(0).idxmax():])
        m["seed"] = seed
        m["turnover_ann"] = float(turn.mean() * 52)
        rows.append(m)
        if (seed + 1) % 25 == 0:
            print(f"  ...{seed + 1}/{n_seeds} seeds")
    return pd.DataFrame(rows)


def report_null(null_df: pd.DataFrame, real: dict, metric: str = "Sharpe",
                label: str = "", real_turnover: float = None) -> None:
    vals = null_df[metric].dropna().values
    real_v = real[metric]
    pct = float((vals < real_v).mean() * 100)
    null_to = float(null_df["turnover_ann"].median())
    print(f"\n=== RANDOM-K NULL {label} ({len(vals)} seeds), metric = {metric} ===")
    print(f"  Null distribution: mean {vals.mean():.3f}, sd {vals.std(ddof=1):.3f}")
    for q in (5, 25, 50, 75, 95, 99):
        print(f"    p{q:<2d} {np.percentile(vals, q): .3f}")
    print(f"  Strategy (real signal): {real_v:.3f}")
    print(f"  --> percentile of null: {pct:.1f}")

    # Turnover parity is a precondition for the test to mean anything.
    if real_turnover is not None:
        ratio = null_to / real_turnover if real_turnover > 0 else np.inf
        print(f"  Turnover: null {null_to:.1f}x/yr vs strategy {real_turnover:.1f}x/yr "
              f"(ratio {ratio:.1f}x)")
        if ratio > 1.5:
            print("  *** CONFOUNDED: the null churns far more than the strategy, so "
                  "it pays much higher costs. This comparison credits the signal for "
                  "a turnover advantage. Use the turnover-matched (sticky) null and "
                  "the zero-cost run below. ***")

    if pct >= 95:
        verdict = "Signal is doing real work (beats 95% of random draws)."
    elif pct <= 60:
        verdict = ("Signal NOT distinguishable from random selection. The result "
                   "is a concentration/leverage expression, not stock picking.")
    else:
        verdict = ("Weak positive edge - better than random but inside the noise "
                   "band. Size expectations down accordingly.")
    print(f"  VERDICT: {verdict}")


# ---------------------------------------------------------------- sensitivity
def one_at_a_time(prices, bench, base: dict, axes: dict, cost_bps: float = 10.0,
                  warmup: int = None) -> pd.DataFrame:
    """Vary one parameter at a time from `base`; all configs share one warmup."""
    warmup = warmup if warmup is not None else grid_warmup(axes, base)
    rows = []
    for param, values in axes.items():
        for v in values:
            cfg = dict(base)
            cfg[param] = v
            rets, turn = _run(prices, bench, cfg, cost_bps, warmup)
            m = performance_metrics(rets.loc[rets.ne(0).idxmax():])
            m.update(param=param, value=str(v), turnover_ann=float(turn.mean() * 52),
                     is_base=(v == base[param]))
            rows.append(m)
            print(f"  {param}={v}: Sharpe {m['Sharpe']:.3f}, CAGR {m['CAGR']:.3f}, "
                  f"turnover {m['turnover_ann']:.1f}x")
    return pd.DataFrame(rows)


def rebalance_weekday_sweep(prices, bench, base: dict, cost_bps: float = 10.0,
                            warmup: int = None) -> pd.DataFrame:
    """
    Rebalance on each weekday. THE most diagnostic robustness test: a real edge
    is not day-of-week specific. Wide dispersion here means artifact.
    """
    warmup = warmup if warmup is not None else grid_warmup({}, base)
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    rows = []
    for wd, name in names.items():
        dates = weekly_rebalance_dates_on(prices.index, wd)
        rets, turn = _run(prices, bench, base, cost_bps, warmup, rebal_dates=dates)
        m = performance_metrics(rets.loc[rets.ne(0).idxmax():])
        m.update(weekday=name, turnover_ann=float(turn.mean() * 52))
        rows.append(m)
        print(f"  {name}: Sharpe {m['Sharpe']:.3f}, CAGR {m['CAGR']:.3f}")
    return pd.DataFrame(rows)


def summarize(oat: pd.DataFrame, weekday: pd.DataFrame, base_metrics: dict,
              ew_sharpe: float) -> None:
    print("\n=== PARAMETER SENSITIVITY SUMMARY ===")
    print(f"Base config Sharpe: {base_metrics['Sharpe']:.3f} "
          f"(equal-weight universe: {ew_sharpe:.3f})")
    s = oat["Sharpe"].dropna()
    print(f"Across all {len(s)} one-at-a-time configs:")
    print(f"  median {s.median():.3f} | p10 {s.quantile(0.10):.3f} | "
          f"min {s.min():.3f} | max {s.max():.3f}")
    print(f"  fraction beating equal-weight: {(s > ew_sharpe).mean() * 100:.0f}%")

    print("\nPer-parameter Sharpe range (fragility - a wide range means the "
          "result depends on that knob):")
    for param, grp in oat.groupby("param"):
        v = grp["Sharpe"].dropna()
        best = grp.loc[grp["Sharpe"].idxmax()]
        print(f"  {param:14s} min {v.min(): .3f}  max {v.max(): .3f}  "
              f"spread {v.max() - v.min(): .3f}   (best at {param}={best['value']})")

    w = weekday["Sharpe"].dropna()
    print(f"\nRebalance-weekday dispersion: min {w.min():.3f}, max {w.max():.3f}, "
          f"spread {w.max() - w.min():.3f}")
    if w.max() - w.min() > 0.35:
        print("  WARNING: large day-of-week dispersion - suggests the result is "
              "partly an artifact of the rebalance calendar, not a real edge.")
    else:
        print("  OK: result is broadly stable across rebalance weekdays.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--null-test", action="store_true", help="Run the random-k null test.")
    p.add_argument("--oat", action="store_true", help="Run one-at-a-time sensitivity sweeps.")
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--replace-n", type=int, default=2,
                   help="Names swapped per week in the turnover-matched null.")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--cache", default=None, help="Path to cache the price panel.")
    args = p.parse_args()
    if not (args.null_test or args.oat):
        p.error("pick at least one of --null-test / --oat")

    prices, bench = load_panel(args.years, args.tickers, args.cache)
    warmup = grid_warmup(AXES, BASE)
    print(f"Shared warm-up pinned at {warmup} rows so every config is scored "
          f"over an identical window.")

    # Baseline (real signal) over the same warmup
    base_rets, base_turn = _run(prices, bench, BASE, args.cost_bps, warmup)
    base_rets = base_rets.loc[base_rets.ne(0).idxmax():]
    base_metrics = performance_metrics(base_rets)
    window = base_rets.index
    ew = prices.pct_change().mean(axis=1).reindex(window).fillna(0.0)
    ew_sharpe = performance_metrics(ew)["Sharpe"]
    print(f"\nBaseline {window[0].date()} -> {window[-1].date()}: "
          f"Sharpe {base_metrics['Sharpe']:.3f}, CAGR {base_metrics['CAGR']:.3f}, "
          f"turnover {base_turn.mean() * 52:.1f}x | equal-weight Sharpe {ew_sharpe:.3f}")
    print(f"Worst rolling 3y Sharpe: {rolling_3y_min_sharpe(base_rets, window):.3f}")

    if args.null_test:
        real_to = float(base_turn.mean() * 52)

        # (1) Naive i.i.d. null - kept for reference, but it churns the whole
        #     book weekly and is therefore confounded by transaction costs.
        print(f"\n[1/3] i.i.d. random null (NOT turnover-matched), "
              f"{args.n_seeds} seeds at {args.cost_bps} bps...")
        iid = random_k_null(prices, bench, BASE, args.n_seeds, args.cost_bps,
                            warmup, mode="iid")
        report_null(iid, base_metrics, "Sharpe", "[i.i.d., costed]", real_to)

        # (2) Same thing at ZERO cost, which removes the confound entirely.
        print(f"\n[2/3] i.i.d. random null at ZERO cost (removes the cost "
              f"confound), {args.n_seeds} seeds...")
        zr, zt = _run(prices, bench, BASE, 0.0, warmup)
        zr = zr.loc[zr.ne(0).idxmax():]
        zero_real = performance_metrics(zr)
        iid0 = random_k_null(prices, bench, BASE, args.n_seeds, 0.0, warmup, mode="iid")
        print(f"  (strategy at 0 bps: Sharpe {zero_real['Sharpe']:.3f}, "
              f"CAGR {zero_real['CAGR']:.3f})")
        report_null(iid0, zero_real, "Sharpe", "[i.i.d., zero-cost]",
                    float(zt.mean() * 52))

        # (3) Turnover-matched null: the honest test.
        print(f"\n[3/3] TURNOVER-MATCHED (sticky) random null, {args.n_seeds} "
              f"seeds at {args.cost_bps} bps...")
        sticky = random_k_null(prices, bench, BASE, args.n_seeds, args.cost_bps,
                               warmup, mode="sticky", replace_n=args.replace_n)
        report_null(sticky, base_metrics, "Sharpe", "[sticky, turnover-matched]", real_to)
        report_null(sticky, base_metrics, "CAGR", "[sticky, turnover-matched]", real_to)

    if args.oat:
        print("\nRunning one-at-a-time sweeps...")
        oat = one_at_a_time(prices, bench, BASE, AXES, args.cost_bps, warmup)
        print("\nRunning rebalance-weekday sweep...")
        wd = rebalance_weekday_sweep(prices, bench, BASE, args.cost_bps, warmup)
        summarize(oat, wd, base_metrics, ew_sharpe)

    print("\nNot investment advice; past performance does not guarantee future results.")


if __name__ == "__main__":
    main()
