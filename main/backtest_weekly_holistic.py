# main/backtest_weekly_holistic.py
"""
Layer-by-layer comparison of the holistic weekly method (review section 4)
against the original weekly momentum method, on identical data.

Variants (each layer admitted only if it improves out-of-sample results):
    Original     : momentum + inverse-vol + 200dma gate  (what is deployed)
    +Reversal    : adds layer 2 (5-day reversal, negative weight)
    +ML ranker   : adds layer 3 (pooled cross-sectional XGBoost rank model)
    Holistic     : adds layer 4 (HMM + turbulence + 200dma gate, {0.3,0.65,1.0})

Usage (from the repo root, with .venv active):

    python main/backtest_weekly_holistic.py                    # NASDAQ-100, 10y, zero cost
    python main/backtest_weekly_holistic.py --cost-bps 10      # run this too
    python main/backtest_weekly_holistic.py --years 8 --top-k 12
    python main/backtest_weekly_holistic.py --tickers AAPL MSFT NVDA ...

Same caveat as backtest_weekly_momentum.py: today's index membership means
survivorship bias; the equal-weight universe row is the fairest benchmark.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, ".")

from src.strategies.weekly_momentum import performance_metrics
from src.strategies.weekly_holistic import run_walkforward_holistic
from main.backtest_weekly_momentum import fetch_close_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=0.0)
    p.add_argument("--weight-cap", type=float, default=0.20)
    p.add_argument("--rev-weight", type=float, default=0.25)
    p.add_argument("--ml-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.tickers:
        universe = args.tickers
    else:
        from src.data.data_loader import fetch_nasdaq_100_symbols
        universe = fetch_nasdaq_100_symbols()
    print(f"Universe: {len(universe)} tickers | top_k={args.top_k} | "
          f"cost={args.cost_bps} bps | rev_w={args.rev_weight} | ml_w={args.ml_weight}")

    end = datetime.now()
    start = end - timedelta(days=int(args.years * 365.25) + 400)
    print("Downloading daily closes...")
    prices = fetch_close_matrix(sorted(set(universe)), start, end)
    bench = fetch_close_matrix(["SPY", "QQQ"], start, end)
    prices = prices.loc[:, prices.notna().sum() > 300]
    prices = prices.reindex(bench.index).ffill()
    print(f"Got {prices.shape[1]} usable tickers, {prices.shape[0]} days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    common = dict(top_k=args.top_k, weight_cap=args.weight_cap,
                  cost_bps=args.cost_bps, random_state=args.seed,
                  rev_weight=args.rev_weight)
    variants = {
        "Original (L1+dma gate)": dict(use_reversal=False, ml_weight=0.0, gate="simple"),
        "+Reversal (L2)": dict(use_reversal=True, ml_weight=0.0, gate="simple"),
        "+ML ranker (L3)": dict(use_reversal=True, ml_weight=args.ml_weight, gate="simple"),
        "Holistic (L2+L3+L4)": dict(use_reversal=True, ml_weight=args.ml_weight, gate="hmm"),
    }

    runs = {}
    for name, kw in variants.items():
        t0 = time.perf_counter()
        runs[name] = run_walkforward_holistic(prices, bench["SPY"], **common, **kw)
        note = ""
        if runs[name]["ml_active_from"] is not None:
            note = f" (ML active from {runs[name]['ml_active_from'].date()})"
        print(f"  ran {name} in {time.perf_counter() - t0:.0f}s{note}")

    # Common evaluation window: first day any variant trades (identical warmup)
    r0 = runs["Original (L1+dma gate)"]["daily_returns"]
    start_eval = r0.ne(0).idxmax()
    window = r0.loc[start_eval:].index
    period = f"{window[0].date()} -> {window[-1].date()}"

    spy_r = bench["SPY"].pct_change().reindex(window).fillna(0.0)
    qqq_r = bench["QQQ"].pct_change().reindex(window).fillna(0.0)
    ew_r = prices.pct_change().mean(axis=1).reindex(window).fillna(0.0)

    rows = {name: performance_metrics(res["daily_returns"].reindex(window))
            for name, res in runs.items()}
    rows["Equal-weight universe"] = performance_metrics(ew_r)
    rows["QQQ buy&hold"] = performance_metrics(qqq_r)
    rows["SPY buy&hold"] = performance_metrics(spy_r)

    pd.set_option("display.float_format", lambda x: f"{x: .3f}")
    print(f"\n=== Walk-forward comparison, weekly rebalance, {period} ===")
    print(pd.DataFrame(rows).T.to_string())

    print("\nAvg gross exposure / annualized turnover:")
    for name, res in runs.items():
        to = res["turnover"].mean() * 52
        print(f"  {name:26s}: expo {res['exposure'].mean():.2f}, "
              f"turnover {to:.1f}x (~{to * 10 / 100:.1f}%/yr at 10 bps)")

    yearly = {name: (1 + res["daily_returns"].reindex(window)).groupby(window.year).prod() - 1
              for name, res in runs.items()}
    yearly["QQQ"] = (1 + qqq_r).groupby(window.year).prod() - 1
    print("\nYearly returns:")
    print(pd.DataFrame(yearly).to_string())

    holi = runs["Holistic (L2+L3+L4)"]
    last_date = max(holi["weights"])
    w_today = holi["weights"][last_date]
    print(f"\n=== Holistic target portfolio, latest rebalance ({last_date.date()}) ===")
    print((w_today * 100).round(2).to_string())
    print(f"Cash: {(1 - w_today.sum()) * 100:.2f}%")
    print("\nNot investment advice; past performance does not guarantee future results.")


if __name__ == "__main__":
    main()
