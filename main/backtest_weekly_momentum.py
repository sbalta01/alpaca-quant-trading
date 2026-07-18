# main/backtest_weekly_momentum.py
"""
Walk-forward backtest of the weekly cross-sectional momentum portfolio,
plus the portfolio it would hold if deployed TODAY.

Usage (from the repo root, with your .venv active):

    python main/backtest_weekly_momentum.py                       # NASDAQ-100, 10y, zero cost
    python main/backtest_weekly_momentum.py --years 8 --top-k 12
    python main/backtest_weekly_momentum.py --cost-bps 10         # realistic-cost variant
    python main/backtest_weekly_momentum.py --tickers AAPL MSFT NVDA AMZN GOOG META AVGO TSLA

Notes
-----
* Signals only ever use data up to each rebalance date (walk-forward), so the
  whole backtest period is out-of-sample with respect to the rules.
* CAVEAT - survivorship bias: using TODAY'S index membership for the whole
  history overstates returns (dropped losers are excluded). Treat absolute
  numbers as optimistic; relative comparisons vs the same universe's
  equal-weight benchmark are the fairer read.
* Default cost_bps=0 (per request). Run with --cost-bps 10 before believing
  anything.
"""
import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.strategies.weekly_momentum import (
    compute_target_weights, performance_metrics, run_walkforward,
)


def fetch_close_matrix(tickers, start, end) -> pd.DataFrame:
    """Daily adjusted closes, wide format, via yfinance."""
    import yfinance as yf
    df = yf.download(
        tickers=" ".join(tickers), start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"), interval="1d", group_by="ticker",
        auto_adjust=True, threads=True, progress=False,
    )
    closes = {}
    for t in tickers:
        try:
            s = df[t]["Close"].dropna() if isinstance(df.columns, pd.MultiIndex) else df["Close"].dropna()
            if len(s) > 0:
                closes[t] = s
        except KeyError:
            print(f"  (no data for {t}, skipping)")
    out = pd.DataFrame(closes).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Universe. Default: NASDAQ-100 (survivorship-biased - see caveat).")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=0.0, help="Cost per unit turnover, bps.")
    p.add_argument("--weight-cap", type=float, default=0.20)
    p.add_argument("--low-exposure", type=float, default=0.4,
                   help="Gross exposure when SPY < 200dma.")
    p.add_argument("--no-regime-gate", action="store_true")
    p.add_argument("--buffer-mult", type=float, default=1.5,
                   help="Rank-buffer width (matches the live default). "
                        "1.0 = no buffering (pre-2026-07 behavior).")
    p.add_argument("--min-trade-fraction", type=float, default=0.005,
                   help="No-trade band, mirroring the live executor. 0 disables.")
    args = p.parse_args()

    if args.tickers:
        universe = args.tickers
    else:
        from src.data.data_loader import fetch_nasdaq_100_symbols
        universe = fetch_nasdaq_100_symbols()
    print(f"Universe: {len(universe)} tickers | top_k={args.top_k} | "
          f"cost={args.cost_bps} bps | regime gate={'off' if args.no_regime_gate else 'on'} | "
          f"buffer={args.buffer_mult}x | band={args.min_trade_fraction:.3f}")

    end = datetime.now()
    start = end - timedelta(days=int(args.years * 365.25) + 400)  # +400d for warm-up
    print("Downloading daily closes...")
    prices = fetch_close_matrix(sorted(set(universe)), start, end)
    bench = fetch_close_matrix(["SPY", "QQQ"], start, end)
    # keep tickers with enough history to ever be scored
    # No global "has enough history" filter: that used whole-sample data
    # availability to decide inclusion, which silently erased recent index
    # additions. Per-date eligibility is already handled by momentum_scores
    # returning NaN until the lookback is computable.
    # Bounded ffill so a dead ticker drops out instead of holding a flat price.
    prices = prices.reindex(bench.index).ffill(limit=5)
    print(f"Got {prices.shape[1]} usable tickers, {prices.shape[0]} days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    low_expo = 1.0 if args.no_regime_gate else args.low_exposure
    from src.strategies.weekly_momentum import BufferedSelector
    res = run_walkforward(
        prices, bench["SPY"], top_k=args.top_k, weight_cap=args.weight_cap,
        low_exposure=low_expo, cost_bps=args.cost_bps,
        selector_factory=lambda: BufferedSelector(args.buffer_mult),
        min_trade_fraction=args.min_trade_fraction,
    )
    r = res["daily_returns"]
    r = r.loc[r.ne(0).idxmax():]  # trim pre-warmup zeros
    period = f"{r.index[0].date()} -> {r.index[-1].date()}"

    # Benchmarks over the same window
    spy_r = bench["SPY"].pct_change().reindex(r.index).fillna(0.0)
    qqq_r = bench["QQQ"].pct_change().reindex(r.index).fillna(0.0)
    ew_r = prices.pct_change().mean(axis=1).reindex(r.index).fillna(0.0)

    rows = {
        "Strategy": performance_metrics(r),
        "Equal-weight universe": performance_metrics(ew_r),
        "QQQ buy&hold": performance_metrics(qqq_r),
        "SPY buy&hold": performance_metrics(spy_r),
    }
    table = pd.DataFrame(rows).T
    pd.set_option("display.float_format", lambda x: f"{x: .3f}")
    print(f"\n=== Walk-forward results, weekly rebalance, {period} ===")
    print(table.to_string())
    ann_turnover = res["turnover"].mean() * 52
    print(f"\nAvg gross exposure : {res['exposure'].mean():.2f}")
    print(f"Annualized turnover: {ann_turnover:.1f}x  "
          f"(at 10 bps this would cost ~{ann_turnover * 10 / 100:.1f}%/yr)")

    # Sub-period sanity: yearly returns
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    yearly_qqq = (1 + qqq_r).groupby(qqq_r.index.year).prod() - 1
    yr = pd.DataFrame({"Strategy": yearly, "QQQ": yearly_qqq})
    print("\nYearly returns:")
    print(yr.to_string())

    # What you would hold if deployed today (buffered against the walk-forward's
    # final book, the same way live buffers against actual account positions)
    sel_today = BufferedSelector(args.buffer_mult)
    sel_today.held = list(res["weights"][max(res["weights"])].index)
    w_today = compute_target_weights(prices, bench["SPY"], top_k=args.top_k,
                                     weight_cap=args.weight_cap, low_exposure=low_expo,
                                     selector=sel_today)
    print(f"\n=== Target portfolio if deployed today ({prices.index[-1].date()}) ===")
    print((w_today * 100).round(2).to_string())
    print(f"Cash: {(1 - w_today.sum()) * 100:.2f}%")
    print("\nNot investment advice; past performance does not guarantee future results.")


if __name__ == "__main__":
    main()
