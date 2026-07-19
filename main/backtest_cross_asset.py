# main/backtest_cross_asset.py
"""
Cross-asset trend/momentum sleeve: the diversification answer to "what if the
regime changes?" - instead of predicting the turn (which the holistic
experiment showed we can't), hold whatever asset CLASSES are trending, across
instruments that thrive in different regimes.

Reuses the weekly momentum machinery unchanged (momentum scores -> top-k ->
inverse-vol -> exposure) on a small ETF universe:

    SPY  US large-cap equity        TLT  20y+ Treasuries
    QQQ  US growth/tech             IEF  7-10y Treasuries
    VEA  Intl developed equity      GLD  gold
    EEM  Emerging markets           DBC  broad commodities
    VNQ  US REITs

Risk control is VOL TARGETING (default 10% ann - balanced-portfolio class),
not the SPY 200dma gate: when equities roll over, the right response here is
rotating into bonds/gold, not going to cash on an equity signal.

Benchmarks over the identical window: 60/40 (SPY/IEF, daily-rebalanced),
equal-weight universe, SPY. Also prints the sleeve's correlation to SPY and
QQQ - the point of the sleeve is a LOW number there.

Usage (from the repo root, with .venv active):

    python main/backtest_cross_asset.py                  # zero cost
    python main/backtest_cross_asset.py --cost-bps 5     # realistic ETF costs
    python main/backtest_cross_asset.py --no-vol-target  # ablation
    python main/backtest_cross_asset.py --target-vol 0.12 --top-k 5
"""
import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.strategies.weekly_momentum import (
    BufferedSelector, compute_target_weights, make_vol_target_exposure,
    performance_metrics, run_walkforward,
)
from main.backtest_weekly_momentum import fetch_close_matrix

DEFAULT_UNIVERSE = ["SPY", "QQQ", "VEA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--years", type=int, default=18)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--cost-bps", type=float, default=0.0)
    p.add_argument("--weight-cap", type=float, default=0.35)
    p.add_argument("--buffer-mult", type=float, default=1.5)
    p.add_argument("--target-vol", type=float, default=0.10,
                   help="Annualized portfolio vol target (0.10 = balanced-class).")
    p.add_argument("--no-vol-target", action="store_true")
    p.add_argument("--use-dma-gate", action="store_true",
                   help="Also apply the SPY 200dma gate (off by default here: "
                        "bonds/gold ARE the defensive asset).")
    args = p.parse_args()

    end = datetime.now()
    start = end - timedelta(days=int(args.years * 365.25) + 400)
    print(f"Universe: {args.tickers} | top_k={args.top_k} | cost={args.cost_bps} bps | "
          f"vol target={'off' if args.no_vol_target else args.target_vol} | "
          f"dma gate={'on' if args.use_dma_gate else 'off'}")
    print("Downloading daily closes...")
    prices = fetch_close_matrix(sorted(set(args.tickers)), start, end)
    bench = fetch_close_matrix(["SPY", "QQQ", "IEF"], start, end)
    prices = prices.reindex(bench.index).ffill(limit=5)
    print(f"Got {prices.shape[1]} ETFs, {prices.shape[0]} days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    low_expo = 0.4 if args.use_dma_gate else 1.0
    exposure_fn = None if args.no_vol_target else make_vol_target_exposure(
        target_vol=args.target_vol, with_regime_gate=args.use_dma_gate,
        low_exposure=low_expo)

    res = run_walkforward(
        prices, bench["SPY"], top_k=args.top_k, weight_cap=args.weight_cap,
        low_exposure=low_expo, cost_bps=args.cost_bps,
        selector_factory=lambda: BufferedSelector(args.buffer_mult),
        min_trade_fraction=0.005, exposure_fn=exposure_fn,
    )
    r = res["daily_returns"]
    r = r.loc[r.ne(0).idxmax():]
    window = r.index
    period = f"{window[0].date()} -> {window[-1].date()}"

    spy_r = bench["SPY"].pct_change(fill_method=None).reindex(window).fillna(0.0)
    qqq_r = bench["QQQ"].pct_change(fill_method=None).reindex(window).fillna(0.0)
    ief_r = bench["IEF"].pct_change(fill_method=None).reindex(window).fillna(0.0)
    sixty_forty = 0.6 * spy_r + 0.4 * ief_r
    ew_r = prices.pct_change(fill_method=None).mean(axis=1).reindex(window).fillna(0.0)

    rows = {
        "Cross-asset trend sleeve": performance_metrics(r),
        "60/40 SPY/IEF": performance_metrics(sixty_forty),
        "Equal-weight universe": performance_metrics(ew_r),
        "SPY buy&hold": performance_metrics(spy_r),
    }
    pd.set_option("display.float_format", lambda x: f"{x: .3f}")
    print(f"\n=== Walk-forward results, weekly rebalance, {period} ===")
    print(pd.DataFrame(rows).T.to_string())

    ann_to = res["turnover"].mean() * 52
    print(f"\nAvg gross exposure : {res['exposure'].mean():.2f}")
    print(f"Annualized turnover: {ann_to:.1f}x "
          f"(at 5 bps this would cost ~{ann_to * 5 / 100:.2f}%/yr)")
    print(f"Correlation to SPY : {r.corr(spy_r):.2f}   (diversification wants LOW)")
    print(f"Correlation to QQQ : {r.corr(qqq_r):.2f}   (proxy for the momentum sleeve)")

    yearly = pd.DataFrame({
        "Sleeve": (1 + r).groupby(window.year).prod() - 1,
        "60/40": (1 + sixty_forty).groupby(window.year).prod() - 1,
        "SPY": (1 + spy_r).groupby(window.year).prod() - 1,
    })
    print("\nYearly returns:")
    print(yearly.to_string())

    sel_today = BufferedSelector(args.buffer_mult)
    last_w = res["weights"][max(res["weights"])]
    sel_today.held = list(last_w.index)
    w_today = compute_target_weights(
        prices, bench["SPY"], top_k=args.top_k, weight_cap=args.weight_cap,
        low_exposure=low_expo, selector=sel_today, exposure_fn=exposure_fn)
    print(f"\n=== Target sleeve if deployed today ({prices.index[-1].date()}) ===")
    print((w_today * 100).round(2).to_string())
    print(f"Cash: {(1 - w_today.sum()) * 100:.2f}%")
    print("\nNot investment advice; past performance does not guarantee future results.")


if __name__ == "__main__":
    main()
