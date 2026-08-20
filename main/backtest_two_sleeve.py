# main/backtest_two_sleeve.py
"""
Walk-forward backtest of the TWO-SLEEVE portfolio: the concentrated momentum
book plus an uncorrelated diversifier sleeve, blended at a fixed capital split.

Usage (from the repo root, with your .venv active):

    python main/backtest_two_sleeve.py --years 9 --cost-bps 10
    python main/backtest_two_sleeve.py --split 0.70 --sweep
    python main/backtest_two_sleeve.py --cache .cache_two_sleeve.pkl   # reuse prices

Why this exists
---------------
Vol targeting sizes the momentum book against its own realized vol, which for a
10-name semiconductor cluster is ~70% annualized - so gross exposure lands near
0.25 and most of the account sits in cash. Adding a low-correlation sleeve puts
that cash to work without going deeper into the momentum ranking (which is pure
alpha dilution: Sharpe 1.01 at k=10 -> 0.79 at k=40).

Notes
-----
* Both sleeves are walk-forward and share one trading calendar, so the blend is
  computed over an identical window for every split.
* CAVEAT - the momentum sleeve is survivorship-flattered (today's S&P 500
  membership applied to all history); the ETF sleeve is not. The bias therefore
  works AGAINST the blend, which makes a blend-beats-momentum result credible
  rather than suspect.
"""
import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.strategies.sleeves import (DIVERSIFIER, MOMENTUM, SLEEVES, SleeveConfig,
                                    validate_sleeves)
from src.strategies.weekly_momentum import (BufferedSelector, make_vol_target_exposure,
                                            performance_metrics, run_walkforward)
from main.backtest_weekly_momentum import fetch_close_matrix


def sleeve_returns(prices: pd.DataFrame, bench: pd.Series, sleeve: SleeveConfig,
                   cost_bps: float = 10.0, warmup: int = None) -> dict:
    """
    Walk-forward one sleeve on its own panel, using its own risk settings.

    Thin wrapper over run_walkforward so both sleeves are guaranteed to use the
    same engine, the same no-trade band and the same buffering as live.
    """
    exposure_fn = None
    if sleeve.target_vol and sleeve.target_vol > 0:
        exposure_fn = make_vol_target_exposure(
            target_vol=sleeve.target_vol, with_regime_gate=sleeve.use_dma_gate,
            low_exposure=sleeve.low_exposure)
    return run_walkforward(
        prices, bench, top_k=sleeve.top_k, weight_cap=sleeve.weight_cap,
        low_exposure=sleeve.low_exposure, cost_bps=cost_bps, warmup=warmup,
        selector_factory=lambda: BufferedSelector(sleeve.buffer_mult),
        min_trade_fraction=sleeve.min_trade_fraction, exposure_fn=exposure_fn,
    )


def align(returns: dict) -> dict:
    """
    Restrict every sleeve's return series to the common window, starting at the
    first date on which ALL sleeves are live. Without this the blend would
    silently compare different periods as the split changes.
    """
    idx = None
    for r in returns.values():
        idx = r.index if idx is None else idx.intersection(r.index)
    start = max(r.loc[idx].ne(0).idxmax() for r in returns.values())
    return {k: r.loc[idx].loc[start:] for k, r in returns.items()}


def blend(returns: dict, allocs: dict, rebalance: str = "daily") -> pd.Series:
    """
    Blend aligned sleeve returns at the given capital split.

    rebalance="daily"  : constant weights, i.e. the split is restored every day.
                         This is the standard comparison and the cheapest to reason
                         about.
    rebalance="weekly" : track each sleeve's own equity curve and reset the split
                         only on the weekly schedule - which is what the live
                         deployer actually does, since it rebalances on Fridays and
                         lets the sleeves drift in between.

    The gap between the two is the live/backtest mismatch. Report both rather than
    assuming it is second-order.
    """
    names = list(returns)
    if rebalance == "daily":
        return sum(returns[n] * allocs[n] for n in names)

    if rebalance != "weekly":
        raise ValueError(f"rebalance must be 'daily' or 'weekly', got {rebalance!r}")

    idx = returns[names[0]].index
    iso = idx.isocalendar()
    week_id = pd.Series(list(zip(iso.year.values, iso.week.values)), index=idx)
    reset = week_id != week_id.shift(1)          # first trading day of each ISO week

    value = {n: allocs[n] for n in names}        # sleeve capital, total normalized to 1
    out = pd.Series(0.0, index=idx)
    for date in idx:
        if reset.loc[date]:                      # restore the split
            total = sum(value.values())
            value = {n: total * allocs[n] for n in names}
        before = sum(value.values())
        for n in names:
            value[n] *= (1.0 + returns[n].loc[date])
        after = sum(value.values())
        out.loc[date] = after / before - 1.0 if before > 0 else 0.0
    return out


def _row(label: str, r: pd.Series, extra: dict = None) -> dict:
    m = performance_metrics(r)
    row = {"CAGR": m["CAGR"], "Vol (ann)": m["Vol (ann)"], "Sharpe": m["Sharpe"],
           "Max Drawdown": m["Max Drawdown"], "Calmar": m["Calmar"]}
    row.update(extra or {})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=9)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--split", type=float, default=None,
                   help="Momentum allocation (default: the configured sleeve split).")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep the split 0..100%% to show the Sharpe plateau.")
    p.add_argument("--weekly-blend", action="store_true",
                   help="Also report the weekly-reset blend (what live does).")
    p.add_argument("--cache", type=str, default=None,
                   help="Pickle path to cache the price panels between runs.")
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500")
    args = p.parse_args()

    validate_sleeves(SLEEVES)
    mom_alloc = args.split if args.split is not None else MOMENTUM.allocation
    div_alloc = 1.0 - mom_alloc
    allocs = {MOMENTUM.name: mom_alloc, DIVERSIFIER.name: div_alloc}

    # ---- data -------------------------------------------------------------
    panels = None
    if args.cache:
        try:
            panels = pd.read_pickle(args.cache)
            print(f"Loaded price panels from {args.cache}")
        except FileNotFoundError:
            panels = None
    if panels is None:
        now = datetime.now()
        start = now - timedelta(days=args.years * 365.25 + 400)   # + warm-up
        if args.universe == "sp500":
            from src.data.universe import fetch_sp500_symbols
            universe = fetch_sp500_symbols()
        else:
            from src.data.universe import fetch_nasdaq_100_symbols
            universe = fetch_nasdaq_100_symbols()
        stocks = fetch_close_matrix(sorted(set(universe)), start, now)
        bench = fetch_close_matrix(["SPY"], start, now)
        etfs = fetch_close_matrix(list(DIVERSIFIER.universe), start, now)
        panels = {"stocks": stocks, "bench": bench, "etfs": etfs}
        if args.cache:
            pd.to_pickle(panels, args.cache)
            print(f"Cached price panels to {args.cache}")

    bench = panels["bench"]
    # One shared trading calendar for both sleeves.
    stocks = panels["stocks"].reindex(bench.index).ffill(limit=5)
    stocks = stocks.drop(columns=stocks.columns[stocks.iloc[-1].isna()])
    etfs = panels["etfs"].reindex(bench.index).ffill(limit=5)
    etfs = etfs.drop(columns=etfs.columns[etfs.iloc[-1].isna()])
    spy = bench["SPY"]
    print(f"Panel: {stocks.shape[1]} stocks, {etfs.shape[1]} ETFs "
          f"({list(etfs.columns)}), {stocks.index[0].date()} -> {stocks.index[-1].date()}")

    # ---- sleeves ----------------------------------------------------------
    res = {MOMENTUM.name: sleeve_returns(stocks, spy, MOMENTUM, args.cost_bps),
           DIVERSIFIER.name: sleeve_returns(etfs, spy, DIVERSIFIER, args.cost_bps)}
    rets = align({k: v["daily_returns"] for k, v in res.items()})
    window = f"{rets[MOMENTUM.name].index[0].date()} -> {rets[MOMENTUM.name].index[-1].date()}"

    blended = blend(rets, allocs, "daily")
    spy_r = spy.pct_change().reindex(blended.index).fillna(0.0)

    rows = {
        f"momentum only": _row("m", rets[MOMENTUM.name]),
        f"diversifier only": _row("d", rets[DIVERSIFIER.name]),
        f"BLEND {mom_alloc:.0%}/{div_alloc:.0%}": _row("b", blended),
        "SPY buy&hold": _row("s", spy_r),
    }
    print(f"\n=== Two-sleeve walk-forward ({window}), {args.cost_bps:.0f} bps ===")
    df = pd.DataFrame(rows).T
    print(df.to_string(formatters={
        "CAGR": "{:.1%}".format, "Vol (ann)": "{:.1%}".format,
        "Sharpe": "{:.2f}".format, "Max Drawdown": "{:.1%}".format,
        "Calmar": "{:.2f}".format}))

    corr = rets[MOMENTUM.name].corr(rets[DIVERSIFIER.name])
    invested = sum(res[n]["exposure"].mean() * allocs[n] for n in allocs)
    print(f"\nSleeve correlation      : {corr:.2f}   (diversification wants LOW)")
    print(f"Avg gross exposure      : momentum {res[MOMENTUM.name]['exposure'].mean():.2f}, "
          f"diversifier {res[DIVERSIFIER.name]['exposure'].mean():.2f}")
    print(f"Blended invested capital: {invested:.2f}  "
          f"(vs {res[MOMENTUM.name]['exposure'].mean():.2f} momentum-only)")
    for n in allocs:
        print(f"Turnover {n:<12}: {res[n]['turnover'].mean() * 52:.1f}x/yr")

    if args.weekly_blend:
        wk = blend(rets, allocs, "weekly")
        mw = performance_metrics(wk)
        md = performance_metrics(blended)
        print(f"\nBlend rebalance sensitivity (live resets weekly, not daily):")
        print(f"  daily  reset: Sharpe {md['Sharpe']:.3f}  CAGR {md['CAGR']:.2%}  "
              f"MaxDD {md['Max Drawdown']:.2%}")
        print(f"  weekly reset: Sharpe {mw['Sharpe']:.3f}  CAGR {mw['CAGR']:.2%}  "
              f"MaxDD {mw['Max Drawdown']:.2%}")

    if args.sweep:
        print(f"\n=== Split sweep (momentum allocation) ===")
        print(f"{'mom%':>6}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'invested':>10}")
        for a in np.arange(0.0, 1.01, 0.1):
            al = {MOMENTUM.name: a, DIVERSIFIER.name: 1 - a}
            m = performance_metrics(blend(rets, al, "daily"))
            inv = sum(res[n]["exposure"].mean() * al[n] for n in al)
            print(f"{a:>6.0%}{m['CAGR']:>8.1%}{m['Vol (ann)']:>8.1%}{m['Sharpe']:>8.2f}"
                  f"{m['Max Drawdown']:>9.1%}{m['Calmar']:>8.2f}{inv:>10.2f}")

    # ---- yearly ------------------------------------------------------------
    yearly = pd.DataFrame({
        "Blend": blended.groupby(blended.index.year).apply(lambda r: (1 + r).prod() - 1),
        "Momentum": rets[MOMENTUM.name].groupby(rets[MOMENTUM.name].index.year).apply(
            lambda r: (1 + r).prod() - 1),
        "SPY": spy_r.groupby(spy_r.index.year).apply(lambda r: (1 + r).prod() - 1),
    })
    print(f"\n=== Yearly returns ===")
    print(yearly.to_string(formatters={c: "{:.1%}".format for c in yearly.columns}))


if __name__ == "__main__":
    main()
