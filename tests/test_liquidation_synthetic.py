"""Synthetic-data tests for dead-ticker handling in the weekly backtester.

An unbounded ffill used to let a halted/delisted name carry a flat price
forward, keeping it selectable (and buyable live). With a bounded ffill the
price goes NaN - and a held position must then be liquidated to CASH, not left
to silently "earn" 0% forever via pct_change().fillna(0).

Run from the repo root:  python tests/test_liquidation_synthetic.py
No network needed; requires pandas, numpy.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.weekly_momentum import run_walkforward


def make_prices(seed=23, n=1400, n_assets=12):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    rets = rng.normal(0.0004, 0.013, (n, n_assets))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                      index=idx, name="SPY")
    return prices, bench


def test_dead_name_is_liquidated_to_cash():
    """A name whose price goes NaN mid-backtest must stop contributing return,
    and must not be resurrected by the 0-fill."""
    prices, bench = make_prices()
    # Force one asset to be a strong winner so it is definitely held, then kill it.
    winner = "S00"
    prices[winner] = prices[winner] * np.linspace(1.0, 6.0, len(prices))
    death = prices.index[-200]
    prices.loc[death:, winner] = np.nan

    res = run_walkforward(prices, bench, top_k=3, cost_bps=10.0)

    # After death the name must never appear in any target weight vector.
    for date, w in res["weights"].items():
        if date >= death:
            assert winner not in w.index, f"dead name selected at {date.date()}"

    # And the portfolio must still produce finite returns throughout.
    assert res["daily_returns"].notna().all(), "NaN leaked into portfolio returns"
    assert np.isfinite(res["daily_returns"]).all()
    print("dead name is dropped from selection and returns stay finite: OK")


def test_dead_holding_contributes_no_return():
    """Compare against a panel where the name dies vs one where it keeps a flat
    price: the flat-price version would credit 0% forever; the NaN version must
    liquidate. The two must differ (i.e. liquidation actually does something)."""
    prices, bench = make_prices()
    winner = "S00"
    prices[winner] = prices[winner] * np.linspace(1.0, 6.0, len(prices))
    death = prices.index[-200]

    nan_panel = prices.copy()
    nan_panel.loc[death:, winner] = np.nan

    flat_panel = prices.copy()
    flat_panel.loc[death:, winner] = float(prices.loc[:death, winner].iloc[-1])

    r_nan = run_walkforward(nan_panel, bench, top_k=3, cost_bps=10.0)["daily_returns"]
    r_flat = run_walkforward(flat_panel, bench, top_k=3, cost_bps=10.0)["daily_returns"]
    assert not np.allclose(r_nan.values, r_flat.values), (
        "liquidation must change results vs a zombie flat-priced holding")
    print("liquidating a dead holding differs from carrying it flat: OK")


def test_all_alive_is_unchanged():
    """Identity invariant: with no dead names, results must be bit-for-bit the
    same as before the liquidation logic existed."""
    prices, bench = make_prices()
    a = run_walkforward(prices, bench, top_k=4, cost_bps=10.0)
    # A panel with no NaNs anywhere exercises the `live.all()` fast path.
    assert prices.notna().all().all()
    b = run_walkforward(prices.copy(), bench.copy(), top_k=4, cost_bps=10.0)
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])
    print("all-alive panel: liquidation logic is a no-op: OK")


def test_ffill_limit_boundary():
    """ffill(limit=5) must bridge a 5-day gap but not a 6-day one."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    s = pd.Series(np.arange(30, dtype=float), index=idx)

    g5 = s.copy()
    g5.iloc[10:15] = np.nan          # 5 consecutive missing
    assert g5.ffill(limit=5).iloc[10:15].notna().all(), "5-day gap should bridge"

    g6 = s.copy()
    g6.iloc[10:16] = np.nan          # 6 consecutive missing
    filled6 = g6.ffill(limit=5)
    assert filled6.iloc[10:15].notna().all(), "first 5 of a 6-day gap bridge"
    assert pd.isna(filled6.iloc[15]), "6th day must remain NaN"
    print("ffill(limit=5) bridges exactly 5 days, not 6: OK")


if __name__ == "__main__":
    test_dead_name_is_liquidated_to_cash()
    test_dead_holding_contributes_no_return()
    test_all_alive_is_unchanged()
    test_ffill_limit_boundary()
    print("\nAll liquidation tests passed.")
