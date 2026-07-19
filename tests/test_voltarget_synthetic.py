"""Synthetic-data tests for volatility targeting (exposure_fn seam).

Run from the repo root:  python tests/test_voltarget_synthetic.py
No network needed; requires pandas, numpy.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.weekly_momentum import (
    make_vol_target_exposure, realized_portfolio_vol, run_walkforward,
    vol_target_exposure,
)


def make_prices(seed=41, n=1500, n_assets=15, vol=0.015, shift_at=None, shift_vol=0.05):
    """GBM panel; optionally switch every asset to `shift_vol` from row `shift_at`."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    sig = np.full((n, n_assets), vol)
    if shift_at is not None:
        sig[shift_at:, :] = shift_vol
    rets = rng.normal(0, 1, (n, n_assets)) * sig + 0.0003
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                      index=idx, name="SPY")
    return prices, bench


def test_identity_when_target_unreachable():
    """A huge target_vol with max_exposure=1.0 always asks for full exposure,
    so combined with the same regime gate it must reproduce the default run
    bit-for-bit - the seam is provably a no-op at its identity setting."""
    prices, bench = make_prices()
    a = run_walkforward(prices, bench, top_k=5, cost_bps=10.0)
    b = run_walkforward(prices, bench, top_k=5, cost_bps=10.0,
                        exposure_fn=make_vol_target_exposure(
                            target_vol=99.0, with_regime_gate=True))
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])
    pd.testing.assert_series_equal(a["turnover"], b["turnover"])
    print("unreachable target_vol + same gate == default run, bit-for-bit: OK")


def test_exposure_falls_after_vol_shift():
    """When every asset's vol quadruples, exposure and drawdown must both drop
    versus the untargeted run."""
    prices, bench = make_prices(shift_at=1000, shift_vol=0.06)
    base = run_walkforward(prices, bench, top_k=5, cost_bps=10.0, low_exposure=1.0)
    vt = run_walkforward(prices, bench, top_k=5, cost_bps=10.0, low_exposure=1.0,
                         exposure_fn=make_vol_target_exposure(
                             target_vol=0.20, with_regime_gate=False))
    shift_date = prices.index[1000]
    expo_before = vt["exposure"][vt["exposure"].index < shift_date].mean()
    expo_after = vt["exposure"][vt["exposure"].index >= prices.index[1030]].mean()
    assert expo_after < expo_before * 0.7, (
        f"exposure should fall after the vol shift: {expo_before:.2f} -> {expo_after:.2f}")

    def max_dd(r):
        c = (1 + r).cumprod()
        return float((c / c.cummax() - 1).min())

    r_base = base["daily_returns"].loc[shift_date:]
    r_vt = vt["daily_returns"].loc[shift_date:]
    assert max_dd(r_vt) > max_dd(r_base), (
        f"vol targeting should shrink the post-shift drawdown: "
        f"{max_dd(r_vt):.3f} vs {max_dd(r_base):.3f}")
    print(f"vol shift: exposure {expo_before:.2f} -> {expo_after:.2f}, "
          f"post-shift DD {max_dd(r_base):.1%} -> {max_dd(r_vt):.1%}: OK")


def test_causality_only_trailing_window():
    """The estimate must use only the trailing max(windows) rows: poisoning
    older history must not change the answer."""
    prices, _ = make_prices()
    w = pd.Series(0.2, index=prices.columns[:5])
    clean = vol_target_exposure(prices, w, target_vol=0.15, windows=(21, 63))
    poisoned = prices.copy()
    # 63 trailing returns need 64 trailing prices; poison everything older.
    poisoned.iloc[:-64] = 1e9
    dirty = vol_target_exposure(poisoned, w, target_vol=0.15, windows=(21, 63))
    assert abs(clean - dirty) < 1e-12, "estimate leaked outside its window"
    print("only the trailing window is used (older history poisoned, no change): OK")


def test_exposure_bounds():
    prices, _ = make_prices()
    w = pd.Series(0.2, index=prices.columns[:5])
    for tv in (0.001, 0.05, 0.2, 1.0, 99.0):
        e = vol_target_exposure(prices, w, target_vol=tv,
                                max_exposure=1.0, min_exposure=0.1)
        assert 0.1 - 1e-12 <= e <= 1.0 + 1e-12, f"exposure {e} out of bounds at tv={tv}"
    assert vol_target_exposure(prices, pd.Series(dtype=float)) == 1.0
    print("exposure always within [min_exposure, max_exposure]: OK")


def test_full_covariance_not_diagonal():
    """Two perfectly correlated assets: portfolio vol must equal single-asset
    vol (the diagonal-only shortcut would report vol/sqrt(2))."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.RandomState(7)
    r = rng.normal(0, 0.02, 300)
    a = 100 * np.exp(np.cumsum(r))
    prices = pd.DataFrame({"A": a, "B": a * 1.5}, index=idx)   # identical returns
    w = pd.Series({"A": 0.5, "B": 0.5})
    port = realized_portfolio_vol(prices, w, window=250)
    single = realized_portfolio_vol(prices, pd.Series({"A": 1.0}), window=250)
    assert abs(port - single) / single < 0.02, (
        f"correlated pair should have ~single-asset vol: {port:.4f} vs {single:.4f}")
    print(f"full covariance honored (corr pair vol {port:.3f} == single {single:.3f}): OK")


def test_regime_gate_composes():
    """With the gate on and the benchmark forced below its 200dma, exposure
    must be scaled by low_exposure on top of the vol target."""
    prices, _ = make_prices()
    down = pd.Series(np.linspace(200, 100, len(prices)), index=prices.index)
    w = pd.Series(0.2, index=prices.columns[:5])
    fn_gated = make_vol_target_exposure(target_vol=99.0, with_regime_gate=True,
                                        low_exposure=0.4)
    fn_free = make_vol_target_exposure(target_vol=99.0, with_regime_gate=False)
    assert abs(fn_free(prices, down, w) - 1.0) < 1e-12
    assert abs(fn_gated(prices, down, w) - 0.4) < 1e-12
    print("200dma gate composes multiplicatively with vol target: OK")


if __name__ == "__main__":
    test_identity_when_target_unreachable()
    test_exposure_falls_after_vol_shift()
    test_causality_only_trailing_window()
    test_exposure_bounds()
    test_full_covariance_not_diagonal()
    test_regime_gate_composes()
    print("\nAll vol-targeting tests passed.")
