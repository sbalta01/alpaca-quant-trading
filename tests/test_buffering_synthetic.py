"""Synthetic-data tests for BufferedSelector (turnover reduction).

Run from the repo root:  python tests/test_buffering_synthetic.py
No network needed; requires pandas, numpy.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.weekly_momentum import (
    BufferedSelector, apply_no_trade_band, default_selector, run_walkforward,
)


def make_prices(seed=31, n=1500, n_assets=30):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    drifts = rng.normal(0.0003, 0.0005, n_assets)
    rets = rng.normal(0, 0.015, (n, n_assets)) + drifts
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                      index=idx, name="SPY")
    return prices, bench


def test_identity_at_buffer_1(prices, bench):
    """buffer_mult=1.0 must reproduce plain nlargest EXACTLY - the feature is
    provably a no-op at its identity setting."""
    scores = pd.Series({f"S{i:02d}": 30 - i for i in range(30)}, dtype=float)
    sel = BufferedSelector(buffer_mult=1.0)
    for _ in range(5):
        assert list(sel(scores, 10)) == list(default_selector(scores, 10))

    a = run_walkforward(prices, bench, top_k=10, cost_bps=10.0)
    b = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                        selector_factory=lambda: BufferedSelector(1.0))
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])
    pd.testing.assert_series_equal(a["turnover"], b["turnover"])
    print("buffer_mult=1.0 == default_selector, bit-for-bit: OK")


def test_buffering_reduces_turnover(prices, bench):
    base = run_walkforward(prices, bench, top_k=10, cost_bps=10.0)
    buf = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                          selector_factory=lambda: BufferedSelector(1.5))
    to_base = base["turnover"].mean() * 52
    to_buf = buf["turnover"].mean() * 52
    assert to_buf < to_base, f"buffering must reduce turnover: {to_buf} vs {to_base}"
    print(f"buffering cuts turnover {to_base:.1f}x -> {to_buf:.1f}x/yr: OK")


def test_monotonic_in_buffer_mult(prices, bench):
    tos = []
    for m in (1.0, 1.5, 2.0, 3.0):
        res = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                              selector_factory=lambda m=m: BufferedSelector(m))
        tos.append(res["turnover"].mean() * 52)
    assert tos == sorted(tos, reverse=True), f"turnover should fall with buffer: {tos}"
    print(f"turnover monotonically falls with buffer_mult: "
          f"{[round(t, 1) for t in tos]}: OK")


def test_holds_through_buffer_zone_then_drops():
    """A name walked from rank 3 -> 12 -> 20 must be HELD at 12 (inside the
    1.5*10=15 buffer) and DROPPED at 20 (outside it)."""
    names = [f"S{i:02d}" for i in range(30)]
    sel = BufferedSelector(buffer_mult=1.5)   # keep_depth = 15

    def scores_with(target, rank):
        others = [n for n in names if n != target]
        order = others[:rank] + [target] + others[rank:]
        return pd.Series({n: float(len(order) - i) for i, n in enumerate(order)})

    held = sel(scores_with("S05", 2), 10)
    assert "S05" in held, "should be selected at rank 3"
    held = sel(scores_with("S05", 11), 10)
    assert "S05" in held, "should be HELD at rank 12 (inside buffer)"
    held = sel(scores_with("S05", 19), 10)
    assert "S05" not in held, "should be DROPPED at rank 20 (outside buffer)"
    print("name held at rank 12, dropped at rank 20: OK")


def test_missing_incumbent_does_not_raise():
    """A delisted holding must be dropped silently, not KeyError."""
    sel = BufferedSelector(buffer_mult=1.5)
    full = pd.Series({f"S{i:02d}": float(30 - i) for i in range(30)})
    held = sel(full, 10)
    assert len(held) == 10

    shrunk = full.drop(list(held)[:3])       # three incumbents vanish
    out = sel(shrunk, 10)
    assert len(out) == 10
    assert all(t in shrunk.index for t in out)
    print("vanished incumbents are dropped without raising: OK")


def test_no_trade_band(prices, bench):
    """min_trade_fraction=0 must be a bit-for-bit no-op; 0.005 (the live
    executor's setting) must reduce turnover without changing holdings much."""
    a = run_walkforward(prices, bench, top_k=10, cost_bps=10.0)
    b = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                        min_trade_fraction=0.0)
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])

    band = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                           min_trade_fraction=0.005)
    to_a = a["turnover"].mean() * 52
    to_band = band["turnover"].mean() * 52
    assert to_band < to_a, f"band must cut turnover: {to_band:.1f} vs {to_a:.1f}"
    print(f"no-trade band: 0.0 is a no-op; 0.005 cuts turnover "
          f"{to_a:.1f}x -> {to_band:.1f}x/yr: OK")


def test_band_extraction_matches_inline(prices, bench):
    """`apply_no_trade_band` was extracted out of run_walkforward so the live
    deployer and the backtest share one implementation. It must reproduce the
    inline logic exactly - including that it is a no-op with no current book."""
    target = pd.Series({"A": 0.30, "B": 0.20, "C": 0.10})
    current = pd.Series({"A": 0.299, "B": 0.15, "D": 0.05})

    # reference: the pre-extraction inline block, verbatim
    names = target.index.union(current.index)
    tgt = target.reindex(names, fill_value=0.0)
    cur = current.reindex(names, fill_value=0.0)
    small = (tgt - cur).abs() < 0.005
    tgt[small] = cur[small]
    expected = tgt[tgt > 0].sort_values(ascending=False)

    pd.testing.assert_series_equal(
        apply_no_trade_band(target, current, 0.005), expected)
    # identity paths
    pd.testing.assert_series_equal(
        apply_no_trade_band(target, current, 0.0), target)
    pd.testing.assert_series_equal(
        apply_no_trade_band(target, pd.Series(dtype=float), 0.005), target)
    print("apply_no_trade_band reproduces the inline block bit-for-bit: OK")


def test_always_returns_k(prices, bench):
    """Selector must always fill exactly k slots (or all names if fewer)."""
    sel = BufferedSelector(buffer_mult=2.0)
    scores = pd.Series({f"S{i:02d}": float(30 - i) for i in range(30)})
    for _ in range(10):
        assert len(sel(scores, 10)) == 10
    small = scores.iloc[:4]
    assert len(BufferedSelector(1.5)(small, 10)) == 4
    print("selector fills exactly k slots (and degrades gracefully): OK")


if __name__ == "__main__":
    prices, bench = make_prices()
    test_identity_at_buffer_1(prices, bench)
    test_buffering_reduces_turnover(prices, bench)
    test_monotonic_in_buffer_mult(prices, bench)
    test_holds_through_buffer_zone_then_drops()
    test_missing_incumbent_does_not_raise()
    test_no_trade_band(prices, bench)
    test_band_extraction_matches_inline(prices, bench)
    test_always_returns_k(prices, bench)
    print("\nAll buffering tests passed.")
