"""Synthetic-data regression tests for the walk-forward harness.

Run from the repo root:  python tests/test_walkforward_synthetic.py
No network needed; requires pandas, numpy.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.walkforward import WalkForwardHarness
from src.strategies.base_strategy import Strategy
from src.strategies.macd import MACDStrategy

WARMUP = 30      # rows the mock drops before splitting, like real feature warmup
HORIZON = 10


class RecordingMLStrategy(Strategy):
    """Mirrors the real ML strategies' structure: drop warmup rows, split
    chronologically at train_frac, 'train' on the first part, emit positions.
    Records the training span so the purge gap can be checked, and uses a
    deterministic causal rule (close > SMA20) so stitching is verifiable."""
    name = "RecordingML"
    multi_symbol = False
    train_spans = []   # class-level: (last_train_timestamp) per generate_signals call

    def __init__(self):
        self.train_frac = 0.7
        self.horizon = HORIZON

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["sma20"] = df["close"].rolling(20).mean()
        feat = df.iloc[WARMUP:]                       # warmup dropna stand-in
        split = int(len(feat) * self.train_frac)      # chronological split
        RecordingMLStrategy.train_spans.append(feat.index[split - 1])
        out = df.copy()
        out["position"] = (df["close"] > df["sma20"]).astype(float)
        out["signal"] = out["position"].diff().fillna(0.0)
        return out


def make_data(seed=3, n=1500, symbols=("AAA", "BBB")):
    rng = np.random.RandomState(seed)
    frames = []
    for sym in symbols:
        r = rng.normal(0.0004, 0.012, n)
        close = 100 * np.exp(np.cumsum(r))
        idx = pd.bdate_range("2019-01-02", periods=n)
        f = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                          "close": close, "volume": np.full(n, 1e6)}, index=idx)
        f.index.name = "timestamp"
        f["symbol"] = sym
        frames.append(f.reset_index().set_index(["symbol", "timestamp"]))
    return pd.concat(frames).sort_index()


def test_purge_gap_respected(data):
    """The realized last training row must never enter the purge gap."""
    RecordingMLStrategy.train_spans = []
    h = WalkForwardHarness(RecordingMLStrategy, data, train_days=500, test_days=63,
                           purge_days=HORIZON, warmup_buffer=WARMUP + 20,
                           cost_bps=10.0, verbose=False)
    h.run()
    i = 0
    for rec in h.folds_:
        sub = data.xs(rec["symbol"], level="symbol")
        for f in rec["folds"]:
            last_allowed = sub.index[f["test_start"] - h.purge_days - 1]
            realized = RecordingMLStrategy.train_spans[i]
            assert realized <= last_allowed, (
                f"leak: fold trained through {realized}, allowed {last_allowed}")
            i += 1
    assert i == len(RecordingMLStrategy.train_spans)
    print(f"purge gap respected across {i} folds: OK")


def test_stitching_matches_causal_rule(data):
    """Stitched OOS positions must equal the causal rule computed on the full
    history (folds only re-slice the data; the rule itself is deterministic)."""
    h = WalkForwardHarness(RecordingMLStrategy, data, train_days=500, test_days=63,
                           purge_days=HORIZON, warmup_buffer=WARMUP + 20,
                           cost_bps=0.0, verbose=False)
    res = h.run()
    for sym in ("AAA", "BBB"):
        sub = data.xs(sym, level="symbol")
        expected = (sub["close"] > sub["close"].rolling(20).mean()).astype(float)
        got = res.xs(sym, level="symbol")
        oos = got["test_mask"] == 1.0
        assert (got.loc[oos, "position"] == expected[oos.values]).all(), \
            f"stitched positions diverge from causal rule for {sym}"
        assert (got.loc[~oos, "position"] == 0.0).all(), \
            f"non-zero position before first OOS date for {sym}"
    print("stitched OOS positions match causal single-pass rule: OK")


def test_costs_and_mask(data):
    def total_ret(cost):
        h = WalkForwardHarness(RecordingMLStrategy, data, train_days=500,
                               test_days=63, purge_days=HORIZON,
                               warmup_buffer=WARMUP + 20, cost_bps=cost, verbose=False)
        res = h.run()
        oos = res[res["test_mask"] == 1.0]
        assert (res.groupby(level="symbol")["test_mask"].sum() > 0).all()
        return (1 + oos.groupby(level="symbol")["returns"].apply(
            lambda r: (1 + r).prod() - 1)).prod()

    r0, r10 = total_ret(0.0), total_ret(10.0)
    assert r10 < r0, "costs must reduce stitched OOS returns"
    print(f"costs reduce OOS returns: {r0:.4f} (0 bps) -> {r10:.4f} (10 bps)  OK")


def test_rule_based_path(data):
    """MACD has no train_frac: single causal pass, OOS window still enforced."""
    h = WalkForwardHarness(MACDStrategy, data, train_days=500, test_days=63,
                           purge_days=5, cost_bps=10.0, verbose=False)
    res = h.run()
    perf = h.performance(res)
    assert ("AAA", "Strategy") in perf.index and ("AAA", "Buy&Hold") in perf.index
    assert ("ALL (equal-weight)", "Strategy") in perf.index
    sub = res.xs("AAA", level="symbol")
    assert (sub.loc[sub["test_mask"] == 0.0, "position"] == 0.0).all()
    print("rule-based path + performance table: OK")


if __name__ == "__main__":
    data = make_data()
    test_purge_gap_respected(data)
    test_stitching_matches_causal_rule(data)
    test_costs_and_mask(data)
    test_rule_based_path(data)
    print("\nAll walk-forward tests passed.")
