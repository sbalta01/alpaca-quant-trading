"""Synthetic-data regression tests for the holistic weekly method (layers 2-4).

Run from the repo root:  python tests/test_weekly_holistic_synthetic.py
No network needed; requires pandas, numpy, xgboost, hmmlearn.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.weekly_momentum import run_walkforward, weekly_rebalance_dates
from src.strategies.weekly_holistic import (
    CrossSectionalRanker, RegimeGate, build_feature_panel,
    run_walkforward_holistic, signal_scores,
)


def make_prices(seed=11, n=1600, n_assets=20):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    drifts = rng.normal(0.0003, 0.0004, n_assets)
    rets = rng.normal(0, 0.015, (n, n_assets)) + drifts
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                      index=idx, name="SPY")
    return prices, bench


def test_layers_off_equals_original(prices, bench):
    """use_reversal=False, ml_weight=0, gate='simple' must reproduce
    weekly_momentum.run_walkforward exactly."""
    orig = run_walkforward(prices, bench, top_k=5, cost_bps=10.0)
    holi = run_walkforward_holistic(prices, bench, top_k=5, cost_bps=10.0,
                                    use_reversal=False, ml_weight=0.0, gate="simple")
    pd.testing.assert_series_equal(orig["daily_returns"], holi["daily_returns"])
    pd.testing.assert_series_equal(orig["turnover"], holi["turnover"])
    print("layers-off holistic == original walk-forward: OK")


def test_reversal_dampens_spikes(prices):
    """A stock that spiked in the last 5 days must score lower with the
    reversal layer than without it."""
    px = prices.copy()
    spiker = px.columns[0]
    px.loc[px.index[-5]:, spiker] *= np.linspace(1.0, 1.30, 5)  # +30% weekly spike
    base = signal_scores(px, use_reversal=False).iloc[-1]
    with_rev = signal_scores(px, use_reversal=True, rev_weight=0.25).iloc[-1]
    drop_spiker = base[spiker] - with_rev[spiker]
    others = (base.drop(spiker) - with_rev.drop(spiker)).median()
    assert drop_spiker > others, "reversal layer should penalize the spiker most"
    print(f"reversal penalizes 5d spike (score -{drop_spiker:.2f} vs median -{others:.2f}): OK")


def test_ml_ranker_no_lookahead(prices):
    """The ranker must train only on cross-sections whose targets were realized
    by the prediction date (dates <= previous rebalance)."""
    rebals = weekly_rebalance_dates(prices.index)
    panel = build_feature_panel(prices, rebals)
    ranker = CrossSectionalRanker(refit_every=1, min_train_weeks=52)
    checked = 0
    # features need ~55 weeks of warmup, so 52 training weeks exist from ~week 110
    for i in range(120, min(150, len(rebals))):
        scores = ranker.scores_for(panel, rebals[i], rebals[i - 1])
        if len(scores) > 0:
            assert ranker.last_train_date <= rebals[i - 1], (
                f"leak: trained through {ranker.last_train_date}, "
                f"predicting {rebals[i]}")
            checked += 1
    assert checked > 0, "ranker never became active"
    # the target of the last training cross-section must be realized by now:
    # its forward window is (last_train_date, next rebalance] <= prediction date
    print(f"ML ranker trained strictly on realized past cross-sections "
          f"({checked} predictions): OK")


def test_feature_panel_target_alignment(prices):
    """Target at rebalance t must equal the (t, t+1] rebalance-to-rebalance
    return rank - verified directly for one date."""
    rebals = weekly_rebalance_dates(prices.index)
    panel = build_feature_panel(prices, rebals)
    t, t1 = rebals[100], rebals[101]
    fwd = prices.loc[t1] / prices.loc[t] - 1.0
    expected = fwd.rank(pct=True)
    got = panel.xs(t, level=0)["target"]
    common = got.index.intersection(expected.index)
    assert np.allclose(got[common], expected[common]), "target misaligned"
    print("feature-panel target = next-rebalance return rank: OK")


def test_regime_gate_values(prices, bench):
    """Gate output must always be in {0.3, 0.65, 1.0}; a strong uptrend with
    calm markets must give 1.0."""
    gate = RegimeGate()
    rebals = [d for d in weekly_rebalance_dates(prices.index) if d >= prices.index[600]]
    vals = set()
    for d in rebals[:40]:
        vals.add(gate.exposure(bench.loc[:d], prices.loc[:d], d))
    assert vals <= {0.3, 0.65, 1.0}, f"unexpected exposure values: {vals}"

    # With history shorter than hmm_min_history and fewer than turb_min_obs
    # observations, only the 200dma check is active -> deterministic outcome.
    short_idx = bench.index[:400]
    up = pd.Series(100 * np.exp(np.linspace(0, 0.5, 400)), index=short_idx)
    down = pd.Series(100 * np.exp(np.linspace(0.5, 0, 400)), index=short_idx)
    d = short_idx[-1]
    assert RegimeGate().exposure(up, prices.loc[:d], d) == 1.0, "uptrend must be 1.0"
    assert RegimeGate().exposure(down, prices.loc[:d], d) == 0.65, "downtrend must be 0.65"
    print(f"regime gate values {sorted(vals)} within contract; "
          f"MA check drives 1.0 vs 0.65: OK")


def test_full_holistic_runs(prices, bench):
    """Full pipeline (L2+L3+L4) runs end-to-end with sane outputs."""
    res = run_walkforward_holistic(prices, bench, top_k=5, cost_bps=10.0,
                                   use_reversal=True, ml_weight=0.5,
                                   ml_min_train_weeks=52, gate="hmm")
    assert res["ml_active_from"] is not None, "ML layer never activated"
    for w in res["weights"].values():
        assert w.sum() <= 1.0 + 1e-9, "weights exceed full exposure"
        assert (w >= -1e-12).all(), "negative weight"
    assert res["daily_returns"].notna().all()
    print(f"full holistic pipeline runs (ML active from "
          f"{res['ml_active_from'].date()}): OK")


if __name__ == "__main__":
    prices, bench = make_prices()
    test_layers_off_equals_original(prices, bench)
    test_reversal_dampens_spikes(prices)
    test_ml_ranker_no_lookahead(prices)
    test_feature_panel_target_alignment(prices)
    test_regime_gate_values(prices, bench)
    test_full_holistic_runs(prices, bench)
    print("\nAll holistic-method tests passed.")
