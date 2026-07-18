"""Synthetic-data regression tests for the selector seam and sensitivity harness.

Run from the repo root:  python tests/test_sensitivity_synthetic.py
No network needed; requires pandas, numpy.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.weekly_momentum import (
    compute_target_weights, default_selector, momentum_scores, run_walkforward,
    weekly_rebalance_dates, weekly_rebalance_dates_on,
)


def make_prices(seed=17, n=1500, n_assets=25):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    drifts = rng.normal(0.0003, 0.0004, n_assets)
    rets = rng.normal(0, 0.015, (n, n_assets)) + drifts
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                      index=idx, name="SPY")
    return prices, bench


def test_default_selector_is_nlargest(prices):
    """The seam's default must be exactly the old nlargest behavior."""
    scores = momentum_scores(prices).iloc[-1].dropna()
    assert list(default_selector(scores, 10)) == list(scores.nlargest(10).index)
    print("default_selector == nlargest: OK")


def test_selector_default_preserves_weights(prices, bench):
    """compute_target_weights with selector=None must equal passing the default."""
    a = compute_target_weights(prices, bench, top_k=8)
    b = compute_target_weights(prices, bench, top_k=8, selector=default_selector)
    pd.testing.assert_series_equal(a, b)
    print("compute_target_weights: selector=None == default_selector: OK")


def test_run_walkforward_seam_is_noop(prices, bench):
    """selector_factory=None must reproduce the pre-seam run bit-for-bit."""
    a = run_walkforward(prices, bench, top_k=8, cost_bps=10.0)
    b = run_walkforward(prices, bench, top_k=8, cost_bps=10.0,
                        selector_factory=lambda: default_selector)
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])
    pd.testing.assert_series_equal(a["turnover"], b["turnover"])
    print("run_walkforward: selector_factory=None == explicit default: OK")


def test_selector_factory_isolates_state(prices, bench):
    """A stateful selector must not leak state between runs - the factory is
    called once per run, so two runs must produce identical results."""
    class Counting:
        def __init__(self):
            self.calls = 0

        def __call__(self, scores, top_k):
            self.calls += 1
            # deliberately state-dependent: alternates the tie-break direction
            k = min(top_k, len(scores))
            order = scores.nlargest(k) if self.calls % 2 else scores.nsmallest(k)
            return order.index

    a = run_walkforward(prices, bench, top_k=8, cost_bps=10.0,
                        selector_factory=Counting)
    b = run_walkforward(prices, bench, top_k=8, cost_bps=10.0,
                        selector_factory=Counting)
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])
    print("selector_factory gives each run fresh state: OK")


def test_random_selector_changes_results(prices, bench):
    """Sanity: a random selector must actually differ from the real signal,
    otherwise the null test would be measuring nothing."""
    def factory():
        rng = np.random.RandomState(0)
        return lambda s, k: pd.Index(rng.choice(s.index, size=min(k, len(s)),
                                                replace=False))
    real = run_walkforward(prices, bench, top_k=8, cost_bps=10.0)
    rand = run_walkforward(prices, bench, top_k=8, cost_bps=10.0,
                           selector_factory=factory)
    assert not np.allclose(real["daily_returns"].values, rand["daily_returns"].values)
    print("random-k selector produces a genuinely different track record: OK")


def test_sticky_null_is_turnover_matched(prices, bench):
    """The sticky random null must churn far less than the i.i.d. one - that is
    the whole point of it, since an unmatched null is confounded by costs."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main.sensitivity_weekly_momentum import (
        _iid_random_factory, _sticky_random_factory,
    )
    iid = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                          selector_factory=_iid_random_factory(0))
    sticky = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                             selector_factory=_sticky_random_factory(0, replace_n=2))
    iid_to = iid["turnover"].mean() * 52
    sticky_to = sticky["turnover"].mean() * 52
    assert sticky_to < iid_to / 2, (
        f"sticky null must churn much less: {sticky_to:.1f}x vs iid {iid_to:.1f}x")
    print(f"sticky null turnover {sticky_to:.1f}x/yr vs i.i.d. {iid_to:.1f}x/yr: OK")


def test_sticky_null_holds_names(prices, bench):
    """replace_n=0 must hold its initial picks forever (turnover from weight
    drift only); larger replace_n must churn more."""
    from main.sensitivity_weekly_momentum import _sticky_random_factory
    tos = []
    for n in (0, 2, 5):
        res = run_walkforward(prices, bench, top_k=10, cost_bps=10.0,
                              selector_factory=_sticky_random_factory(1, replace_n=n))
        tos.append(res["turnover"].mean() * 52)
    assert tos[0] < tos[1] < tos[2], f"turnover must rise with replace_n, got {tos}"
    print(f"sticky turnover rises with replace_n: "
          f"{tos[0]:.1f} < {tos[1]:.1f} < {tos[2]:.1f}: OK")


def test_weekday_schedule(prices, bench):
    """Friday schedule must match the original; earlier weekdays must differ
    and still produce one rebalance per ISO week."""
    idx = prices.index
    fri = weekly_rebalance_dates_on(idx, 4)
    orig = weekly_rebalance_dates(idx)
    assert list(fri) == list(orig), "weekday=4 must reproduce weekly_rebalance_dates"

    tue = weekly_rebalance_dates_on(idx, 1)
    assert (tue.dayofweek <= 1).all(), "weekday=1 schedule must be Mon/Tue only"
    assert abs(len(tue) - len(fri)) <= 2, "should be ~one rebalance per week"
    print(f"weekly_rebalance_dates_on: Fri=={len(fri)} matches original, "
          f"Tue={len(tue)} dates: OK")


def test_rebal_dates_passthrough(prices, bench):
    """Passing the default schedule explicitly must be a no-op."""
    a = run_walkforward(prices, bench, top_k=8, cost_bps=10.0)
    b = run_walkforward(prices, bench, top_k=8, cost_bps=10.0,
                        rebal_dates=weekly_rebalance_dates(prices.index))
    pd.testing.assert_series_equal(a["daily_returns"], b["daily_returns"])
    print("rebal_dates passthrough of the default schedule is a no-op: OK")


def test_shared_warmup_gives_identical_windows(prices, bench):
    """Configs with different lookbacks must be scored over an identical window
    when warmup is pinned - otherwise the sweep silently favors short lookbacks."""
    warmup = 300
    a = run_walkforward(prices, bench, top_k=8, lookbacks=(252, 126),
                        cost_bps=10.0, warmup=warmup)
    b = run_walkforward(prices, bench, top_k=8, lookbacks=(126,),
                        cost_bps=10.0, warmup=warmup)
    first_a = a["daily_returns"].ne(0).idxmax()
    first_b = b["daily_returns"].ne(0).idxmax()
    assert first_a == first_b, (
        f"pinned warmup must align start dates, got {first_a} vs {first_b}")
    assert min(a["weights"]) == min(b["weights"]), "first rebalance must align"
    print(f"pinned warmup aligns evaluation windows across lookbacks "
          f"(both start {first_a.date()}): OK")


if __name__ == "__main__":
    prices, bench = make_prices()
    test_default_selector_is_nlargest(prices)
    test_selector_default_preserves_weights(prices, bench)
    test_run_walkforward_seam_is_noop(prices, bench)
    test_selector_factory_isolates_state(prices, bench)
    test_random_selector_changes_results(prices, bench)
    test_sticky_null_is_turnover_matched(prices, bench)
    test_sticky_null_holds_names(prices, bench)
    test_weekday_schedule(prices, bench)
    test_rebal_dates_passthrough(prices, bench)
    test_shared_warmup_gives_identical_windows(prices, bench)
    print("\nAll sensitivity/selector tests passed.")
