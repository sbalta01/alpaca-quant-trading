"""
Synthetic tests for the two-sleeve architecture.

Run from the repo root:  python tests/test_sleeves_synthetic.py

The load-bearing property here is SHARED-ACCOUNT SAFETY. Both sleeves trade one
Alpaca account, and build_orders liquidates any position it is shown that is not
in its target list. So the tests that actually protect money are:

  * a sleeve never emits an order outside its own managed set, and
  * a sleeve that FAILED to compute leaves its holdings completely alone
    (rather than having them read as "target 0" and sold).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.orders import build_orders
from src.strategies.sleeves import (SleeveConfig, managed_symbols, net_targets,
                                    validate_sleeves)
from src.strategies.weekly_momentum import apply_no_trade_band

STOCKS = ("AAPL", "MU", "NVDA")
ETFS = ("TLT", "IEF", "GLD", "DBC", "VNQ")

MOM = SleeveConfig("momentum", 0.70, universe=None, residual=True)
DIV = SleeveConfig("diversifier", 0.30, universe=ETFS, top_k=2, weight_cap=0.60,
                   target_vol=0.10, use_dma_gate=False, low_exposure=1.0)
BOTH = (MOM, DIV)


def test_build_orders_identity_without_managed():
    """managed=None must reproduce the pre-refactor behavior exactly."""
    targets = {"AAPL": 0.3, "MU": 0.2}
    current = {"AAPL": 10_000.0, "NVDA": 5_000.0}
    equity = 100_000.0
    a = build_orders(targets, current, equity)
    b = build_orders(targets, current, equity,
                     managed=set(targets) | set(current))
    assert a == b, f"managed-set identity broken:\n{a}\n{b}"
    # and the un-managed call still liquidates the untargeted holding
    assert any(o[0] == "NVDA" and o[1] == "sell" for o in a), a
    print("build_orders identity with/without managed: OK")


def test_shared_account_liquidation_safety():
    """
    THE headline property: an order for symbol X may only be emitted by the
    sleeve whose managed set contains X, and the managed sets are disjoint.
    """
    positions = {"AAPL": 5_000.0, "MU": 3_000.0, "TLT": 2_000.0, "GLD": 1_000.0}
    equity = 100_000.0
    m_managed = managed_symbols(MOM, BOTH, STOCKS, positions)
    d_managed = managed_symbols(DIV, BOTH, ETFS, positions)

    assert not (m_managed & d_managed), \
        f"managed sets overlap: {sorted(m_managed & d_managed)}"

    m_orders = build_orders({"AAPL": 0.5}, positions, equity, managed=m_managed)
    d_orders = build_orders({"TLT": 0.5}, positions, equity, managed=d_managed)

    for sym, *_ in m_orders:
        assert sym in m_managed, f"momentum touched unmanaged {sym}"
        assert sym not in ETFS, f"momentum sleeve emitted an ETF order: {sym}"
    for sym, *_ in d_orders:
        assert sym in d_managed, f"diversifier touched unmanaged {sym}"
        assert sym not in STOCKS, f"diversifier sleeve emitted a stock order: {sym}"
    print("shared-account liquidation safety (neither sleeve touches the other): OK")


def test_skipped_sleeve_leaves_its_book_alone():
    """
    A sleeve that fails must produce ZERO orders for its symbols - not
    'sell all'. This is the failure mode that would cost real money: one
    yfinance hiccup on the ETF panel must not flatten the ETF book.
    """
    positions = {"AAPL": 5_000.0, "TLT": 20_000.0, "GLD": 10_000.0}
    equity = 100_000.0
    # diversifier skipped => excluded from the netted book AND the managed set
    combined = net_targets({"momentum": pd.Series({"AAPL": 0.5})}, [MOM])
    managed_all = managed_symbols(MOM, BOTH, STOCKS, positions)
    orders = build_orders(combined.to_dict(), positions, equity, managed=managed_all)

    touched = {o[0] for o in orders}
    assert not (touched & set(ETFS)), \
        f"skipped sleeve's book was traded: {sorted(touched & set(ETFS))}"

    # contrast: the WRONG implementation (target 0, still managed) sells them
    wrong = build_orders(combined.to_dict(), positions, equity,
                         managed=managed_all | set(ETFS))
    assert {o[0] for o in wrong} & set(ETFS), \
        "control case failed - the test would not detect the bug"
    print("skipped sleeve leaves its holdings untouched: OK")


def test_sizing_against_sleeve_equity():
    """A 0.5 weight in a 0.30 sleeve on $100k is $15k, not $50k."""
    combined = net_targets({"diversifier": pd.Series({"TLT": 0.5, "GLD": 0.5})}, [DIV])
    assert abs(combined["TLT"] - 0.15) < 1e-12, combined["TLT"]
    orders = build_orders(combined.to_dict(), {}, 100_000.0,
                          managed=set(ETFS))
    notional = {o[0]: o[2] for o in orders}
    assert abs(notional["TLT"] - 15_000.0) < 1e-6, notional
    print("sleeve weights size against allocated capital, not total equity: OK")


def test_netting_equals_per_sleeve():
    """On disjoint universes, one netted pass equals two per-sleeve passes."""
    positions = {"AAPL": 5_000.0, "TLT": 2_000.0}
    equity = 100_000.0
    wm, wd = pd.Series({"AAPL": 0.4, "MU": 0.3}), pd.Series({"TLT": 0.6})
    m_managed = managed_symbols(MOM, BOTH, STOCKS, positions)
    d_managed = managed_symbols(DIV, BOTH, ETFS, positions)

    combined = net_targets({"momentum": wm, "diversifier": wd}, BOTH)
    netted = build_orders(combined.to_dict(), positions, equity,
                          managed=m_managed | d_managed)
    separate = (build_orders((wm * MOM.allocation).to_dict(), positions, equity,
                             managed=m_managed)
                + build_orders((wd * DIV.allocation).to_dict(), positions, equity,
                               managed=d_managed))
    assert {o[0]: (o[1], o[2]) for o in netted} == \
           {o[0]: (o[1], o[2]) for o in separate}, f"{netted}\n{separate}"
    print("netted single pass == two per-sleeve passes (disjoint universes): OK")


def test_orphan_is_liquidated_by_residual():
    """
    A position no sleeve's universe claims must still be sellable, or the
    managed gate would strand delisted names in the account forever.
    """
    positions = {"AXON": 4_000.0, "TLT": 1_000.0}
    m_managed = managed_symbols(MOM, BOTH, STOCKS, positions)
    assert "AXON" in m_managed, "residual sleeve did not absorb the orphan"
    assert "TLT" not in m_managed, "residual sleeve absorbed another sleeve's symbol"
    orders = build_orders({"AAPL": 0.5}, positions, 100_000.0, managed=m_managed)
    assert any(o[0] == "AXON" and o[1] == "sell" for o in orders), orders
    print("orphan positions are absorbed and liquidated by the residual sleeve: OK")


def test_targets_outside_managed_are_rejected():
    """A sleeve targeting outside its universe is a config error - fail loud."""
    try:
        build_orders({"TLT": 0.5}, {}, 100_000.0, managed={"AAPL", "MU"})
    except ValueError as e:
        assert "TLT" in str(e), e
        print("targets outside the managed set raise: OK")
        return
    raise AssertionError("expected ValueError for out-of-universe target")


def test_only_flag_does_not_widen_residual_claim():
    """
    Regression: ownership must derive from the DECLARED registry, not from the
    sleeves being run. If `--only momentum` also narrowed the ownership map, the
    residual momentum sleeve would stop seeing the ETFs as claimed, absorb them
    as orphans, and liquidate the whole ETF book.
    """
    positions = {"AAPL": 5_000.0, "TLT": 20_000.0, "GLD": 10_000.0}
    correct = managed_symbols(MOM, BOTH, STOCKS, positions)      # declared registry
    assert not (correct & set(ETFS)), sorted(correct & set(ETFS))

    buggy = managed_symbols(MOM, (MOM,), STOCKS, positions)      # filtered registry
    assert buggy & set(ETFS), "control case failed - bug would go undetected"

    orders = build_orders({"AAPL": 0.5}, positions, 100_000.0, managed=correct)
    assert not ({o[0] for o in orders} & set(ETFS)), orders
    print("--only does not widen the residual sleeve's claim: OK")


def test_config_validation():
    """validate_sleeves must catch every misconfiguration that misprices orders."""
    validate_sleeves(BOTH)                       # the real config passes

    def expect(bad, needle):
        try:
            validate_sleeves(bad)
        except ValueError as e:
            assert needle in str(e), f"wrong error for {needle}: {e}"
            return
        raise AssertionError(f"expected ValueError containing {needle!r}")

    expect((SleeveConfig("a", 0.5), SleeveConfig("b", 0.3, universe=ETFS)),
           "sum to 1.0")
    expect((SleeveConfig("a", 0.7, residual=True),
            SleeveConfig("b", 0.3, universe=ETFS, residual=True)), "residual")
    expect((SleeveConfig("a", 0.7, universe=("GLD", "AAPL")),
            SleeveConfig("b", 0.3, universe=ETFS)), "share symbols")
    # the latent inverse_vol_weights bug: top_k * weight_cap < 1 under-invests
    expect((SleeveConfig("a", 0.7, residual=True),
            SleeveConfig("b", 0.3, universe=ETFS, top_k=2, weight_cap=0.35)),
           "top_k * weight_cap")
    print("sleeve config validation (sums, residual, overlap, top_k*cap): OK")


def test_net_targets_skips_absent_sleeves():
    """An absent sleeve contributes nothing; it is never zero-filled."""
    combined = net_targets({"momentum": pd.Series({"AAPL": 0.5})}, BOTH)
    assert list(combined.index) == ["AAPL"], combined
    assert abs(combined["AAPL"] - 0.35) < 1e-12, combined
    assert len(net_targets({}, BOTH)) == 0
    print("net_targets omits absent sleeves rather than zero-filling: OK")


def test_no_trade_band_in_sleeve_space():
    """
    The band is a fraction of SLEEVE capital. A $600 drift on a $30k sleeve is
    2% and trades; the same $600 on $100k of equity is 0.6% and would not - so
    applying the band in the wrong space changes what executes.
    """
    target = pd.Series({"TLT": 0.50, "GLD": 0.50})
    current = pd.Series({"TLT": 0.52, "GLD": 0.48})       # 2% drift, sleeve space
    kept = apply_no_trade_band(target, current, 0.005)
    assert abs(kept["TLT"] - 0.50) < 1e-12, "2% drift should trade"
    banded = apply_no_trade_band(target, current, 0.05)   # wider band
    assert abs(banded["TLT"] - 0.52) < 1e-12, "sub-band drift should be held"
    print("no-trade band operates in sleeve weight space: OK")


if __name__ == "__main__":
    test_build_orders_identity_without_managed()
    test_shared_account_liquidation_safety()
    test_skipped_sleeve_leaves_its_book_alone()
    test_sizing_against_sleeve_equity()
    test_netting_equals_per_sleeve()
    test_orphan_is_liquidated_by_residual()
    test_targets_outside_managed_are_rejected()
    test_only_flag_does_not_widen_residual_claim()
    test_config_validation()
    test_net_targets_skips_absent_sleeves()
    test_no_trade_band_in_sleeve_space()
    print("\nAll two-sleeve tests passed.")
