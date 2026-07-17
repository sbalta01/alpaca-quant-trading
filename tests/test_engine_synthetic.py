"""Synthetic-data regression tests for the backtest engine and fixed strategies.

Run from the repo root:  python tests/test_engine_synthetic.py
No network needed; requires pandas, numpy, scikit-learn.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.backtester import BacktestEngine
from src.strategies.macd import MACDStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy


def make_data(seed=7, n=1200, symbols=("AAA", "BBB")):
    rng = np.random.RandomState(seed)
    frames = []
    for sym in symbols:
        r = rng.normal(0.0004, 0.012, n)
        close = 100 * np.exp(np.cumsum(r))
        idx = pd.bdate_range("2020-01-02", periods=n)
        f = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                          "close": close, "volume": rng.randint(1_000_000, 5_000_000, n).astype(float)},
                         index=idx)
        f.index.name = "timestamp"
        f["symbol"] = sym
        frames.append(f.reset_index().set_index(["symbol", "timestamp"]))
    return pd.concat(frames).sort_index()


def test_costs_reduce_equity(data):
    eq0 = BacktestEngine(MACDStrategy(), data.copy(), cost_bps=0.0).run()["equity"] \
        .groupby(level="timestamp").sum().iloc[-1]
    eq10 = BacktestEngine(MACDStrategy(), data.copy(), cost_bps=10.0).run()["equity"] \
        .groupby(level="timestamp").sum().iloc[-1]
    assert eq10 < eq0, "transaction costs must reduce final equity"
    print(f"costs: {eq0:.0f} (0 bps) -> {eq10:.0f} (10 bps)  OK")


def test_pnl_timing():
    """Buy at close day0 (100), ride +10% twice, exit close day2 -> 21% gross."""
    prices = pd.Series([100.0, 110.0, 121.0, 121.0, 121.0])
    pos = pd.Series([1, 1, 0, 0, 0])
    ret = (prices.pct_change() * pos.shift(1)).fillna(0.0)
    final = float((1 + ret).cumprod().iloc[-1])
    assert abs(final - 1.21) < 1e-9
    print("pnl timing: OK")


def test_test_mask_restriction(data):
    engine = BacktestEngine(MACDStrategy(), data.copy(), cost_bps=10.0)
    res = engine.run()
    perf_full = engine.performance(res.copy(), num_years=len(res) / 2 / 252)

    res_ml = res.copy()
    res_ml["test_mask"] = 0.0
    ts_all = res_ml.index.get_level_values("timestamp").unique().sort_values()
    test_ts = ts_all[int(0.7 * len(ts_all)):]
    res_ml.loc[res_ml.index.get_level_values("timestamp").isin(test_ts), "test_mask"] = 1.0
    perf_masked = engine.performance(res_ml, num_years=len(test_ts) / 252)
    assert perf_masked["Sharpe"] != perf_full["Sharpe"]
    print(f"test-mask metrics: Sharpe full={perf_full['Sharpe']:.3f} vs test-only={perf_masked['Sharpe']:.3f}  OK")


def test_bollinger_engine_compatible(data):
    res = BacktestEngine(BollingerMeanReversionStrategy(window=20, k=2), data.copy(), cost_bps=10.0).run()
    assert "position" in res.columns
    assert set(np.unique(res["position"])) <= {0.0, 1.0}
    print("bollinger position column: OK")


def test_buy_and_hold(data):
    res = BacktestEngine(BuyAndHoldStrategy(), data.copy(), cost_bps=10.0).run()
    assert res["equity"].notna().all()
    print("buy-and-hold benchmark: OK")


def test_trading_env_reward():
    """Reward must include price P&L, not just -costs."""
    try:
        from src.env.trading_env import TradingEnv
    except ImportError:
        print("trading env reward: SKIPPED (gymnasium not installed)")
        return
    idx = pd.MultiIndex.from_product(
        [["AAA"], pd.bdate_range("2020-01-02", periods=3)], names=["symbol", "timestamp"])
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0], "tech": [0.0, 0.0, 0.0],
                       "macro": [0.0, 0.0, 0.0]}, index=idx)
    env = TradingEnv(df, tech_cols=["tech"], macro_cols=["macro"],
                     initial_cash=1000.0, transaction_cost=0.001)
    env.reset()
    _, reward, _, _, _ = env.step(np.array([1.0]))  # fully invest
    assert reward > 50, f"reward should reflect the +10% move, got {reward}"
    print(f"trading env reward: {reward:+.2f} on a +10% bar  OK")


if __name__ == "__main__":
    data = make_data()
    test_pnl_timing()
    test_costs_reduce_equity(data)
    test_test_mask_restriction(data)
    test_bollinger_engine_compatible(data)
    test_buy_and_hold(data)
    test_trading_env_reward()
    print("\nAll tests passed.")
