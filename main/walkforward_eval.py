# main/walkforward_eval.py
"""
Walk-forward re-evaluation of the single-name strategy stack (fix-order item 4).

Every result is out-of-sample: models are refit every --test-days with a
--purge gap, and metrics are computed only on the stitched OOS segments, net
of --cost-bps per unit turnover. Compare each strategy against Buy&Hold of the
same symbol over the identical OOS window.

Usage (from the repo root, with .venv active):

    python main/walkforward_eval.py --strategy macd --symbols AAPL MSFT SPY
    python main/walkforward_eval.py --strategy xgboost --symbols AAPL --fast
    python main/walkforward_eval.py --strategy xgboost --symbols AAPL MSFT NVDA \
        --years 8 --train-days 756 --test-days 63
    python main/walkforward_eval.py --strategy bollinger --symbols SPY QQQ

--fast disables Optuna hyperparameter search and RFECV feature selection so a
multi-fold run finishes in minutes; drop it for the full research configuration.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, ".")

from src.backtesting.walkforward import WalkForwardHarness


def build_factory(name: str, fast: bool):
    """Return (factory, default_purge). Factories import lazily so rule-based
    runs don't need torch/xgboost installed."""
    name = name.lower()
    if name == "macd":
        from src.strategies.macd import MACDStrategy
        return (lambda: MACDStrategy(fast=12, slow=26, signal=9,
                                     hist_mom=3, zero_filter=True)), 5
    if name == "moving_average":
        from src.strategies.moving_average import MovingAverageStrategy
        return (lambda: MovingAverageStrategy(short_window=9, long_window=14,
                                              angle_threshold_deg=15.0, ma="ema",
                                              atr_window=14, vol_threshold=0.04)), 5
    if name == "bollinger":
        from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
        return (lambda: BollingerMeanReversionStrategy(window=20, k=2)), 5
    if name == "xgboost":
        from src.strategies.xgboost_regression_ML import XGBoostRegressionStrategy
        horizon = 10
        return (lambda: XGBoostRegressionStrategy(
            horizon=horizon, cv_splits=3, n_models=1, signal_thresh=0.0,
            n_iter_search=10 if fast else 50, min_features=10,
            with_hyperparam_fit=not fast, with_feature_selection=not fast,
            adjust_threshold=False)), horizon
    if name == "adaboost":
        from src.strategies.adaboost_ML import AdaBoostStrategy
        d = 10
        return (lambda: AdaBoostStrategy(
            d=d, cv_splits=3,
            param_grid={"clf__n_estimators": [50, 100, 200],
                        "clf__learning_rate": [0.1, 0.5, 1.0]},
            n_iter_search=5 if fast else 50)), d
    raise SystemExit(f"Unknown strategy '{name}'. "
                     "Choose from: macd, moving_average, bollinger, xgboost, adaboost")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True,
                   help="macd | moving_average | bollinger | xgboost | adaboost")
    p.add_argument("--symbols", nargs="+", default=["AAPL"])
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--train-days", type=int, default=756, help="~3y of daily bars")
    p.add_argument("--test-days", type=int, default=63, help="refit quarterly")
    p.add_argument("--purge", type=int, default=None,
                   help="Purge gap in days (default: the strategy's label horizon).")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rolling", action="store_true",
                   help="Rolling train window instead of expanding.")
    p.add_argument("--fast", action="store_true",
                   help="Skip Optuna/RFECV for a quick multi-fold run.")
    args = p.parse_args()

    factory, default_purge = build_factory(args.strategy, args.fast)
    purge = args.purge if args.purge is not None else default_purge

    from src.data.data_loader import fetch_yahoo_data
    end = datetime.now()
    start = end - timedelta(days=int(args.years * 365.25))
    print(f"Fetching {args.symbols} daily closes, {start.date()} -> {end.date()}...")
    data = fetch_yahoo_data(symbol=args.symbols, start=start, end=end,
                            timeframe="1d", feed=None)

    harness = WalkForwardHarness(
        factory, data, train_days=args.train_days, test_days=args.test_days,
        purge_days=purge, expanding=not args.rolling, cost_bps=args.cost_bps)
    print(f"Strategy {args.strategy} | train {args.train_days}d "
          f"{'expanding' if not args.rolling else 'rolling'} | refit every "
          f"{args.test_days}d | purge {purge}d | cost {args.cost_bps} bps/side")

    t0 = time.perf_counter()
    results = harness.run()
    elapsed = time.perf_counter() - t0

    perf = harness.performance(results)
    pd.set_option("display.float_format", lambda x: f"{x: .3f}")
    oos = results[results["test_mask"] == 1.0]
    ts = oos.index.get_level_values("timestamp")
    print(f"\n=== Walk-forward OOS results, {ts.min().date()} -> {ts.max().date()} "
          f"({elapsed:.0f}s) ===")
    print(perf.to_string())
    print("\nAll rows are out-of-sample and net of costs; Buy&Hold is gross, over "
          "the identical window. A strategy earns deployment consideration only "
          "if it beats its own Buy&Hold row on risk-adjusted metrics.")


if __name__ == "__main__":
    main()
