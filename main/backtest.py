# examples/backtest_moving_avg.py
from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame
import pandas as pd

from src.data.data_loader import fetch_sp500_symbols
from src.strategies.adaboost_ML import AdaBoostStrategy
from src.strategies.hybrid_adaboost_filter_ML import HybridAdaBoostFilterStrategy
from src.strategies.momentum_ranking_adaboost_ML import MomentumRankingAdaBoostStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.execution.backtest_executor import run_backtest_strategy
from src.backtesting.visualizer import plot_returns, plot_signals
from src.strategies.random_forest_ML import RandomForestStrategy
from src.strategies.regime_switching_factor_ML import RegimeSwitchingFactorStrategy
from src.strategies.rolling_window_ML import RollingWindowStrategy

import time


if __name__ == "__main__":
    # symbols = ["AAPL"]
    # symbols = ["USO"]
    # symbols = ["SPY"]
    # symbols = ["AAPL","AMZN","MSFT","GOOG"]
    symbols = ["AAPL","AMZN"]

    # fetch_sp500_symbols()
    sp500 = pd.read_csv("sp500.csv")["Symbol"].tolist()
    sp500.remove('CEG')
    sp500.remove('GEHC')
    sp500.remove('KVUE')
    sp500.remove('VLTO')
    # symbols = sp500

    # start   = datetime(2015, 1, 1)
    start   = datetime(2023, 1, 1)
    end     = datetime(2025, 5, 28)
    # end     = datetime(2025, 1, 1)
    timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)

    # strat = MovingAverageStrategy(short_window=9, long_window=20, angle_threshold_deg = 45.0, ma = 'ema')
    # strat = BollingerMeanReversionStrategy(window=20, k=2,)
    # strat = RandomForestStrategy(train_frac=0.7, n_estimators=100)
    # strat = RollingWindowStrategy(
    #     train_window=252,        # use ~1 year of daily bars
    #     retrain_every=5,         # retrain weekly
    #     n_estimators=200,
    #     max_depth=3,
    #     random_state=42
    # )

    # predictor = AdaBoostStrategy(
    #     d=10,
    #     train_frac=0.7,
    #     cv_splits=5,
    #     # leave param_grid=None for broad RandomizedSearch
    #     # ratio_outliers = 1.75,
    #     n_iter_search = 50
    # )
    # strat = MomentumRankingAdaBoostStrategy(
    #     predictor=predictor,
    #     top_k=1,
    #     n_jobs=-1
    # )

    strat = AdaBoostStrategy(
        d=10, #10 has the best
        train_frac=0.7,
        cv_splits=5,
        param_grid={
            'clf__n_estimators': [50,100,200],
            'clf__learning_rate': [0.1,0.5,1.0]
        },
        # ratio_outliers = 1.75,
        n_iter_search = 50
    )

    # predictor = AdaBoostStrategy(
    #     d=10,
    #     train_frac=0.7,
    #     cv_splits=5,
    #     param_grid={
    #         'clf__n_estimators': [50,100,200],
    #         'clf__learning_rate': [0.1,0.5,1.0]
    #     },
    #     # ratio_outliers = 3.00,
    #     n_iter_search = 50
    # )
    # strat = HybridAdaBoostFilterStrategy(
    #     predictor=predictor,
    #     short_ma=9,
    #     long_ma=20,
    #     angle_threshold_deg=10,
    #     atr_window=14,
    #     vol_threshold=0.01
    # )

    # strat = RegimeSwitchingFactorStrategy(
    #     regime_symbol = "SPY",
    #     hmm_states = 3,
    #     hmm_window = 2707,
    #     vol_window = 10,
    #     return_col = "close",
    #     vol_threshold = 0.3,
    #     ret_threshold = 0.5,
    #     random_state = 42
    # )

    start_backtest = time.perf_counter()

    results, results_control = run_backtest_strategy(
        strategy=strat,
        symbols=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        initial_cash_per_stock=10_000,
        feed = None
    )

    end_backtest = time.perf_counter()
    elapsed_seconds = end_backtest - start_backtest
    formatted = str(timedelta(seconds=elapsed_seconds))
    print(f"Elapsed time: {formatted}")

    # Plot the equity curve
    plot_returns(results, results_control, title=f"{strat.name} Equity Curve")
    # Price with buy/sell markers for each symbol
    plot_signals(
        results,
        results_control,
        price_col='close',
        signal_col='signal',
        title=f"{strat.name} Signals on Price"
    )