# examples/backtest_moving_avg.py
from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame
from matplotlib import pyplot as plt
import pandas as pd

from src.data.data_loader import fetch_sp500_symbols
from src.strategies.adaboost_ML import AdaBoostStrategy
from src.strategies.momentum_ranking_adaboost_ML import MomentumRankingAdaBoostStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.execution.backtest_executor import run_backtest_strategy
from src.backtesting.visualizer import plot_returns, plot_signals
from src.strategies.random_forest_ML import RandomForestStrategy
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

    start   = datetime(2023, 5, 10)
    end     = datetime(2025, 5, 28)
    # end     = datetime(2025, 1, 1)
    timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)

    # strat = MovingAverageStrategy(short_window=9, long_window=20, angle_threshold_deg = 45.0, ma = 'ema')
    # strat = BollingerMeanReversionStrategy(window=20, k=2,)
    # strat = RandomForestStrategy(train_val_frac=0.7, n_estimators=100)
    # strat = RollingWindowStrategy(
    #     train_window=252,        # use ~1 year of daily bars
    #     retrain_every=5,         # retrain weekly
    #     n_estimators=200,
    #     max_depth=3,
    #     random_state=42
    # )

    strat = AdaBoostStrategy(
        d=10, #10 has the best
        train_val_frac=0.7,
        val_ratio= 0.25,
        cv_splits=5,
        param_grid={
            'clf__n_estimators': [50,100,200],
            'clf__learning_rate': [0.1,0.5,1.0]
        }
    )

    # predictor = AdaBoostStrategy(
    #     d=5,
    #     train_val_frac=0.7,
    #     cv_splits=5,
    #     param_grid={
    #         'clf__n_estimators': [50, 100],
    #         'clf__learning_rate': [0.5, 1.0]
    #     }
    # )
    # strat = MomentumRankingAdaBoostStrategy(
    #     predictor=predictor,
    #     top_k=1,
    #     n_jobs=-1
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