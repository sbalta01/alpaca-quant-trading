# examples/backtest_moving_avg.py
from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame
from matplotlib import pyplot as plt
import pandas as pd

from src.data.data_loader import fetch_nasdaq_100_symbols
from src.strategies.adaboost_ML import AdaBoostStrategy
from src.strategies.hybrid_adaboost_filter_ML import HybridAdaBoostFilterStrategy
from src.strategies.lstm_event_arima_garch_ML import LSTMEventStrategy
from src.strategies.lstm_event_technical_ML import LSTMEventTechnicalStrategy
from src.strategies.lstm_regression_ML import LSTMRegressionStrategy
from src.strategies.momentum_ranking_adaboost_ML import MomentumRankingAdaBoostStrategy
from src.execution.backtest_executor import run_backtest_strategy
from src.backtesting.visualizer import plot_returns, plot_signals

import time

from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.pair_trading import CointegrationPairTradingStrategy
from src.strategies.penalized_regression_ML import PenalizedRegressionStrategy
from src.strategies.xgboost_regression_ML import XGBoostRegressionStrategy


if __name__ == "__main__":
    # symbols = ["AAPL"]
    # symbols = ["USO"]
    # symbols = ["SPY"]
    # symbols = ["AAPL","AMZN","MSFT","GOOG","ROP", "VRTX"]
    symbols = ["MSFT"]
    # symbols = ["ROP"]
    # symbols = ["AAPL","MSFT"]
    # symbols = ["PFE"]
    # symbols = ["HAG.DE"]
    # symbols = ["RHM.DE"]
    # symbols = ["MRK"]
    # symbols = ["LMT"]
    # symbols = ["WOLF"]
    # symbols = ["IDR.MC"]
    # symbols = ["SATS"]
    # symbols = ["ECR.MC"]
    # symbols = ["HAG.DE","RHM.DE","IDR.MC","ECR.MC"]
    # symbols = ["NVDA"]
    # symbols = ["NDX"]

    # fetch_sp500_symbols()
    sp500 = pd.read_csv("sp500.csv")["Symbol"].to_list()
    sp500.remove('CEG')
    sp500.remove('GEHC')
    sp500.remove('KVUE')
    sp500.remove('VLTO')
    # symbols = sp500

    # symbols = fetch_nasdaq_100_symbols()

    start   = datetime(2020, 7, 15)
    # start   = datetime(2025, 1, 5)
    end     = datetime.now()
    # end     = datetime(2025, 6, 13)
    timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)

    # strat = MovingAverageStrategy(short_window=9, long_window=14, angle_threshold_deg = 15.0, ma = 'ema',
    #                             atr_window = 14, vol_threshold = 0.04)

    # strat = BollingerMeanReversionStrategy(window=20, k=2,)
    # strat = RandomForestStrategy(train_frac=0.7, n_estimators=100)
    # strat = RollingWindowStrategy(
    #     train_window=252,        # use ~1 year of daily bars
    #     retrain_every=5,         # retrain weekly
    #     n_estimators=200,
    #     max_depth=3,
    #     random_state=42
    # )

    # strat = AdaBoostStrategy(
    #     d=10,
    #     train_frac=0.7,
    #     cv_splits=5,
    #     param_grid={
    #         'clf__n_estimators': [50,100,200],
    #         'clf__learning_rate': [0.1,0.5,1.0]
    #     },
    #     # ratio_outliers = 1.75,
    #     n_iter_search = 50
    # )

    # predictor = AdaBoostStrategy(
    #     d=10,
    #     train_frac=0.7,
    #     cv_splits=5,
    #     param_grid={
    #         'clf__n_estimators': [50,100,200],
    #         'clf__learning_rate': [0.1,0.5,1.0]
    #     },
    #     # ratio_outliers = 1.75,
    #     n_iter_search = 50
    # )
    # strat = MomentumRankingAdaBoostStrategy(
    #     predictor=predictor,
    #     top_k=10,
    # )

    # strat = HybridAdaBoostFilterStrategy(
    #     predictor=predictor,
    #     atr_window=14,
    #     vol_threshold=0.01
    # )

    # strat = RegimeSwitchingFactorStrategy( #Usually trained daily for 7 year window
    #     regime_symbol = "HAG.DE",
    #     hmm_states = 3,
    #     vol_window = 10,
    #     vol_thresh = 0.3,
    #     ret_thresh = 0.5,
    #     random_state = 42
    # )

    # strat = CointegrationPairTradingStrategy(
    #     corr_threshold = 0.8,
    #     pvalue_threshold = 0.05,
    #     lookback=20,
    #     z_entry=2.0,
    #     z_exit=0.5
    # )

    # strat = PenalizedRegressionStrategy(
    #     train_frac=0.7,
    #     cv_splits=5,
    #     rfecv_step= 0.1,
    #     param_grid = {
    #         # 'reg__alpha':   [1e-3, 1e-2, 1e-1, 1.0, 10.0],
    #         'reg__alpha':   [0.0],
    #         # 'reg__l1_ratio':[0.1, 0.5, 0.9]
    #         'reg__l1_ratio':[0.5]
    #     },
    #     # ratio_outliers = 1.75,
    #     n_iter_search = 50
    # )

    # strat = XGBoostRegressionStrategy(
    #     horizon = 10,
    #     train_frac = 0.7,
    #     cv_splits = 5,
    #     rfecv_step = 0.1,
    #     n_models = 50,
    #     bootstrap= 0.8,
    #     signal_thresh = 0.0,
    #     n_iter_search = 25,
    #     min_features = 10, #Good to avoid killing too many
    #     # objective = 'reg:squarederror',
    #     objective = 'reg:quantileerror',
    #     quantile = 0.4, #Quantile to fit for when objective is 'reg:quantileerror'. The lower the more confidence.
    #     random_state = 48,
    #     with_hyperparam_fit = False, #Seems a bit useless
    #     with_feature_selection =True, #Could kill too many features but seems useful (prevent with min_features)
    #     adjust_threshold = False,
    # )

    strat = LSTMEventTechnicalStrategy(
        horizon=10,        # predict horizon-day return
        threshold=0.05,   # event = next-horizon-day log-return > threshold
        train_frac = 0.7,
        cv_splits = 2, #For optuna hyperparameter fitting
        n_models = 5,
        bootstrap = 0.8,
        random_state=42,
        sequences_length = 25,
        prob_positive_threshold = 0.7,
        with_hyperparam_fit = True, #Seems useful
        with_feature_attn = False,  #Seems useless
        with_pos_weight = True, #Crucial
        adjust_threshold = True, #More appropriate but obv it is safer a higher flat threshold
    )

    # strat = LSTMRegressionStrategy(
    #     horizon=20,        # predict next horizon-day return
    #     threshold=0.05,   # allow only next-horizon-day log-return > threshold signals
    #     train_frac = 0.7,
    #     cv_splits = 5, #For optuna hyperparameter fitting
    #     n_models = 5,
    #     bootstrap = 0.7,
    #     random_state=42,
    #     # sequences_length = 20,
    #     with_hyperparam_fit = True,
    #     with_feature_attn = False,
    # )

    start_backtest = time.perf_counter()

    results, results_control = run_backtest_strategy(
        strategy=strat,
        symbols=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        initial_cash_per_stock=10_000,
        # feed = 'iex'
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
    plt.show()