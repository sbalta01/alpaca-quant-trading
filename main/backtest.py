# examples/backtest_moving_avg.py
from datetime import datetime

from alpaca.data.timeframe import TimeFrame

from src.strategies.adaboost_ML import AdaBoostStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.execution.backtest_executor import run_backtest_strategy
from src.backtesting.visualizer import plot_equity, plot_signals
from src.strategies.random_forest_ML import RandomForestStrategy
from src.strategies.rolling_window_ML import RollingWindowStrategy


if __name__ == "__main__":
    # symbols = ["AAPL"]
    # symbols = "USO"
    # symbols = "SPY"
    symbols = ["AAPL","AMZN","MSFT","GOOG"]
    # # fetch_sp500_symbols()
    # sp500 = pd.read_csv("sp500.csv")["Symbol"].tolist()
    # sp500.remove("GEV")
    # sp500.remove("SOLV")
    # symbols = sp500
    
    start   = datetime(2015, 1, 1)
    end     = datetime(2025, 5, 25)
    timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)

    # strat = MovingAverageStrategy(short_window=5, long_window=20, ma = 'sma')
    # strat = BollingerMeanReversionStrategy(window=20, k=2,)
    # strat = RandomForestStrategy(train_frac=0.7, n_estimators=100)

    # strat = RollingWindowStrategy(
    #     train_window=252,        # use ~1 year of daily bars
    #     retrain_every=5,         # retrain weekly
    #     n_estimators=200,
    #     max_depth=3,
    #     random_state=42
    # )
    strat = AdaBoostStrategy(
        d=10, #10 has the best
        train_frac=0.7,
        cv_splits=5,
        param_grid={
            'clf__n_estimators': [50,100,200],
            'clf__learning_rate': [0.1,0.5,1.0]
        }
    )

    # predictor = AdaBoostStrategy(
    # d=10,
    # train_frac=0.7,
    # cv_splits=5,
    # param_grid={
    #     'clf__n_estimators': [50,100],
    #     'clf__learning_rate': [0.5,1.0]
    # }
    # )
    # strat = RankingTopKStrategy(predictor=predictor, top_k=1, n_jobs=-1)



    results = run_backtest_strategy(
        strategy=strat,
        symbols=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        initial_cash=10_000,
        feed = "sip"
    )

    # Plot the equity curve
    plot_equity(results, title=f"{strat.name} Equity Curve")
    # Price with buy/sell markers for each symbol
    plot_signals(
        results,
        price_col='close',
        signal_col='signal',
        title=f"{strat.name} Signals on Price"
    )