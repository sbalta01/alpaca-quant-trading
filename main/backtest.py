# examples/backtest_moving_avg.py
from datetime import datetime

from alpaca.data.timeframe import TimeFrame

from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from src.execution.backtest_executor import run_backtest_strategy
from src.backtesting.visualizer import plot_equity, plot_signals
from src.strategies.random_forest_ML import RandomForestStrategy
from src.strategies.rolling_window_ML import RollingWindowStrategy


if __name__ == "__main__":
    # symbols = ["AAPL", "AMZN"]
    # symbols = "USO"
    symbols = "SPY"
    start   = datetime(2001, 1, 1)
    end     = datetime(2025, 5, 1)
    timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)

    # strat = MovingAverageStrategy(short_window=20, long_window=100, ma = 'sma')
    # strat = BollingerMeanReversionStrategy(window=20, k=2,)
    # strat = RandomForestStrategy(train_frac=0.7, n_estimators=100)
    strat = RollingWindowStrategy(
    train_window=252,        # use ~1 year of daily bars
    retrain_every=5,         # retrain weekly
    n_estimators=200,
    max_depth=3,
    random_state=42
)
    results = run_backtest_strategy(
        strategy=strat,
        symbols=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        initial_cash=10_000,
        feed = None
    )

    # print(results[['close','position','signal','equity']].groupby(level=0).tail(10))

    # Plot the equity curve
    plot_equity(results, title=f"{strat.name} Equity Curve")
    # Price with buy/sell markers for each symbol
    plot_signals(
        results,
        price_col='close',
        signal_col='signal',
        title=f"{strat.name} Signals on Price"
    )