# examples/backtest_moving_avg.py
from datetime import datetime

from alpaca.data.timeframe import TimeFrame

from src.strategies.moving_average import MovingAverageStrategy
from src.execution.backtest_executor import run_backtest_strategy
from src.backtesting.visualizer import plot_equity, plot_signals


if __name__ == "__main__":
    symbols = ["AAPL", "AMZN"]
    start   = datetime(2023, 1, 1)
    end     = datetime(2025, 5, 1)
    timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)

    strat = MovingAverageStrategy(short_window=20, long_window=100, ma = 'sma')
    results = run_backtest_strategy(
        strategy=strat,
        symbols=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        initial_cash=10_000
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