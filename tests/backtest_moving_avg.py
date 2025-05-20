# examples/backtest_moving_avg.py

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.data.data_loader import fetch_alpaca_data
from src.strategies.moving_average import MovingAverageStrategy
from src.backtesting.backtester import BacktestEngine
from alpaca.data.timeframe import TimeFrame

from src.backtesting.visualizer import plot_equity, plot_signals

def main():
    # ————————————————————————
    # Fetch data from Alpaca
    # ————————————————————————
    symbol = "AAPL"
    start  = datetime(2025, 1, 1)
    end    = datetime(2025, 5, 1)
    data = fetch_alpaca_data(symbol, start, end, timeframe=TimeFrame.Day)

    # ————————————————————————
    # Configure & run strategy
    # ————————————————————————
    strategy = MovingAverageStrategy(short_window=10, long_window=20)
    engine   = BacktestEngine(strategy, data, initial_cash=10_000)
    results  = engine.run()

    # ————————————————————————
    # Report
    # ————————————————————————

    print(f"\n=== Backtest: {strategy.name} {symbol} ===")
    print(f"Period      : {start.date()} → {end.date()}")
    print(f"Initial Cash: {engine.initial_cash:.2f}")
    print(f"Final Equity: {results['equity'].iloc[-1]:.2f}")
    print(f"Return (%)  : {(results['equity'].iloc[-1] / engine.initial_cash - 1) * 100:.2f}%\n")
    print(results[['close','sma_short','sma_long','position','signal','equity']].tail(10))

    # Equity curve
    plot_equity(results, title=f"{strategy.name} Equity Curve")

    # Price with entry/exit markers
    plot_signals(results, price_col='close', signal_col='signal',
                title=f"{strategy.name} Signals on Price")

if __name__ == "__main__":
    main()