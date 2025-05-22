# examples/backtest_moving_avg.py

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from src.data.data_loader import fetch_alpaca_data
from src.strategies.moving_average import MovingAverageStrategy
from src.backtesting.backtester import BacktestEngine
from src.backtesting.visualizer import plot_equity, plot_signals

def main():
    # —————————————————————————————
    # 1) Choose symbol(s) & date range
    # —————————————————————————————
    # Can be a single ticker str, or a list of tickers:
    symbols = ["AAPL", "AMZN"]
    start   = datetime(2025, 1, 1)
    end     = datetime(2025, 5, 1)

    # —————————————————————————————
    # 2) Fetch data from Alpaca
    # —————————————————————————————
    df = fetch_alpaca_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=TimeFrame.Day
    )

    # —————————————————————————————
    # 3) Set up & run the backtest
    # —————————————————————————————
    strat  = MovingAverageStrategy(short_window=10, long_window=20)
    engine = BacktestEngine(strat, df, initial_cash=10_000)
    results = engine.run()

    # —————————————————————————————
    # 4) Print summary
    # —————————————————————————————
    print(f"\n=== Backtest: {strat.name} on {symbols} ===")
    print(f"Period       : {start.date()} → {end.date()}")
    print(f"Initial Cash : {engine.initial_cash:,.2f}")
    print(f"Final Equity : {results['equity'].iloc[-1]:,.2f}")
    print(f"Return (%)   : {(results['equity'].iloc[-1] / engine.initial_cash - 1) * 100:,.2f}%\n")

    # Show last 10 rows for each symbol
    # (MultiIndex: symbol, timestamp)
    print(results[['close','sma_short','sma_long','position','signal','equity']].groupby(level=0).tail(10))

    # —————————————————————————————
    # 5) Visualize
    # —————————————————————————————
    # Equity curve (all symbols together)
    plot_equity(results, title=f"{strat.name} Equity Curve")

    # Price with buy/sell markers for each symbol
    plot_signals(
        results,
        price_col='close',
        signal_col='signal',
        title=f"{strat.name} Signals on Price"
    )

if __name__ == "__main__":
    main()
