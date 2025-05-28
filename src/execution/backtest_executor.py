# src/execution/backtest_executor.py

from datetime import datetime, timedelta
from typing import Union, List

import pandas as pd

from src.data.data_loader import fetch_alpaca_data
from src.backtesting.backtester import BacktestEngine
from src.strategies.base_strategy import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from alpaca.data.timeframe import TimeFrame

def run_backtest_strategy(
    strategy: Strategy,
    symbols: Union[str, List[str]],
    start: datetime,
    end: datetime,
    timeframe: TimeFrame,  # e.g. TimeFrame.Day. Fetch data time frame.
    initial_cash_per_stock: float = 10_000.0,
    feed: str = 'iex'
) -> pd.DataFrame:
    """
    Run a backtest of `strategy` on `symbols` between `start` and `end`.
    
    Parameters
    ----------
    strategy : Strategy
    symbols  : str or list of str
    start    : datetime
    end      : datetime
    timeframe: alpaca TimeFrame
    initial_cash_per_stock: float
    
    Returns
    -------
    results : pd.DataFrame
        MultiIndex [symbol, timestamp] if multiple symbols, else DatetimeIndex.
    """
    # 1) Fetch historical data
    delta = end - start
    try:
        train_frac = strategy.train_frac

        df = fetch_alpaca_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        feed = feed
        )

        start_control = end - timedelta(days=delta.days*(1-train_frac))
        delta = end - start_control

        df_control = fetch_alpaca_data(
        symbol=symbols,
        start=start_control,
        end=end,
        timeframe=timeframe,
        feed = feed
        )

    except:
        df = fetch_alpaca_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        feed = feed
        )
        df_control = df.copy()
    num_years = delta.days / 365

    # 2) Initialize and run backtest
    engine = BacktestEngine(strategy=strategy, data=df, initial_cash_per_stock=initial_cash_per_stock)
    results = engine.run()


    engine_control = BacktestEngine(strategy=BuyAndHoldStrategy(), data=df_control, initial_cash_per_stock=initial_cash_per_stock)
    results_control = engine_control.run()
    
    # 3) Compute performance
    perf = engine.performance(results,num_years)
    perf_ctrl = engine_control.performance(results_control,num_years)

    print(f"--- Backtest: {strategy.name} on {symbols} ---")
    print(f"Period       : {start.date()} → {end.date()}")
    print(f"Initial Cash       : {perf['Initial Cash']:.2f}")
    print(f"Final Equity       : {perf['Final Equity']:.2f}  |  Benchmark: {perf_ctrl['Final Equity']:.2f}")
    print(f"Profit       : {perf['Profit']:.2f}  |  Benchmark: {perf_ctrl['Profit']:.2f}")
    print(f"Max Drawdown       : {perf['Max Drawdown']*100:.2f}%  |  Benchmark: {perf_ctrl['Max Drawdown']*100:.2f}%")
    print(f"CAGR       : {perf['CAGR']*100:.2f}%  |  Benchmark: {perf_ctrl['CAGR']*100:.2f}%")
    print(f"Final Return       : {perf['Final Return']*100:.2f}%  |  Benchmark: {perf_ctrl['Final Return']*100:.2f}%\n")
    print(f"Sharpe       : {perf['Sharpe']:.2f}  |  Benchmark: {perf_ctrl['Sharpe']:.2f}")
    print(f"Sortino      : {perf['Sortino']:.2f}  |  Benchmark: {perf_ctrl['Sortino']:.2f}")
    print(f"Calmar       : {perf['Calmar']:.2f}  |  Benchmark: {perf_ctrl['Calmar']:.2f}")
    print(f"Turnover     : {perf['Turnover']:.4f} |  Benchmark: {perf_ctrl['Turnover']:.4f}")
    print(f"Fitness      : {perf['Fitness']:.2f} |  Benchmark: {perf_ctrl['Fitness']:.2f}")    
    return results, results_control