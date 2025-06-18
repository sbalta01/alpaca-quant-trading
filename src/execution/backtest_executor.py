# src/execution/backtest_executor.py

from datetime import datetime, timedelta
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.backtesting.backtester import BacktestEngine
from src.data.data_loader import attach_factors
from src.strategies.base_strategy import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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
    if timeframe.unit_value == TimeFrameUnit.Month:
        print('Timeframe set to Month')
        timeframe_yahoo = '1mo'
    elif timeframe.unit_value == TimeFrameUnit.Week:
        print('Timeframe set to Week')
        timeframe_yahoo = '1wk'
    elif timeframe.unit_value == TimeFrameUnit.Day:
        print('Timeframe set to Day')
        timeframe_yahoo = '1d'
    elif timeframe.unit_value == TimeFrameUnit.Hour:
        print('Timeframe set to Hour')
        timeframe_yahoo = '1h'
    elif timeframe.unit_value == TimeFrameUnit.Minute:
        print('Timeframe set to Minute')
        timeframe_yahoo = '1m'
    try:
        from src.data.data_loader import fetch_alpaca_data as fetch_data
        jkandc
        print('USING ALPACA DATA')
    except:
        from src.data.data_loader import fetch_yahoo_data as fetch_data
        print('USING YAHOO DATA')
        timeframe = timeframe_yahoo
        feed = None

    delta = end - start
    try:
        train_frac = strategy.train_frac

        df = fetch_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        feed = feed
        )
        df_single = fetch_data(
        symbol=[symbols[0]],
        start=start,
        end=end,
        timeframe=timeframe,
        feed = feed
        )
        _, df_control_single, _, _ = train_test_split(
            df_single, df_single, train_size=train_frac, shuffle=False
        )
        start_control = list(df_control_single.droplevel('symbol').index)[0]
        delta = end.date() - start_control.date()
        df_control = fetch_data(
        symbol=symbols,
        start=start_control,
        end=end,
        timeframe=timeframe,
        feed = feed
        )

    except:
        df = fetch_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        feed = feed
        )
        df_control = df.copy()
        start_control = start
    num_years = delta.days / 365

    if strategy.name == "XGBoostRegression" or strategy.name == 'LSTMEvent':
        df = attach_factors(df, timeframe=timeframe_yahoo)
        print('Macros fetched')

    print('Strategy:', strategy.name)

    # 2) Initialize and run backtest
    engine = BacktestEngine(strategy=strategy, data=df, initial_cash_per_stock=initial_cash_per_stock)
    results = engine.run()


    engine_control = BacktestEngine(strategy=BuyAndHoldStrategy(), data=df_control, initial_cash_per_stock=initial_cash_per_stock)
    results_control = engine_control.run()
    
    # 3) Compute performance
    perf = engine.performance(results,num_years)
    perf_ctrl = engine_control.performance(results_control,num_years)

    print(f"Max Drawdown       : {perf['Max Drawdown']*100:.2f}%  |  Benchmark: {perf_ctrl['Max Drawdown']*100:.2f}%  |  <-20% risky. Less negative better")
    print(f"CAGR       : {perf['CAGR']*100:.2f}%  |  Benchmark: {perf_ctrl['CAGR']*100:.2f}%  |  >5-10% good. Annualized return, more better")
    print(f"Sharpe       : {perf['Sharpe']:.2f}  |  Benchmark: {perf_ctrl['Sharpe']:.2f}  |  >1 okay, >2 good. Risk adjusted return (Penalizes volatility)")
    print(f"Sortino      : {perf['Sortino']:.2f}  |  Benchmark: {perf_ctrl['Sortino']:.2f}  |  >1.5 good. Focuses only on downside volatility")
    print(f"Calmar       : {perf['Calmar']:.2f}  |  Benchmark: {perf_ctrl['Calmar']:.2f}  |  >1 good. Higher is more reward for risk")
    print(f"Turnover     : {perf['Turnover']:.4f} |  Benchmark: {perf_ctrl['Turnover']:.4f}  |  0.1-0.5 okay/good, >1 bad. Higher is worse (unless transaction-free)")
    print(f"Fitness      : {perf['Fitness']:.2f} |  Benchmark: {perf_ctrl['Fitness']:.2f}  |  >1 strong. Good for comparing strategies (How good is my strategy per trade)")
    print(f"Profit per Trade      : {perf['Profit per Trade']*100:.2f}%  |  Benchmark: {perf_ctrl['Profit per Trade']*100:.2f}%  | How good is my strategy per trade\n")

    try:
        print("ML metrics")
        for key, value in perf['ML metrics'].items():
            print(key + "    : ", f"{value[0]:.2f}  |  ", value[1])
    except:
        pass

    print(f"\n--- Backtest: {strategy.name} on {symbols} ---")
    print(f"Period       : {start_control} → {end}")
    print(f"Initial Cash       : {perf['Initial Cash']:.2f}")
    print(f"Final Equity       : {perf['Final Equity']:.2f}  |  Benchmark: {perf_ctrl['Final Equity']:.2f}")
    print(f"Profit       : {perf['Profit']:.2f}  |  Benchmark: {perf_ctrl['Profit']:.2f}")
    print(f"Final Return       : {perf['Final Return']*100:.2f}%  |  Benchmark: {perf_ctrl['Final Return']*100:.2f}%")
    return results, results_control