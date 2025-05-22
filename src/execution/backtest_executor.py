# src/execution/backtest_executor.py

from datetime import datetime
from typing import Union, List

import pandas as pd

from src.data.data_loader import fetch_alpaca_data
from src.backtesting.backtester import BacktestEngine
from src.strategies.base_strategy import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy

def run_backtest_strategy(
    strategy: Strategy,
    symbols: Union[str, List[str]],
    start: datetime,
    end: datetime,
    timeframe: Union[pd.Timedelta, object],  # e.g. TimeFrame.Day. Fetch data time frame.
    initial_cash: float = 10_000.0
) -> pd.DataFrame:
    """
    Run a backtest of `strategy` on `symbols` between `start` and `end`.
    
    Parameters
    ----------
    strategy : Strategy
    symbols  : str or list of str
    start    : datetime
    end      : datetime
    timeframe: alpaca TimeFrame or pandas Timedelta
    initial_cash: float
    
    Returns
    -------
    results : pd.DataFrame
        MultiIndex [symbol, timestamp] if multiple symbols, else DatetimeIndex.
    """
    # 1) Fetch historical data
    df = fetch_alpaca_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe
    )
    
    # 2) Initialize and run backtest
    engine = BacktestEngine(strategy=strategy, data=df, initial_cash=initial_cash)
    results = engine.run()

    engine_control = BacktestEngine(strategy=BuyAndHoldStrategy(), data=df, initial_cash=initial_cash)
    results_control = engine_control.run()

    
    # 3) Print summary
    pnl = results["equity"].iloc[-1] - initial_cash
    ret = pnl / initial_cash * 100.0
    pnl_control = results_control["equity"].iloc[-1] - initial_cash
    ret_control = pnl_control / initial_cash * 100.0
    print(f"--- Backtest: {strategy.name} on {symbols} ---")
    print(f"Period       : {start.date()} → {end.date()}")
    print(f"Initial Cash : {initial_cash:,.2f}")
    print(f"Final Equity : {results['equity'].iloc[-1]:,.2f}")
    print(f"Net P&L      : {pnl:,.2f}")
    print(f"Return (%)   : {ret:,.2f}%")
    print(f"Return Control (%)   : {ret_control:,.2f}%\n")
    
    return results