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
    initial_cash: float = 10_000.0,
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
    initial_cash: float
    
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

        print(start_control)

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
    engine = BacktestEngine(strategy=strategy, data=df, initial_cash=initial_cash)
    results = engine.run()

    engine_control = BacktestEngine(strategy=BuyAndHoldStrategy(), data=df_control, initial_cash=initial_cash)
    results_control = engine_control.run()


    
    # 3) Print summary
    pnl = results["equity"].iloc[-1] - initial_cash
    ret = pnl / initial_cash * 100.0
    pnl_control = results_control["equity"].iloc[-1] - initial_cash
    ret_control = pnl_control / initial_cash * 100.0
    max_drawdown = ((results['equity'] / results['equity'].cummax()) - 1).min()*100
    max_drawdown_control = ((results_control['equity'] / results_control['equity'].cummax()) - 1).min()*100
    cagr = ((results['equity'].iloc[-1] / initial_cash) ** (1 / num_years) - 1)*100
    cagr_control = ((results_control['equity'].iloc[-1] / initial_cash) ** (1 / num_years) - 1)*100
    print(f"--- Backtest: {strategy.name} on {symbols} ---")
    print(f"Period       : {start.date()} → {end.date()}")
    print(f"Initial Cash : {initial_cash:,.2f}")
    print(f"Final Equity : {results['equity'].iloc[-1]:,.2f}")
    print(f"Net P&L      : {pnl:,.2f}")
    print(f"Max Drawdown (%)   : {max_drawdown:,.2f}%") ##Highest possible loss during the process
    print(f"Max D Control (%)   : {max_drawdown_control:,.2f}%")
    print(f"CAGR (%)   : {cagr:,.2f}%")
    print(f"CAGR Control(%)   : {cagr_control:,.2f}%")
    print(f"Return (%)   : {ret:,.2f}%")
    print(f"Return Control (%)   : {ret_control:,.2f}%\n")
    
    return results