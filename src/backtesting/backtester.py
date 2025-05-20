# src/backtesting/backtester.py

import pandas as pd
import numpy as np
from typing import Dict
from src.strategies.base_strategy import Strategy

class BacktestEngine:
    """
    A simple backtest engine that:
      - Takes a Strategy instance and historical price data (DataFrame).
      - Calls strategy.generate_signals(data) to get 'signal' & 'position' columns.
      - Simulates P&L over time assuming:
          * Full allocation on each trade (all‐in / all‐out)
          * Zero transaction costs
          * No leverage
      - Computes basic performance metrics.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_cash: float = 10_000.0,
    ):
        self.strategy = strategy
        self.data = data.copy()
        self.initial_cash = initial_cash

        # Exposed attributes for tests / reporting:
        self.cash: float = initial_cash
        self.position: float = 0.0  # number of shares held

    def run(self) -> pd.DataFrame:
        df = self.strategy.generate_signals(self.data).copy()

        # — Initialize bookkeeping columns as floats —
        df["holdings"] = float(0.0)
        df["cash"]     = float(self.initial_cash)
        df["equity"]   = float(self.initial_cash)

        # Reset engine state
        self.cash = self.initial_cash
        self.position = 0.0

        for t in range(1, len(df)):
            price  = df["close"].iat[t]
            signal = df["signal"].iat[t]

            if signal == 1.0 and self.position == 0:
                self.position = self.cash / price
                self.cash = 0.0 #Assumes I spend all my cash
            elif signal == -1.0 and self.position > 0:
                self.cash = self.position * price
                self.position = 0.0 #Assumes I liquidate all my stocks

            # Now we can safely assign floats into these columns
            df.at[df.index[t], "holdings"] = self.position * price
            df.at[df.index[t], "cash"]     = self.cash
            df.at[df.index[t], "equity"]   = self.cash + (self.position * price)

        df["returns"] = df["equity"].pct_change().fillna(0.0)
        return df


    def performance(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Compute basic performance metrics on the backtest results:
          - total_return       : (final equity / initial cash) − 1
          - annualized_sharpe  : Sharpe ratio assuming daily bars (252 trading days)
          - max_drawdown       : worst peak‐to‐trough decline
        """
        total_return = results["equity"].iat[-1] / self.initial_cash - 1.0

        # annualized Sharpe
        mean_ret = results["returns"].mean()
        std_ret  = results["returns"].std(ddof=1) or np.nan
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

        # max drawdown
        cumulative_max = results["equity"].cummax()
        drawdowns = (results["equity"] - cumulative_max) / cumulative_max
        max_dd = drawdowns.min()

        return {
            "total_return": total_return,
            "annualized_sharpe": sharpe,
            "max_drawdown": max_dd,
        }
