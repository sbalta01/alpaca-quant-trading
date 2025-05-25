# src/backtesting/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, Union
from src.strategies.base_strategy import Strategy

class BacktestEngine:
    """
    A simple backtest engine that:
      - Takes a Strategy instance and historical price data (DataFrame).
      - Calls strategy.generate_signals(data) to get 'signal' & (new) 'position' columns.
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

        # Exposed for tests/reporting
        self.cash: float = initial_cash
        self.position: float = 0.0  # number of shares held

    def _run_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest for ONE symbol (DatetimeIndex).
        """
        print(df)
        df = self.strategy.generate_signals(df).copy()

        # Initialize
        df["position"] = 0.0   # share count
        df["cash"]     = float(self.initial_cash)
        df["equity"]   = float(self.initial_cash)

        # Ensure float dtype
        df[["position", "cash", "equity"]] = df[["position", "cash", "equity"]].astype(float)

        self.cash = self.initial_cash
        self.position = 0.0

        for t in range(1, len(df)):
            price  = df["close"].iat[t]
            signal = df["signal"].iat[t]  # +1 buy, -1 sell, 0 hold

            if signal ==  1.0 and self.position == 0.0:
                # Buy all‐in
                self.position = self.cash / price
                self.cash = 0.0
            elif signal == -1.0 and self.position > 0.0:
                # Sell all‐out
                self.cash = self.position * price
                self.position = 0.0

            df.iat[t, df.columns.get_loc("position")] = self.position
            df.iat[t, df.columns.get_loc("cash")]     = self.cash
            df.iat[t, df.columns.get_loc("equity")]   = self.cash + self.position * price

        df["returns"] = df["equity"].pct_change().fillna(0.0)
        return df

    def run(self) -> pd.DataFrame:
        """
        If self.data is multi‐symbol (MultiIndex), run each symbol separately
        and concatenate the results with a MultiIndex. Otherwise, run single.
        """
        df = self.data

        if isinstance(df.index, pd.MultiIndex):
            # Expect levels: ['symbol','timestamp']
            results = []
            for symbol, subdf in df.groupby(level="symbol"):
                # drop the symbol level for single‐symbol run
                single = subdf.droplevel("symbol")
                out = self._run_single(single)
                # reattach symbol level
                out["symbol"] = symbol
                results.append(out)
            # concat and re‐set MultiIndex(symbol, timestamp)
            final = pd.concat(results)
            final = final.set_index("symbol", append=True)
            final = final.reorder_levels(["symbol", final.index.names[0]])
            return final.sort_index()
        else:
            return self._run_single(df)

    def performance(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Compute basic performance metrics on the backtest results:
          - total_return       : (final equity / initial cash) − 1
          - annualized_sharpe  : Sharpe ratio assuming daily bars (252 trading days)
          - max_drawdown       : worst peak‐to‐trough decline
        """
        total_return = results["equity"].iat[-1] / self.initial_cash - 1.0

        mean_ret = results["returns"].mean()
        std_ret  = results["returns"].std(ddof=1) or np.nan
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

        cumulative_max = results["equity"].cummax()
        drawdowns = (results["equity"] - cumulative_max) / cumulative_max
        max_dd = drawdowns.min()

        return {
            "total_return": total_return,
            "annualized_sharpe": sharpe,
            "max_drawdown": max_dd,
        }
