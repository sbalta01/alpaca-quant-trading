# src/backtesting/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, Union
from joblib import Parallel, delayed
from src.strategies.base_strategy import Strategy

class BacktestEngine:
    """
    Backtest engine that:
      - Takes a Strategy instance and historical price data (DataFrame).
      - Calls strategy.generate_signals(data) to get 'signal' & (new) 'position' columns.
      - Simulates P&L over time assuming:
          * Full allocation on each trade (all‐in / all‐out)
          * Zero transaction costs
          * No leverage
      - Computes basic performance metrics.
      - Parallelizes one‐symbol strategies.
      - Correctly handles multi‐symbol strategies (e.g. momentum ranking).
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_cash: float = 10_000.0,
        n_jobs: int = -1
    ):
        self.strategy = strategy
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.n_jobs = n_jobs

        # Exposed for reporting
        self.cash: float = initial_cash
        self.position: float = 0.0

    def _simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame indexed by timestamp and containing 'close' and a precomputed
        'signal' column (+1 buy, -1 sell, 0 hold), simulate position, cash, equity.
        """
        df = df.copy()
        df["position"] = 0.0
        df["cash"]     = float(self.initial_cash)
        df["equity"]   = float(self.initial_cash)
        df[["position","cash","equity"]] = df[["position","cash","equity"]].astype(float)

        self.cash = self.initial_cash
        self.position = 0.0

        for t in range(1, len(df)):
            price  = df["close"].iat[t]
            signal = df["signal"].iat[t]

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

    def _run_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest for one symbol: generate signals then simulate.
        """
        df_with_signals = self.strategy.generate_signals(df)
        return self._simulate(df_with_signals)

    def run(self) -> pd.DataFrame:
        """
        Run the backtest.  
        - If multi_symbol=False: treat data as either single-symbol or multi-index, but
          run _run_single per symbol in parallel if needed.
        - If multi_symbol=True: get signals for the whole multi-index, then simulate per symbol.
        Returns a DataFrame:
          - Single-symbol: DatetimeIndex
          - Multi-symbol: MultiIndex ['symbol','timestamp']
        """
        df = self.data

        # Multi-symbol strategy
        if getattr(self.strategy, "multi_symbol", False):
            # 1) Get multi-symbol signals
            signals = self.strategy.generate_signals(df)
            # 2) Merge signals onto price data
            #    `signals` and `df` share the same MultiIndex
            merged = df.join(signals[["signal"]], how="inner")
            # 3) Simulate each symbol in parallel
            groups = [
                (symbol, subdf.droplevel("symbol"))
                for symbol, subdf in merged.groupby(level="symbol")
            ]
            outs = Parallel(n_jobs=self.n_jobs)(
                delayed(self._simulate)(subdf) for _, subdf in groups
            )
            # 4) Reattach symbol level and concat
            for (symbol, _), out in zip(groups, outs):
                out["symbol"] = symbol
            result = pd.concat(outs)
            result = result.set_index("symbol", append=True)
            result = result.reorder_levels(["symbol", result.index.names[0]])
            return result.sort_index()

        # Single-symbol strategy
        # If data is MultiIndex, run per symbol; else run once
        if isinstance(df.index, pd.MultiIndex):
            groups = [
                (symbol, subdf.droplevel("symbol"))
                for symbol, subdf in df.groupby(level="symbol")
            ]
            outs = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_single)(subdf) for _, subdf in groups
            )
            for (symbol, _), out in zip(groups, outs):
                out["symbol"] = symbol
            result = pd.concat(outs)
            result = result.set_index("symbol", append=True)
            result = result.reorder_levels(["symbol", result.index.names[0]])
            return result.sort_index()
        else:
            return self._run_single(df)