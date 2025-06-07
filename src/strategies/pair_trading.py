# src/strategies/pair_trading.py

import numpy as np
import pandas as pd
from typing import Tuple
from src.strategies.base_strategy import Strategy

class PairTradingStrategy(Strategy):
    """
    Basic pair-trading on log-price spread z-score.
    Uses one loop over time to maintain state; zero signals for all other symbols.
    """
    name = "PairTrading"
    multi_symbol = True

    def __init__(
        self,
        pair: Tuple[str, str],
        lookback: int = 20,
        z_entry: float = 2.0,
        z_exit: float = 0.5
    ):
        """
        pair      : (sym1, sym2)
        lookback  : rolling window for mean/std of spread
        z_entry   : z-score threshold to enter
        z_exit    : z-score threshold to exit
        """
        self.sym1, self.sym2 = pair
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 2) Extract each leg's close price
        df1 = data.xs(self.sym1, level="symbol")["close"]
        df2 = data.xs(self.sym2, level="symbol")["close"]

        # 3) Compute log‐spread and rolling z‐score
        spread = np.log(df1) - np.log(df2)
        mean_sp = spread.rolling(self.lookback).mean()
        std_sp  = spread.rolling(self.lookback).std()
        zscore  = (spread - mean_sp) / std_sp

        # 4) Stateful position loop over time
        #   pos = +1 means long spread (long sym1, short sym2)
        #   pos = -1 means short spread (short sym1, long sym2)
        pos1 = pd.Series(0.0, index=zscore.index)
        pos2 = pd.Series(0.0, index=zscore.index)
        position = 0  # 0=flat, +1=long spread, -1=short spread

        for t in zscore.index:
            z = zscore.loc[t]
            if position == 0:
                if z > self.z_entry:
                    position = -1
                elif z < -self.z_entry:
                    position = 1
            elif position == 1 and z > -self.z_exit:
                position = 0
            elif position == -1 and z < self.z_exit:
                position = 0

            pos1.at[t] = position
            pos2.at[t] = -position

        timestamps= zscore.index
        out = data.copy()
        idx1 = pd.MultiIndex.from_product([[self.sym1], timestamps], names=["symbol","timestamp"])
        idx2 = pd.MultiIndex.from_product([[self.sym2], timestamps], names=["symbol","timestamp"])
        out.loc[idx1, "position"] = pos1.values
        out.loc[idx2, "position"] = pos2.values
        out["position"] = out["position"].clip(0,1) #No short selling for the moment
        out["signal"] = out.groupby(level="symbol")["position"].diff().fillna(0.0)
        return out