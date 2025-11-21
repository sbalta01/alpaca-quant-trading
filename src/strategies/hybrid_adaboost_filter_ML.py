# src/strategies/hybrid_adaboost_filter_ML.py

import numpy as np
import pandas as pd
from typing import Optional

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy
from src.utils.tools import atr


class HybridAdaBoostFilterStrategy(Strategy):
    """
    Hybrid of AdaBoost MA-predictor + two filters:
      1) Angle between short & long MA exceeds threshold
      2) ATR/close exceeds volatility threshold

    Only emits a buy/sell if AdaBoost says so AND both filters agree.
    """
    name = "HybridAdaBoostFilter"
    multi_symbol = False

    def __init__(
        self,
        predictor: AdaBoostStrategy,
        atr_window: int = 14,
        vol_threshold: float = 0.01,
    ):
        """
        predictor         : an instance of AdaBoostStrategy
        atr_window        : lookback for ATR
        vol_threshold     : minimum ATR/close (e.g. 0.01 = 1%)
        """
        self.predictor = predictor
        self.atr_window = atr_window
        self.vol_thresh = vol_threshold
        self.train_frac = self.predictor.train_frac

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data: either a single-symbol DataFrame indexed by timestamp,
              or if MultiIndex, backtester will pass each symbol's slice.
        Returns same DataFrame with 'signal' column.
        """
        df = data.copy()

        # 1) Get ML signals
        df = self.predictor.generate_signals(df)
        df["position_ml"] = df["position"]
        df.drop(columns=["signal","position"])

        # 2) Compute ATR volatility filter
        df["atr"]      = atr(df, self.atr_window)
        df["vol_ok"]   = (df["atr"] / df["close"]) > self.vol_thresh

        position = 0
        positions = []
        for t, raw in zip(df.index, df["position_ml"]):
            if raw == 1.0 and (position == 0 or position == -1):
                if df["vol_ok"].at[t]:
                    position = 1
            elif raw == 0.0 and position == 1:
                if df["vol_ok"].at[t]:
                    position = -1
            positions.append(position)

        df["position"] = positions
        df["position"] = df["position"].clip(0,1) #Clip (0,1) if no short selling wanted
        df.drop(columns="position_ml", inplace=True)
        df['signal'] = df['position'].diff().fillna(0).clip(-1,1)

        return df