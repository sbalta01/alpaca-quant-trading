# src/strategies/hybrid_adaboost_filter_ML.py

import numpy as np
import pandas as pd
from typing import Optional

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy


class HybridAdaBoostFilterStrategy(Strategy):
    """
    Hybrid of AdaBoost MA‐predictor + two filters:
      1) Angle between short & long MA exceeds threshold
      2) ATR/close exceeds volatility threshold

    Only emits a buy/sell if AdaBoost says so AND both filters agree.
    """
    name = "HybridAdaBoostFilter"
    multi_symbol = False  # will be backtested per symbol

    def __init__(
        self,
        predictor: AdaBoostStrategy,
        short_ma: int = 10,
        long_ma: int = 50,
        angle_threshold_deg: float = 10.0,
        atr_window: int = 14,
        vol_threshold: float = 0.01,
    ):
        """
        predictor         : an instance of AdaBoostStrategy
        short_ma, long_ma : windows for MA angle filter
        angle_threshold_deg: minimum angle in degrees
        atr_window        : lookback for ATR
        vol_threshold     : minimum ATR/close (e.g. 0.01 = 1%)
        """
        self.predictor = predictor
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.angle_thresh = np.radians(angle_threshold_deg)
        self.atr_window = atr_window
        self.vol_thresh = vol_threshold
        self.train_frac = self.predictor.train_frac

    @staticmethod
    def _atr(df: pd.DataFrame, window: int) -> pd.Series:
        """Average True Range over `window`."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low  - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()

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

        # 2) Compute MA angle filter
        df["ma_short"] = df["close"].rolling(self.short_ma).mean()
        df["ma_long"]  = df["close"].rolling(self.long_ma).mean()
        df["ma_diff"]  = df["ma_short"] - df["ma_long"]
        df["angle"]    = np.arctan(df["ma_diff"])
        df["trend_ok"] = df["angle"].abs() > self.angle_thresh

        # 3) Compute ATR volatility filter
        df["atr"]      = self._atr(df, self.atr_window)
        df["vol_ok"]   = (df["atr"] / df["close"]) > self.vol_thresh

        # 4) Combine into final signal
        df["position_raw"] = -2.0 #New arbitrary number to tell cases apart
        # buy
        buy_mask = (df["position_ml"] ==  1.0) & df["trend_ok"] & df["vol_ok"]
        # sell
        sell_mask = (df["position_ml"] == 0.0) & df["trend_ok"] & df["vol_ok"]
        df.loc[buy_mask,  "position_raw"] =  1.0
        df.loc[sell_mask, "position_raw"] = 0.0

        # Enforce stateful transitions (Avoid double buy/sell)
        position = 0
        signals = []
        for t, raw in zip(df.index, df["position_raw"]):
            if raw == 1.0 and position == 0:
                sig = 1.0
                position = 1
            elif raw == 0.0 and position == 1:
                sig = -1.0
                position = 0
            else:
                sig = 0.0
            signals.append(sig)

        df["signal"] = signals
        df.drop(columns="position_raw", inplace=True)

        # 5) Clean up intermediate cols if you like:
        # df.drop(columns=["signal_ml","ma_short","ma_long","ma_diff","angle","atr","trend_ok","vol_ok"], inplace=True)

        return df