# src/strategies/moving_average.py

import numpy as np
from src.strategies.base_strategy import Strategy
from src.utils.indicators import sma, ema, rsi
import pandas as pd

class MovingAverageStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy:
      - Goes long when the short MA crosses above the long SMA
      - Exits (goes flat) when the short MA crosses below the long SMA
      - Choose between Simple MA, Exponential MA or Relative Strength Index
    """
    name = "MovingAverage"

    def __init__(self, short_window: int = 5, long_window: int = 20, 
        angle_threshold_deg: float = 0.0, ma: str = 'sma'):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.angle_threshold_rad = np.radians(angle_threshold_deg)
        self.ma = ma

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame with at least a 'close' column, compute:
          - 'ma_short': average over short_window
          - 'ma_long': average over long_window
          - 'position':  1.0 when ma_short > ma_long, 0.0 otherwise
          - 'signal': difference of 'signal' to mark entry/exit
        Returns a DataFrame with these additional columns.
        """
        df = data.copy()
        if self.ma == 'sma':
            df['ma_short'] = sma(df['close'], self.short_window)
            df['ma_long']  = sma(df['close'], self.long_window)
        elif self.ma == 'ema':
            df['ma_short'] = ema(df['close'], self.short_window)
            df['ma_long']  = ema(df['close'], self.long_window)
            
        elif self.ma == 'rsi':
            df['ma_short'] = rsi(df['close'], self.short_window)
            df['ma_long']  = rsi(df['close'], self.long_window)
            

        # Generate raw positions
        df['position'] = (df['ma_short'] > df['ma_long']).astype(float)

        # df['position'] = (df['close'] > df['ma_short']).astype(float)
        df['ma_diff'] = df['ma_short'] - df['ma_long']
        df['angle'] = np.arctan(df['ma_diff'])

        df['position'] = (
            (df['ma_short'] > df['ma_long']) &
            (df['angle'].abs() > self.angle_threshold_rad)
        ).astype(float)

        # Generate trading orders: +1 for a buy, -1 for a sell
        df['signal'] = df['position'].diff().fillna(0)

        return df
