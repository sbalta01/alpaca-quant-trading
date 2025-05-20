# src/strategies/moving_average.py

from src.strategies.base_strategy import Strategy
from src.utils.indicators import sma
import pandas as pd

class MovingAverageStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy:
      - Goes long when the short SMA crosses above the long SMA
      - Exits (goes flat) when the short SMA crosses below the long SMA
    """
    name = "MovingAverage"

    def __init__(self, short_window: int = 5, long_window: int = 20):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame with at least a 'close' column, compute:
          - 'sma_short': rolling mean over short_window
          - 'sma_long': rolling mean over long_window
          - 'signal':  1.0 when sma_short > sma_long, 0.0 otherwise
          - 'positions': difference of 'signal' to mark entry/exit
        Returns a DataFrame with these additional columns.
        """
        df = data.copy()
        df['sma_short'] = sma(df['close'], self.short_window)
        df['sma_long']  = sma(df['close'], self.long_window)

        # Generate raw signals
        df['signal'] = 0.0
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1.0

        # Generate trading orders: +1 for a buy, -1 for a sell
        df['positions'] = df['signal'].diff().fillna(0)

        return df
