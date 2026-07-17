# src/strategies/bollinger_mean_reversion.py

import pandas as pd
from src.strategies.base_strategy import Strategy
from src.utils.tools import sma

class BollingerMeanReversionStrategy(Strategy):
    """
    Bollinger Bands Mean Reversion:
      - Lower band = SMA(window) - k * rolling_std(window)
      - Upper band = SMA(window) + k * rolling_std(window)
    Long-only, stateful (engine-compatible):
      - Enter long (position=1) when price closes below the lower band
      - Exit to flat (position=0) when price reverts to the SMA or above
    Emits:
      - 'position': 1.0 long / 0.0 flat
      - 'signal'  : diff of position (+1 buy, -1 sell), as BacktestEngine expects
    """
    name = "BollingerMeanReversion"

    def __init__(self, window: int = 20, k: float = 2.0):
        self.window = window
        self.k = k

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Compute rolling stats
        df["sma"] = sma(df["close"], self.window)
        df["std"] = df["close"].rolling(window=self.window).std()

        # Bollinger bands
        df["upper"] = df["sma"] + self.k * df["std"]
        df["lower"] = df["sma"] - self.k * df["std"]

        # Stateful long-only position
        position = 0
        positions = []
        for close, lower, mid in zip(df["close"], df["lower"], df["sma"]):
            if position == 0 and close < lower:
                position = 1
            elif position == 1 and close >= mid:
                position = 0
            positions.append(position)

        df["position"] = positions
        df["signal"] = df["position"].diff().fillna(0.0).clip(-1, 1)
        return df