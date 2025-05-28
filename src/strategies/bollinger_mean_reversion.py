# src/strategies/bollinger_mean_reversion.py

import pandas as pd
from src.strategies.base_strategy import Strategy
from src.utils.indicators import sma

class BollingerMeanReversionStrategy(Strategy):
    """
    Bollinger Bands Mean Reversion:
      - Lower band = SMA(window) - k * rolling_std(window)
      - Upper band = SMA(window) + k * rolling_std(window)
    Signals:
      * Buy (signal=+1) when price < lower band
      * Sell (signal=-1) when price > upper band
      * Hold (signal=0) otherwise
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

        # Generate signals
        df["signal"] = 0.0
        df.loc[df["close"] < df["lower"], "signal"] = 1.0
        df.loc[df["close"] > df["upper"], "signal"] = -1.0
        
        return df