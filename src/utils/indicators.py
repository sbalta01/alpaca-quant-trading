# src/utils/indicators.py

import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average
    """
    return series.rolling(window=window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average
    """
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average for RSI by convention
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
