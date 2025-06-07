# src/utils/indicators.py

import numpy as np
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

def remove_outliers(df: pd.DataFrame, ratio_outliers: float = np.inf) -> pd.DataFrame:
    """Drop rows where any feature is outside ratio*IQR."""
    if ratio_outliers == np.inf:
        return df
    else:
        clean = df.copy()
        for col in df.columns:
            q1 = clean[col].quantile(0.25)
            q3 = clean[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - ratio_outliers * iqr, q3 + ratio_outliers * iqr
            clean = clean[(clean[col] >= lo) & (clean[col] <= hi)]
        return clean
