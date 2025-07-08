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

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) via Wilder's smoothing.

    Parameters
    ----------
    high, low, close : pd.Series
        series of highs, lows, closes
    window : int
        lookback period (typically 14)

    Returns
    -------
    pd.Series
        ADX values, aligned with `close` index.
    """
    # 1) True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 2) Directional Movement
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # 3) Wilder’s smoothing (EMA with alpha=1/window, adjust=False)
    atr       = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr
    minus_di  = 100 * minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr

    # 4) DX and ADX
    dx  = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=1/window, adjust=False).mean()

    return adx

def compute_turbulence(
    df: pd.DataFrame,
    window: int = 252
) -> pd.Series:
    """
    Compute a turbulence index as the Mahalanobis distance of each day's
    cross-section of returns from the rolling-window historical mean.

    Parameters
    ----------
    df : MultiIndex DataFrame [symbol,timestamp] with 'close'
    window : int
        Lookback window in trading days for mean/cov estimation.

    Returns
    -------
    pd.Series
        Indexed by timestamp, turbulence values.
    """
    # 1) Pivot to wide: rows=timestamp, cols=symbol
    prices = df["close"].unstack(level="symbol")
    # 2) Compute daily returns
    # rets = prices.pct_change().dropna(how="all")
    rets = np.log(df['close'] / df['close'].shift(1)).dropna()
    dates = rets.index

    tur_vals = []
    # Precompute nothing for the first `window` days
    for i, date in enumerate(dates):
        if i < window:
            tur_vals.append(0.0)
        else:
            window_rets = rets.iloc[i-window : i]
            mu  = window_rets.mean()
            cov = window_rets.cov().values
            x   = rets.iloc[i].values
            # Mahalanobis distance: (x-mu).T @ inv(cov) @ (x-mu)
            # use pseudoinverse for stability
            delta = x - mu.values
            inv_cov = np.linalg.pinv(cov)
            dist = float(delta @ inv_cov @ delta)
            tur_vals.append(dist)

    turbulence = pd.Series(tur_vals, index=dates, name="turbulence")
    return turbulence

def compute_turbulence_single_symbol(
    df: pd.DataFrame,
    window: int = 252
) -> pd.Series:
    """
    Compute a (1D) turbulence index for a single symbol as the squared
    deviation of each day's log-return from its rolling-window mean,
    normalized by the rolling-window variance.

    Parameters
    ----------
    df : pd.DataFrame
        Single-index DataFrame (index name 'timestamp') with a 'close' column.
    window : int
        Lookback window (in bars) for mean/variance estimation.

    Returns
    -------
    pd.Series
        Indexed by timestamp, turbulence values.
    """
    # 1) Compute log returns
    rets = np.log(df['close'] / df['close'].shift(1)).dropna()
    dates = rets.index

    tur_vals = []
    # 2) For each date, compute 1D Mahalanobis distance:
    #    (r - μ)^2 / σ^2  over the prior `window` returns.
    for i, date in enumerate(dates):
        if i < window:
            tur_vals.append(0.0)
        else:
            window_rets = rets.iloc[i - window : i]
            mu  = window_rets.mean()
            var = window_rets.var(ddof=1)
            delta = float(rets.iloc[i] - mu)
            # guard against zero variance
            if var > 0:
                dist = (delta * delta) / var
            else:
                dist = 0.0
            tur_vals.append(dist)

    return pd.Series(tur_vals, index=dates, name="turbulence")

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

@staticmethod
def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range over `window`."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def threshold_adjust(factor: pd.Series, horizon: int = 10, base_threshold: float = 0.5, max_shift: float = 0.2):
            short_vol = factor.rolling(horizon//2).std(ddof=1)
            long_vol = factor.rolling(horizon).std(ddof=1)
            ret_mean = factor.rolling(horizon).mean()
            z = (short_vol - ret_mean)/(long_vol + 1e-8)
            shift = max_shift * np.tanh(z)
            return (base_threshold + shift).fillna(base_threshold).clip(0.0,1.0)