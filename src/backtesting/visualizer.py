# src/backtesting/visualizer.py

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

def plot_returns(results: pd.DataFrame, title: str = "returns Curve") -> None:
    """
    Plot the evolution of account returns over time.
    Supports single-symbol (DatetimeIndex) or multi-symbol (MultiIndex with level 'symbol').
    
    Parameters
    ----------
    results : pd.DataFrame
        Must contain an 'returns' column. Index is either:
          - DatetimeIndex for one symbol
          - MultiIndex ['symbol', 'timestamp'] for multiple symbols
    title : str
        Chart title.
    """
    fig, ax = plt.subplots()
    
    if isinstance(results.index, pd.MultiIndex):
        # multiple symbols: plot each on the same axes
        for symbol, grp in results.groupby(level="symbol"):
            grp = grp.droplevel("symbol")
            ax.plot(grp.index, grp["returns"], label=symbol)
        ax.legend()
    else:
        # single symbol: just plot
        ax.plot(results.index, results["returns"])
    
    ax.set_xlabel("Date")
    ax.set_ylabel("returns")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_signals(
    results: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    title: str = "Price & Signals"
) -> None:
    """
    Plot price series with buy/sell markers.
    Supports single or multi-symbol results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Must contain price_col and signal_col columns. Index is either:
          - DatetimeIndex
          - MultiIndex ['symbol', 'timestamp']
    price_col : str
        Column name for price.
    signal_col : str
        Column name where +1 = buy, -1 = sell.
    title : str
        Chart title.
    """
    fig, ax = plt.subplots()
    
    def _plot_for_group(df: pd.DataFrame, label: str = None):
        ax.plot(df.index, df[price_col], label=label or "Price", color = 'k')
        try:
            ax.plot(df.index, df["ma_short"], '--', label="Short MA")
            ax.plot(df.index, df["ma_long"], '--', label="Long MA")
        except:
            pass
        buys  = df[df[signal_col] ==  1.0]
        sells = df[df[signal_col] == -1.0]
        ax.scatter(buys.index,  buys[price_col], marker="^", color = "olive", label=f"{label} Buy" if label else "Buy",  s=50)
        ax.scatter(sells.index, sells[price_col], marker="v", color = "darkslategrey",label=f"{label} Sell" if label else "Sell", s=50)
    
    if isinstance(results.index, pd.MultiIndex):
        # Loop per symbol
        for symbol, grp in results.groupby(level="symbol"):
            single = grp.droplevel("symbol")
            _plot_for_group(single, label=symbol)
        ax.legend()
    else:
        # Single symbol
        _plot_for_group(results)
        ax.legend()
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()