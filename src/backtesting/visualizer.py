# src/backtesting/visualizer.py

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

def plot_returns(results: pd.DataFrame, results_control: pd.DataFrame, title: str = "returns Curve") -> None:
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

    for symbol, grp in results.groupby(level="symbol"):
        grp = grp.droplevel("symbol")
        ax.plot(grp.index, grp["cum_returns"], label=symbol)

    avg_return_control = results_control['cum_returns'].unstack(level='symbol').mean(axis=1) #Avg returns for all symbols with the control strategy
    ax.plot(results_control.index.get_level_values('timestamp').unique(), avg_return_control, label ='Returns control', color = 'k')

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Cum_returns")
    ax.set_title(title)
    ax.set_xlim(results_control.index.get_level_values('timestamp')[0], results_control.index.get_level_values('timestamp')[-1])
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_signals(
    results: pd.DataFrame,
    results_control: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    position_col: str = "position",
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
        ax.plot(df.index, df[price_col], label=label or "Price")
        try:
            ax.plot(df.index, df["ma_short"], '--', label="Short MA")
            ax.plot(df.index, df["ma_long"], '--', label="Long MA")
        except:
            pass
        long  = df[df[signal_col] ==  1.0]
        long  = long[long[position_col] ==  1.0]
        flat = df[df[position_col] ==  0.0]
        flat_1 = flat[flat[signal_col] == -1.0]
        flat_2 = flat[flat[signal_col] == 1.0]
        short = df[df[signal_col] == -1.0]
        short = short[short[position_col] ==  -1.0]
        ax.scatter(long.index,  long[price_col], marker="^", color = "olive",  s=50)
        ax.scatter(short.index, short[price_col], marker="v", color = "darkslategrey", s=50)
        ax.scatter(flat_1.index, flat_1[price_col], marker="o", color = "darkslategrey", s=50)
        ax.scatter(flat_2.index, flat_2[price_col], marker="o", color = "darkslategrey", s=50)

    for symbol, grp in results.groupby(level="symbol"):
        single = grp.droplevel("symbol")
        _plot_for_group(single, label=symbol)
    ax.legend()
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.set_xlim(results_control.index.get_level_values('timestamp')[0], results_control.index.get_level_values('timestamp')[-1])
    ax.set_ylim(results_control['close'].min()*0.75, results_control['close'].max()*1.25)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()