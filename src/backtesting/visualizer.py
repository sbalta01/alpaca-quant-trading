# src/backtesting/visualizer.py

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def plot_equity(results, title: str = "Equity Curve"):
    """
    Plot the evolution of account equity over time.
    
    Parameters
    ----------
    results : pd.DataFrame
        Must contain a datetime index and an 'equity' column.
    title : str
        Chart title.
    """
    fig, ax = plt.subplots()
    ax.plot(results.index, results['equity'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_signals(results, price_col='close', signal_col='signal', title: str = "Price & Signals"):
    """
    Plot price series with markers where positions change (entries/exits).
    
    Parameters
    ----------
    results : pd.DataFrame
        Must contain datetime index, price_col, and signal_col (±1 for buy/sell).
    price_col : str
        Column name for price.
    signal_col : str
        Column name where +1 = buy, -1 = sell.
    title : str
        Chart title.
    """
    fig, ax = plt.subplots()
    ax.plot(results.index, results[price_col], label="Price")
    
    buys  = results[results[signal_col] ==  1.0]
    sells = results[results[signal_col] == -1.0]
    ax.scatter(buys.index,  buys[price_col], marker="^", label="Buy",  s=50)
    ax.scatter(sells.index, sells[price_col], marker="v", label="Sell", s=50)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
