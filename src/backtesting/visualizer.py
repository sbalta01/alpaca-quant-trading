# src/backtesting/visualizer.py

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

from matplotlib import animation
import mplfinance as mpf

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
    # plt.show()


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
    # ax.set_ylim(results_control['close'].min()*0.75, results_control['close'].max()*1.25)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    # plt.show()

def plot_candles_with_macd(df, animate = False):
    """
    Given a MultiIndex DataFrame 'df' with levels ['symbol','timestamp']
    and columns ['open','high','low','close','volume'], create for each symbol:
      - Candlestick chart
      - Volume subplot
      - MACD subplot
    """

    symbols = df.index.get_level_values('symbol').unique()
    for sym in symbols:
        # Slice out symbol data and reframe index
        df_sym = df.xs(sym, level='symbol')

        # Calculate MACD lines
        #  - 12‐period EMA, 26‐period EMA, their difference is MACD
        #  - 9‐period EMA of MACD is the Signal line
        ema_short = df_sym['close'].ewm(span=12, adjust=False).mean()
        ema_long  = df_sym['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal    = macd_line.ewm(span=9, adjust=False).mean()
        hist      = macd_line - signal


        apds = [
        mpf.make_addplot(hist,type='bar',width=0.7,panel=1,
                         color='dimgray',alpha=1,secondary_y=False),
        mpf.make_addplot(macd_line,panel=1,color='fuchsia',secondary_y=True),
        mpf.make_addplot(signal,panel=1,color='b',secondary_y=True, linestyle='--'),
       ]
        s = mpf.make_mpf_style(base_mpf_style='yahoo',rc={'figure.facecolor':'white'})

        fig, axes = mpf.plot(df_sym,type='candle',addplot=apds,figscale=1.5,figratio=(7,5),title=f"{sym} — Candles + MACD",
                            style=s,volume=True,volume_panel=2,panel_ratios=(6,3,2),returnfig=True)

        ax_main = axes[0]
        ax_emav = ax_main
        ax_hisg = axes[2]
        ax_macd = axes[3]
        ax_sign = ax_macd
        ax_volu = axes[4]

        if animate:
            def animate(ival):
                if (20+ival) > len(df):
                    ani.event_source.interval *= 3
                    if ani.event_source.interval > 12000:
                        exit()
                    return
                data = df_sym.iloc[0:(30+ival)]
                exp12     = data['close'].ewm(span=12, adjust=False).mean()
                exp26     = data['close'].ewm(span=26, adjust=False).mean()
                macd      = exp12 - exp26
                signal    = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
                apds = [mpf.make_addplot(exp12,color='lime',ax=ax_emav),
                        mpf.make_addplot(exp26,color='c',ax=ax_emav),
                        mpf.make_addplot(histogram,type='bar',width=0.7,
                                        color='dimgray',alpha=1,ax=ax_hisg),
                        mpf.make_addplot(macd,color='fuchsia',ax=ax_macd),
                        mpf.make_addplot(signal,color='b',ax=ax_sign, linestyle = '--'),
                    ]

                for ax in axes:
                    ax.clear()
                mpf.plot(data,type='candle',addplot=apds,ax=ax_main,volume=ax_volu)

            ani = animation.FuncAnimation(fig,animate,interval=100)
        # plt.show()