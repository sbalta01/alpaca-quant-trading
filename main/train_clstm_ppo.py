import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import attach_factors
from src.strategies.clstm_ppo_ML import train_clstm_ppo
from src.utils.tools import compute_turbulence, rsi, adx
from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    # df is MultiIndex [symbol, timestamp]
    pieces = []
    for sym, group in df.groupby(level="symbol"):
        g = group.droplevel("symbol").copy()
        # MACD:
        ema12 = g["close"].ewm(span=12).mean()
        ema26 = g["close"].ewm(span=26).mean()
        g["macd"] = ema12 - ema26
        # RSI(14):
        g["rsi"]  = rsi(g["close"], 14)
        # CCI(20):
        tp = (g["high"] + g["low"] + g["close"]) / 3
        ma_tp = tp.rolling(20).mean()
        md    = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        g["cci"] = (tp - ma_tp) / (0.015 * md)
        # ADX(14) – you can implement it in src/utils/indicators.py
        g["adx"] = adx(g["high"], g["low"], g["close"], window=14)

        g["symbol"]    = sym
        g["timestamp"] = g.index
        pieces.append(g.reset_index(drop=True))

    out = pd.concat(pieces, ignore_index=True)
    out = out.set_index(["symbol","timestamp"])

    turbulence = compute_turbulence(out, window=252)

    mi = pd.MultiIndex.from_product(
        [out.index.get_level_values("symbol").unique(), turbulence.index],
        names=["symbol","timestamp"]
    )
    turb_rep = pd.Series(
        np.repeat(turbulence.values[None, :], len(mi.levels[0]), axis=0).ravel(),
        index=mi,
        name="turbulence"
    )

    full = out.join(turb_rep, how="left")
    return full.dropna()

if __name__ == "__main__":
    symbols = ["AAPL","MSFT","GOOG"]
    start   = datetime(2018,1,1)
    end     = datetime(2022,1,1)
    timeframe = TimeFrame.Day


    if timeframe.unit_value == TimeFrameUnit.Month:
        print('Timeframe set to Month')
        timeframe_yahoo = '1mo'
    elif timeframe.unit_value == TimeFrameUnit.Week:
        print('Timeframe set to Week')
        timeframe_yahoo = '1wk'
    elif timeframe.unit_value == TimeFrameUnit.Day:
        print('Timeframe set to Day')
        timeframe_yahoo = '1d'
    elif timeframe.unit_value == TimeFrameUnit.Hour:
        print('Timeframe set to Hour')
        timeframe_yahoo = '1h'
    elif timeframe.unit_value == TimeFrameUnit.Minute:
        print('Timeframe set to Minute')
        timeframe_yahoo = '1m'
    try:
        from src.data.data_loader import fetch_alpaca_data as fetch_data
        jkandc
        print('USING ALPACA DATA')
    except:
        from src.data.data_loader import fetch_yahoo_data as fetch_data
        print('USING YAHOO DATA')
        timeframe = timeframe_yahoo
        feed = None
    
    # fetch price + indicators (you’ll need to join tech_cols & macro_cols beforehand)
    df = fetch_data(symbols, start, end, timeframe=timeframe)
    df = attach_factors(df, timeframe=timeframe_yahoo)
    df = add_technicals(df)

    tech_cols  = ["macd","rsi","cci","adx"]
    macro_cols = ["VIX","EURUSD"]

    model = train_clstm_ppo(
        price_df = df,
        tech_cols=tech_cols,
        macro_cols=macro_cols,
        seq_len      = 30,
        lstm_hidden  = 512,
        total_timesteps=500_000,
        n_envs         = 1
    )

    model.save("clstm_ppo_stock")
