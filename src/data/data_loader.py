# src/data/data_loader.py

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Union
import yfinance as yf
from pandas_datareader import data as pdr


from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PAPER = os.getenv("PAPER")

def fetch_sp500_symbols():
    url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    df = pd.read_csv(url)
    df[['Symbol']].to_csv('sp500.csv', index=False)
    print("sp500.csv created with", len(df), "symbols.")

def fetch_nasdaq_100_symbols():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    
    # First table on the page is the component list
    df = tables[4]  # As of now, table 4 contains the companies
    symbols = df['Ticker'].tolist()
    
    # Some tickers have "." instead of "-", fix for Yahoo Finance
    symbols = [s.replace('.', '-') for s in symbols]
    return symbols

def load_csv_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df.sort_index()

def fetch_alpaca_data(
    symbol: Union[str, List[str]],
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
    feed: str = 'iex'
) -> pd.DataFrame:
    """
    Fetch historical bars from Alpaca between start and end.
    
    Parameters
    ----------
    symbol : str or list of str
        One symbol (e.g. "AAPL") or multiple (["AAPL","MSFT"])
    start : datetime
    end   : datetime
    timeframe : TimeFrame
    
    Returns
    -------
    pd.DataFrame
      If `symbol` is a str: index is DatetimeIndex, columns [open, high, low, close, volume].
      If `symbol` is a list: MultiIndex [symbol, timestamp], columns as above.
    """
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed = feed #Defaults IEX (free plan)
    )
    bars = client.get_stock_bars(req).df

    # If a single symbol, extract that one branch of the MultiIndex:
    if isinstance(symbol, str):
        df = bars.xs(symbol, level=0).sort_index()
        df.index.name = "timestamp"
        return df

    # Otherwise, return the full MultiIndex DataFrame
    # (symbol, timestamp) → [open, high, low, close, volume]
    df = bars.sort_index()
    df.index.set_names(["symbol", "timestamp"], inplace=True)
    return df

def fetch_yahoo_data(
    symbol: Union[str, List[str]],
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
    feed: str = None
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance for one or more tickers (including non-US,
    e.g. German: 'BMW.DE', 'SAP.DE'). Returns:
      - If `symbol` is a string: DataFrame [timestamp] * [open, high, low, close, volume]
      - If `symbol` is a list: MultiIndex DataFrame [symbol, timestamp] * [open, high, low, close, volume]

    Parameters
    ----------
    symbol : str or list of str
        One or more Yahoo Finance tickers, e.g. "BMW.DE" or ["BMW.DE","SAP.DE"].
    start   : datetime
        Inclusive start date for fetch.
    end     : datetime
        Inclusive end date for fetch.
    timeframe: str
        Data timeframe; e.g., "1d", "1h", "5m", etc., as per yfinance API.
    """
    # Use yfinance.download which handles both single and list tickers
    if isinstance(symbol, str):
        tickers = symbol
    else:
        tickers = " ".join(symbol)

    # yfinance returns a DataFrame with columns like ('Adj Close','close'), etc.
    df = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=timeframe,
        group_by="ticker",  # if multiple, group columns under each ticker
        auto_adjust=False,  # do not auto‐adjust; keep raw OHLCV
        threads=True,
        progress=False
    )

    # If single symbol: df.columns = ['Open','High','Low','Close','Adj Close','Volume']
    if isinstance(symbol, str):
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
        )[
            ["open", "high", "low", "close", "volume"]
        ].dropna()
        df.index.name = "timestamp"
        return df

    # If multiple symbol: df is a DataFrame with columns MultiIndex (symbol, field)
    # e.g., df['BMW.DE']['Open'], df['BMW.DE']['Close'], etc.
    # We’ll stack it into a MultiIndex as [symbol, timestamp] → [open, high, low, close, volume]
    data_frames = []
    for sym in symbol:
        if (sym, "Close") not in df.columns:
            # No data for this symbol—skip
            continue
        tmp = df[sym].rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
        )[
            ["open", "high", "low", "close", "volume"]
        ].dropna()
        tmp.index.name = "timestamp"
        # Attach symbol as an outer index level
        tmp["symbol"] = sym
        tmp = tmp.reset_index().set_index(["symbol", "timestamp"])
        data_frames.append(tmp)

    if not data_frames:
        raise ValueError(f"No data fetched for any of {symbol}")

    out = pd.concat(data_frames).sort_index()
    return out

def fetch_fundamentals(symbols: List[str]) -> pd.DataFrame:
    data = {}
    for sym in symbols:
        tkr = yf.Ticker(sym)
        info = tkr.info
        data[sym] = {
            'PE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'PB': info.get('priceToBook'),
            'EPS': info.get('trailingEps'),
            'EBITDA': info.get('ebitda'),
            'earningsGrowth': info.get('earningsQuarterlyGrowth')
        }
    df = pd.DataFrame.from_dict(data, orient='index')
    df_clean = df.dropna(axis=1, how='any') #If there is no data for a particular variable (eg, an ETF), just drop the column
    return df_clean

FAMA_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
def fetch_macro_series(
    start: datetime, end: datetime, timeframe: str = "1d"
) -> pd.DataFrame:
    """
    Fetch daily macro series:
      - EUR/USD exchange: 'EURUSD=X'
      - VIX index: '^VIX'
      - US Fed funds rate: FRED 'DFF'
      - Fama-French daily factors
    Returns DataFrame indexed by date with columns
      ['EURUSD','VIX','DFF','MKT','SMB','HML'].
    """
    # 1) FX & VIX
    fx = yf.download(
        "EURUSD=X", start=start, end=end, interval=timeframe, auto_adjust=False, progress=False
        ).rename(columns={'Close':'EURUSD'})[['EURUSD']].dropna()
    fx.columns = fx.columns.droplevel(1)
    fx.index.name = "timestamp"
    vix = yf.download(
        "^VIX",    start=start, end=end, interval=timeframe, auto_adjust=False, progress=False
        ).rename(columns={'Close':'VIX'})[['VIX']].dropna()
    vix.index.name = "timestamp"
    vix.columns = vix.columns.droplevel(1)

    # 2) Fed funds rate
    dff = pdr.DataReader("DFF", "fred", start, end)
    dff.index.name = "timestamp"

    # 3) Fama-French
    ff = pd.read_csv(FAMA_FRENCH_URL, skiprows=3, index_col=0, skipfooter=1, engine = 'python')
    ff.index = pd.to_datetime(ff.index.astype(str), format='%Y%m%d')
    ff = ff.loc[start:end, ['Mkt-RF','SMB','HML','RMW','CMA','RF']].rename(columns={'Mkt-RF':'MKT'})
    ff.index.name = "timestamp"

    # 4) Merge all
    df = pd.concat(
        [fx, vix, dff, ff],
        axis=1,
        join='outer'   # union of all dates
    ).sort_index().ffill().bfill() #Same value for all times
    return df


def attach_factors(
    price_df: pd.DataFrame, timeframe: str = "1d"
) -> pd.DataFrame:
    """
    Given `price_df` with MultiIndex (symbol,timestamp) and price columns,
    fetch fundamentals & macro series, then return price_df augmented
    with columns ['PE','PB','EPS','EBITDA','earningsGrowth','EURUSD','VIX','DFF','MKT','SMB','HML'].
    """
    # 1) Extract list of symbols and overall date range
    dates   = price_df.index.get_level_values('timestamp').unique()
    symbols   = price_df.index.get_level_values('symbol').unique()
    start, end = dates.min(), dates.max()
    # 2) Fetch
    funds = fetch_fundamentals(symbols)       # indexed by symbol
    macro = fetch_macro_series(start, end, timeframe=timeframe)    # indexed by date

    # 3) Broadcast fundamentals to each symbol-date
    #    Create a DataFrame with index=(symbol,timestamp) and the funds columns
    idx = price_df.index
    # funds_panel = pd.DataFrame(index=idx)
    # for col in funds.columns:
    #     # map each symbol to its fundamental
    #     funds_panel[col] = idx.get_level_values('symbol').map(funds[col])

    # 4) Broadcast macro to each (symbol,timestamp)
    macro_panel = pd.DataFrame(index=idx)
    # normalize timestamps to date
    ts_dates = pd.to_datetime(idx.get_level_values('timestamp').normalize())
    for col in macro.columns:
        macro_panel[col] = macro[col].reindex(ts_dates).values

    # 5) Concatenate to original
    # augmented = pd.concat([price_df, funds_panel, macro_panel], axis=1).ffill() #Fill last row with previous value
    augmented = pd.concat([price_df, macro_panel], axis=1).ffill() #Fill last row with previous value
    return augmented
