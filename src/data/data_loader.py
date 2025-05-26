# src/data/data_loader.py

import pandas as pd
from datetime import datetime
from typing import List, Union

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
