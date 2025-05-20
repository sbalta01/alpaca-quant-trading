# src/data/data_loader.py

import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.config import API_KEY, API_SECRET

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
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day
) -> pd.DataFrame:
    """
    Fetch historical bars from Alpaca between start and end.
    Returns a DataFrame indexed by timestamp with columns open, high, low, close, volume.
    """
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end
    )
    bars = client.get_stock_bars(req).df
    # Alpaca's df has a MultiIndex (symbol, timestamp); we unpack it:
    df = bars.xs(symbol, level=0).sort_index()
    return df
