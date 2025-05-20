import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta, timezone
import pandas as pd

# ── CONFIG ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path('.') / '.env')


API_KEY    = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

PAPER = True  # Set to False for live trading

# Clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

##Strategy: moving average crossover

def get_bars(symbol: str, days=1):
    start = datetime(2025, 5, 19, 13, 00, tzinfo=timezone.utc) #NYSE is open 13:00 to 20:00 UTC 
    end = start + timedelta(days=days)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute, #Compute bars every minute
        start=start,
        end=end,
        feed= "iex", #Free plan
    )
    bars = data_client.get_stock_bars(request_params).df
    return bars[bars.index.get_level_values(0) == symbol]

aapl_bars = get_bars("AAPL")

print(aapl_bars)