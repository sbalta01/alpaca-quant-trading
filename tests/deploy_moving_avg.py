import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta, timezone
import pandas as pd

from src.config import APCA_API_KEY_ID as API_KEY
from src.config import APCA_API_SECRET_KEY as API_SECRET

    
PAPER = True  # Set to False for live trading

# Clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

##Strategy: moving average crossover

def get_bars(symbol: str, minutes=30):
    # Get minute-long bars from 'minutes' ago to compute the moving avg.
    # 30 minutes suffices because all we want is moving avg 5 to 20.
    end = datetime.now()
    # end = datetime(2025,5,20,20,00,tzinfo=timezone.utc)
    start = end - timedelta(minutes=minutes)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed= "iex",
    )
    bars = data_client.get_stock_bars(request_params).df
    return bars[bars.index.get_level_values(0) == symbol]


def moving_average_strategy(symbol="AAPL"):
    bars = get_bars(symbol)
    if bars.shape[0] < 21: #because our strategy is moving avg crossover 5 to 20
        print("Not enough data yet.")
        return

    bars["SMA5"] = bars["close"].rolling(window=5).mean()
    bars["SMA20"] = bars["close"].rolling(window=20).mean()

    last = bars.iloc[-1]
    prev = bars.iloc[-2]

    try:
        position = trading_client.get_open_position(symbol) #It returns whatever stock (or asset) I currently own
    except:
        position = []
    currently_holding = any(p.symbol == symbol for p in position)


    if prev.SMA5 < prev.SMA20 and last.SMA5 > last.SMA20 and not currently_holding:
        print("Buy signal")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC #When the order is going to be cancelled. 'GTC' = Good Till Canceled
        )
        trading_client.submit_order(order)
    elif prev.SMA5 > prev.SMA20 and last.SMA5 < last.SMA20 and currently_holding:
        print("Sell signal")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order)
    else:
        print("No trade signal.")

import time

if __name__ == "__main__":
    while True:
        print(f"Checking market at {datetime.now()}")
        moving_average_strategy("AAPL")
        time.sleep(60)

