# src/execution/live_executor.py

import time
from datetime import datetime, timedelta, timezone
from typing import Union, List

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.config import API_KEY, API_SECRET

# Paper vs Live
PAPER = True

# Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client    = StockHistoricalDataClient(API_KEY, API_SECRET)


def get_bars(
    symbols: Union[str, List[str]],
    minutes: int = 30,
    feed: str = "iex"
) -> pd.DataFrame:
    """
    Fetch minute-bars for one or more symbols over the past `minutes`.
    Returns a DataFrame with MultiIndex (symbol, timestamp).
    """
    if isinstance(symbols, str):
        symbols_list = [symbols]
    else:
        symbols_list = symbols

    end = datetime(2025,5,20,20,00,tzinfo=timezone.utc)
    # end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)

    req = StockBarsRequest(
        symbol_or_symbols=symbols_list,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
    )
    bars = data_client.get_stock_bars(req).df
    # bars has MultiIndex [symbol, timestamp]
    return bars.sort_index()


def _run_for_symbol(symbol: str):
    """
    Core strategy for one symbol.
    """
    # 1) Fetch bars
    bars = get_bars(symbol)

    # 2) If insufficient data, skip
    if bars.xs(symbol, level=0).shape[0] < 20 + 1:
        print(f"[{symbol}] Not enough data yet.")
        return

    # 3) Compute indicators
    df = bars.xs(symbol, level=0).copy()
    df["sma_short"] = df["close"].rolling(window=5).mean()
    df["sma_long"]  = df["close"].rolling(window=20).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    # 4) Check current Alpaca position
    try:
        open_positions = trading_client.get_open_positions()
    except Exception:
        open_positions = []

    holding = any(p.symbol == symbol for p in open_positions)

    # 5) Generate and act on signals
    # signal == +1 → buy, -1 → sell
    signal = 0
    if prev.sma_short < prev.sma_long and latest.sma_short > latest.sma_long and not holding:
        signal = 1
    elif prev.sma_short > prev.sma_long and latest.sma_short < latest.sma_long and holding:
        signal = -1

    if signal == 1:
        print(f"[{symbol}] Buy signal at {latest.name}, price {latest.close}")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        trading_client.submit_order(order)

    elif signal == -1:
        print(f"[{symbol}] Sell signal at {latest.name}, price {latest.close}")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        trading_client.submit_order(order)

    else:
        print(f"[{symbol}] No trade signal at {latest.name}.")


def run_live(
    symbols: Union[str, List[str]],
    interval: int = 60
):
    """
    Polls market every `interval` seconds and runs the MA crossover strategy 
    on each symbol (str or list of str).
    """
    if isinstance(symbols, str):
        symbol_list = [symbols]
    else:
        symbol_list = symbols

    while True:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nChecking market at {now} for {symbol_list}")
        for sym in symbol_list:
            _run_for_symbol(sym)
        time.sleep(interval)


if __name__ == "__main__":
    # Example usage: either a single symbol...
    # run_live("AAPL")

    # ...or multiple symbols:
    run_live(["AAPL", "MSFT", "GOOG"])
