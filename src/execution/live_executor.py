# src/execution/live_executor.py

import time
from datetime import datetime, timedelta, timezone
from typing import Union, List

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PAPER = os.getenv("PAPER")

from src.data.data_loader import fetch_alpaca_data
from src.strategies.base_strategy import Strategy
from src.execution.live_tracker import LivePerformanceTracker

# Alpaca trading client
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
account = trading_client.get_account()

initial_cash = float(account.cash)
initial_equity = float(account.equity)  # This includes cash + market value of positions

tracker = LivePerformanceTracker(initial_cash, initial_equity)


def run_live_strategy(
    strategy: Strategy,
    symbols: Union[str, List[str]],
    timeframe: TimeFrame,
    lookback_minutes: int = 30, #data retrieval
    interval_seconds: int = 60, #update time,
    cash_per_trade: float = 1000,
    feed: str = "iex"
):
    """
    Poll market every `interval_seconds`:
      1) Fetch last `lookback_minutes` of timeframe-bars
      2) Call strategy.generate_signals for each symbol
      3) Submit orders based on the latest signal (+1 buy, -1 sell)
    
    Parameters
    ----------
    strategy : Strategy
        Any subclass implementing `generate_signals(df) -> df` with 'signal' col.
    symbols : str or list of str
        Ticker(s) to trade.
    lookback_minutes : int
    interval_seconds : int
    feed : str
    """

    def trade(sig, symbol):
        open_pos = trading_client.get_all_positions()
        holding = any(p.symbol == symbol for p in open_pos)
        qty = 0
        if holding:
            position = trading_client.get_open_position(symbol)
            qty = position.qty
            qty_available = position.qty_available
        # 5) Act on signal
        if sig == 1 and not holding: #I can't buy if I already own stock
            print(f"[{symbol}] BUY at {latest.name} price={latest.close:.2f}")
            order = MarketOrderRequest(
                symbol=symbol,
                # qty=1,
                notional = cash_per_trade,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY #Operate during open hours or else cancel. GTC if I want good till canceled.
            )
            trading_client.submit_order(order)
            tracker.record_trade(latest.name, symbol, 1, "buy", latest.close)

        elif sig == -1 and holding:
            print(f"[{symbol}] SELL at {latest.name} price={latest.close:.2f}")
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty_available, #Sell all shares owned
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            tracker.record_trade(latest.name, symbol, 1, "sell", latest.close)

        else:
            print(f"[{symbol}] No trade signal (signal={sig}, holding={holding}, qty={qty}).")
    running = True
    while running:
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(minutes=lookback_minutes)
        print(f"\n[{now_utc.strftime('%Y-%m-%d %H:%M:%S')}] Fetching bars for {symbols}")

        # end = datetime(2025,5,20,20,00,tzinfo=timezone.utc)
        # start = end - timedelta(minutes=lookback_minutes)

        # 1) Fetch timeframe bars via your data loader
        bars = fetch_alpaca_data(
            symbol=symbols,
            start=start,
            end=now_utc,
            timeframe=timeframe
        )
        if strategy.multi_symbol:
            df = strategy.generate_signals(bars.copy())
            for symbol, subdf in df.groupby(level="symbol"):
                subdf = subdf.droplevel("symbol")
                latest = subdf.iloc[-1]
                sig = latest.signal  # +1 buy, -1 sell, 0 hold
                trade(sig, symbol)
        else:
            for symbol in symbols:
                subdf = bars.xs(symbol, level="symbol")
                subdf = strategy.generate_signals(subdf.copy())
                latest = subdf.iloc[-1]
                sig = latest.signal  # +1 buy, -1 sell, 0 hold
                trade(sig, symbol)

        open_positions = trading_client.get_all_positions()
        tracker.update_equity(open_positions)
        tracker.print_status()

        if interval_seconds is None:
            running = False
        else:
            time.sleep(interval_seconds)