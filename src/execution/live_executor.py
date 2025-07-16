# src/execution/live_executor.py

import time
from datetime import datetime, timedelta, timezone
from typing import Union, List
import holidays


from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from dotenv import load_dotenv
import os
import json
import sys

# Define the path to your JSON file
json_file_path = 'trades_info.json'
with open(json_file_path, 'r') as file:
    trades_info = json.load(file)

md_report_file_path = "live_trading_report.md"
# open(md_report_file_path, "w", encoding="utf-8").close() #Clear file

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
    feed: str = "iex",
    market = "NYSE",
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

    def trade(position, symbol, days_left):
        open_pos = trading_client.get_all_positions()
        holding = any(p.symbol == symbol for p in open_pos) #Whether I own 'symbol'
        # elapsed_open_days = np.inf #Initialize how long have I had this position open to infinity (to trade no matter what)
        if holding:  #If I hold symbol, get my open position
            position = trading_client.get_open_position(symbol)
            qty = position.qty
            qty_available = position.qty_available

            # orders = trading_client.get_orders(filter=GetOrdersRequest(
            #                                                             symbols=[symbol],
            #                                                             side=OrderSide.BUY,
            #                                                             status=QueryOrderStatus.CLOSED,
            #                                                             limit=50,
            #                                                         ))
            # if orders: #if I hold, check when was the last buy
            #     last = orders[0]
            #     print(f"Last order for {last.symbol}:")
            #     print(f"  · Side       : {last.side}")
            #     print(f"  · Status     : {last.status}")
            #     print(f"  · Filled     : {last.filled_at}")   # may be None if not filled
            #     if last.filled_at is not None: #If the last buy was filled, save its date
            #         last_trade_date = last.filled_at.date()
            #         today = datetime.now().date()

            #         # 3) Count business days (Mon–Fri)
            #         trading_days = pd.bdate_range(start=last_trade_date, end=today)

            #         # 4) Elapsed open‑market days **excluding** the day of last_buy itself
            #         elapsed_open_days = len(trading_days) - 1

            # else:
            #     print(f"No orders found for {symbol}.")
        else:
            qty = 0

        # Act on signal
        if position == 1 and not holding:
            report = f"[{symbol}] BUY at {latest.name} price={latest.close:.2f}"
            print(report)
            order = MarketOrderRequest(
                symbol=symbol,
                # qty=1,
                notional = cash_per_trade,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY #Operate during open hours or else cancel. GTC if I want good till canceled. Fractional orders only in DAY.
            )
            trading_client.submit_order(order)
            tracker.record_trade(latest.name, symbol, 1, "buy", latest.close)
            days_left = horizon - 1

        elif position == 0 and holding:
            report = f"[{symbol}] SELL at {latest.name} price={latest.close:.2f}"
            print(report)
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty_available, #Sell all shares owned
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            tracker.record_trade(latest.name, symbol, 1, "sell", latest.close)
        else:
            report = f"[{symbol}] No trade signal (Position={position}, Holding={holding}, Quantity={qty})."
            print(report)
        return days_left, report
    
    running = True
    while running:
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(minutes=lookback_minutes)
        print(f"\n[{now_utc.strftime('%Y-%m-%d %H:%M:%S')}] Fetching bars for {symbols}")

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

        # 1) Fetch timeframe bars via your data loader
        try:
            from src.data.data_loader import fetch_alpaca_data as fetch_data
            lmkdscsa
            bars = fetch_data(
                symbol=symbols,
                start=start,
                end=now_utc,
                timeframe=timeframe,
                feed = feed
            )
            print('USING ALPACA DATA')
        except:
            from src.data.data_loader import fetch_yahoo_data as fetch_data
            timeframe = timeframe_yahoo
            feed = None
            bars = fetch_data(
                symbol=symbols,
                start=start,
                end=now_utc,
                timeframe=timeframe,
                feed = feed
            )
            print('USING YAHOO DATA')

        market_hols = holidays.financial_holidays(market)
        today = now_utc.date()
        if today.weekday() >= 5:
            print("Today is weekend; exiting.")
            sys.exit(0)
        if today in market_hols:
            print(f"Today is a holiday; exiting.")
            sys.exit(0)
        print("Business day; continuing with the rest of the workflow.")

        try: #Fetch strategy's horizon if it has one to hold position until horizon days have passed
            horizon = strategy.horizon
        except:
            horizon = 0
        
        if strategy.multi_symbol:
            # df = strategy.generate_signals(bars.copy())
            # for symbol, subdf in df.groupby(level="symbol"):
            #     subdf = subdf.droplevel("symbol")
            #     latest = subdf.iloc[-1]
            #     position = latest.position  # 1 own, 0 flat
            #     trade(position, symbol)
            pass #Not done yet
        else:
            for symbol in symbols:
                days_left_key = f"days_left_{symbol}"
                try:
                    days_left = trades_info[days_left_key] #Get how many days left for next trade
                except:
                    days_left = 0 #No days left for next trade

                subdf = bars.xs(symbol, level="symbol")
                strategy.fit_and_save(subdf, f"models/{strategy.name}_{symbol}.pkl")
                strategy.load(f"models/{strategy.name}_{symbol}.pkl")
                position, timestamp = strategy.predict_next(subdf)
                latest = subdf.iloc[-1]

                if days_left > 0: #If days left then no trade
                    days_left -= 1
                    if position == 1: #If strategy predicts holding position, extend the number of days holding by one
                        days_left += 1
                    report = f"[{symbol}] Holding position. Days left = {days_left}."
                    print(report)
                else:
                    if market == 'NYSE':
                        days_left, report = trade(position, symbol, days_left)
                    elif market == 'XECB':
                        try:
                            holding = trades_info[f"position_{symbol}"]
                        except:
                            holding = 0
                        if position == 1 and holding == 0:
                            report = f"[{symbol}] MANUALLY BUY at {latest.name}"
                            print(report)
                            days_left = horizon - 1
                            trades_info[f"position_{symbol}"] = int(position)
                        elif position == 0 and holding == 1:
                            report = f"[{symbol}] MANUALLY SELL at {latest.name}"
                            print(report)
                            trades_info[f"position_{symbol}"] = int(position)
                        else:
                            report = f"[{symbol}] No trade signal (Position={position}, Holding={holding})."
                            print(report)

                # Save the updated JSON back to the file
                trades_info[days_left_key] = days_left
                with open(json_file_path, 'w') as file:
                    json.dump(trades_info, file, indent=4)

                with open(md_report_file_path, "a", encoding="utf-8") as md_file:
                    md_file.write(f"- {report}\n")

        if market == 'NYSE':
            open_positions = trading_client.get_all_positions()
            tracker.update_equity(open_positions)
            tracker.print_status()

        if interval_seconds is None:
            running = False
        else:
            time.sleep(interval_seconds)