import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')

# Alpaca API setup
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Strategy parameters
NUM_TOP = 10
MOMENTUM_DAYS = 126  # ~6 months
MAX_EXPOSURE = 0.9

def get_universe_tickers():
    return pd.read_csv("sp500.csv")["Symbol"].tolist()

def fetch_price_data(tickers, start_date, end_date):
    # Batch fetch to handle large lists
    all_data = []
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        bars = api.get_bars(
            batch,
            TimeFrame.Day,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            feed='iex'
        ).df
        all_data.append(bars.reset_index())
    if not all_data:
        raise ValueError("No data fetched.")
    df = pd.concat(all_data)
    df = df.set_index(['timestamp', 'symbol'])
    prices = df['close'].unstack(level='symbol').sort_index()
    return prices

def backtest(prices, initial_capital=100_000):
    # Determine the first business day of each month
    monthly_dates = prices.resample('MS').first().index

    cash = initial_capital
    holdings = pd.Series(0, index=prices.columns)
    portfolio_value = []

    for date in prices.index:
        # Rebalance on first business day of month
        if date in monthly_dates:
            # Compute momentum
            hist = prices.loc[:date]
            mom = hist.iloc[-1] / hist.shift(MOMENTUM_DAYS).iloc[-1] - 1
            top = mom.nlargest(NUM_TOP).index.tolist()
            weight = MAX_EXPOSURE / NUM_TOP

            # Liquidate outside top
            to_sell = holdings[holdings > 0].index.difference(top)
            cash += (holdings[to_sell] * prices.loc[date, to_sell]).sum()
            holdings[to_sell] = 0

            # Calculate new target holdings
            alloc_cash = cash * MAX_EXPOSURE
            target_vals = pd.Series({sym: alloc_cash * weight for sym in top})
            target_qty = (target_vals / prices.loc[date, top]).astype(int)

            # Execute trades
            # Sell differences
            sell_qty = holdings[top] - target_qty
            sell_qty[sell_qty > 0] = sell_qty[sell_qty > 0]
            cash += (sell_qty * prices.loc[date, sell_qty.index]).sum()
            holdings[top] -= sell_qty.clip(lower=0)

            # Buy differences
            buy_qty = target_qty - holdings[top]
            buy_qty[buy_qty > 0] = buy_qty[buy_qty > 0]
            spend = (buy_qty * prices.loc[date, buy_qty.index]).sum()
            cash -= spend
            holdings[top] += buy_qty.clip(lower=0)

        # Record daily portfolio value
        daily_val = cash + (holdings * prices.loc[date]).sum()
        portfolio_value.append({'date': date, 'value': daily_val})

    return pd.DataFrame(portfolio_value).set_index('date')

# Define backtest period
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365 * 3)  # 3 years

# Run backtest
tickers = get_universe_tickers()
prices = fetch_price_data(tickers, start_date, end_date)
results = backtest(prices)

# Display results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(results.index, results['value'])
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()