import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api.rest import REST, TimeFrame

# ── CONFIG ─────────────────────────────────────────────────────────
API_KEY    = os.getenv("PK6B60HN8XYN6BCSAZUD")
API_SECRET = os.getenv("cMlZ5giflRey7Tw9x0VTAJ5XeRLFsqnueubu4B1o")
BASE_URL   = "https://paper-api.alpaca.markets"

UNIVERSE   = "SPY"  # or load a list of S&P500 tickers
NUM_TOP    = 10
MOMENTUM_DAYS = 126  # ~6 months
MAX_EXPOSURE = 0.9
STOP_LOSS_PCT = 0.15

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_universe_tickers():
    # For demo: use SPY holdings via Alpaca
    sp = api.get_asset("SPY")
    # Alternatively, load static CSV of S&P500 tickers
    return [t.symbol for t in api.list_assets(asset_class="us_equity") 
            if t.symbol in pd.read_csv("sp500.csv")["Symbol"].tolist()]

def fetch_price_data(tickers):
    end_dt = datetime.now(pytz.UTC)
    start_dt = end_dt - timedelta(days=MOMENTUM_DAYS + 30)
    bars = api.get_bars(tickers, TimeFrame.Day, start=start_dt.isoformat(), end=end_dt.isoformat()).df
    # bars is a MultiIndex DataFrame: (timestamp, symbol)
    return bars['close'].unstack(level=1)

def compute_momentum(df):
    return df.iloc[-1] / df.shift(MOMENTUM_DAYS).iloc[-1] - 1

def generate_targets(momentum):
    top = momentum.nlargest(NUM_TOP).index.tolist()
    weight = MAX_EXPOSURE / NUM_TOP
    return {sym: weight for sym in top}

def risk_trim(symbol):
    pos = api.get_position(symbol)
    entry = float(pos.avg_entry_price)
    current = float(pos.market_value) / float(pos.qty)
    if (entry - current) / entry > STOP_LOSS_PCT:
        # Trim 50%
        qty = int(float(pos.qty) * 0.5)
        if qty > 0:
            api.submit_order(symbol=symbol, qty=qty, side='sell',
                             type='market', time_in_force='day')

def rebalance(targets):
    # 1) Liquidate unwanted
    current = {p.symbol: float(p.qty) for p in api.list_positions()}
    for sym in current:
        if sym not in targets:
            api.submit_order(symbol=sym, qty=current[sym], side='sell',
                             type='market', time_in_force='day')
    # 2) Send target orders
    account = api.get_account()
    cash = float(account.cash)
    for sym, w in targets.items():
        alloc = cash * w
        bar = api.get_barset(sym, TimeFrame.Day, limit=1).df[sym].iloc[-1]
        qty = int(alloc / bar.close)
        if qty > 0:
            api.submit_order(symbol=sym, qty=qty, side='buy',
                             type='market', time_in_force='day')

def main():
    tickers = get_universe_tickers()
    prices  = fetch_price_data(tickers)
    mom     = compute_momentum(prices)
    targets = generate_targets(mom)
    # Optional: apply stop-loss trims before full rebalance
    for s in [p.symbol for p in api.list_positions()]:
        risk_trim(s)
    rebalance(targets)
    print(f"[{datetime.now()}] Rebalanced into: {list(targets.keys())}")

if __name__ == "__main__":
    main()
