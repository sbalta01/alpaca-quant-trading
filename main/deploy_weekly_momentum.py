# main/deploy_weekly_momentum.py
"""
Live/paper deployment of the weekly cross-sectional momentum portfolio via Alpaca.

SAFE BY DEFAULT: running this prints the target portfolio and the orders it
WOULD submit (dry run). Add --execute to actually submit orders.

Usage (from the repo root, with .env containing API_KEY/API_SECRET/PAPER):

    python main/deploy_weekly_momentum.py               # dry run - always start here
    python main/deploy_weekly_momentum.py --execute     # submit orders (paper if PAPER=True)

Intended schedule: once a week after Friday's close (orders queue for Monday's
open) - see .github/workflows/deploying-weekly-momentum.yml.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import holidays
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.strategies.weekly_momentum import BufferedSelector, compute_target_weights
from main.backtest_weekly_momentum import fetch_close_matrix

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PAPER = os.getenv("PAPER", "True").strip().lower() in ("1", "true", "yes")

REPORT_PATH = "live_weekly_momentum.md"
MIN_ORDER_NOTIONAL = 1.0     # Alpaca minimum
MIN_TRADE_FRACTION = 0.005   # skip rebalance trades < 0.5% of equity (churn control)
MAX_STALE_DAYS = 5           # max consecutive missing prints before a name is dropped
MAX_DATA_AGE_DAYS = 4        # refuse to trade if the latest bar is older than this


def build_orders(targets: dict, current: dict, equity: float) -> list:
    """
    Orders as (symbol, side, notional_or_None, close_all: bool), sells first,
    largest deltas first. `targets` are weights (sum <= 1, remainder cash);
    positions held but not in targets are liquidated.
    """
    symbols = set(targets) | set(current)
    deltas = {}
    for s in symbols:
        target_notional = equity * targets.get(s, 0.0)
        deltas[s] = (target_notional, target_notional - current.get(s, 0.0))

    orders = []
    ordered = sorted(symbols, key=lambda s: (deltas[s][1] > 0, -abs(deltas[s][1])))
    for s in ordered:
        target_notional, delta = deltas[s]
        if abs(delta) < max(MIN_ORDER_NOTIONAL, MIN_TRADE_FRACTION * equity):
            continue
        if target_notional < MIN_ORDER_NOTIONAL and current.get(s, 0.0) > 0:
            orders.append((s, "sell", None, True))          # close position entirely
        elif delta < 0:
            orders.append((s, "sell", round(-delta, 2), False))
        else:
            orders.append((s, "buy", round(delta, 2), False))
    return orders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="Actually submit orders.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--weight-cap", type=float, default=0.20)
    p.add_argument("--low-exposure", type=float, default=0.4)
    p.add_argument("--buffer-mult", type=float, default=1.5,
                   help="Rank-buffer width: hold incumbents until they exit the "
                        "top buffer_mult*k. 1.0 = no buffering (old behavior). "
                        "Backtested 2016-2026: halves turnover at equal Sharpe.")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--holistic", action="store_true",
                   help="Use the full holistic method (layers 2-4: reversal + ML "
                        "ranker + HMM gate). Needs xgboost/hmmlearn and ~3y more "
                        "history; only switch after it beats the default method "
                        "out-of-sample (main/backtest_weekly_holistic.py).")
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    if now.date().weekday() >= 5 or now.date() in holidays.financial_holidays("NYSE"):
        print("Weekend/holiday: orders would just queue; exiting.")
        sys.exit(0)

    # 1) Universe and data
    if args.tickers:
        universe = args.tickers
    else:
        from src.data.data_loader import fetch_nasdaq_100_symbols
        universe = fetch_nasdaq_100_symbols()
    # 600d covers 252+21 momentum + 200dma; the holistic ML layer needs ~3.5y
    # of weekly cross-sections on top of the ~1y feature warm-up.
    start = now - timedelta(days=1700 if args.holistic else 600)
    prices = fetch_close_matrix(sorted(set(universe)), start, now)
    bench = fetch_close_matrix(["SPY"], start, now)
    # Bounded ffill only. An unbounded ffill lets a halted or delisted ticker
    # carry a flat price forward indefinitely, which keeps it selectable - and
    # buyable. Past a week of no prints, drop the name entirely.
    prices = prices.reindex(bench.index).ffill(limit=MAX_STALE_DAYS)
    dead = prices.columns[prices.iloc[-1].isna()]
    if len(dead) > 0:
        print(f"Dropping {len(dead)} ticker(s) with no recent price: {list(dead)}")
        prices = prices.drop(columns=dead)

    # Refuse to trade on stale data (e.g. a silent yfinance failure).
    last_bar = prices.index[-1]
    age_days = (now.replace(tzinfo=None) - last_bar.to_pydatetime()).days
    if age_days > MAX_DATA_AGE_DAYS:
        print(f"ABORT: latest bar {last_bar.date()} is {age_days}d old "
              f"(limit {MAX_DATA_AGE_DAYS}d). Refusing to trade on stale data.")
        sys.exit(1)

    # 2) Account state - fetched BEFORE the weights because rank buffering
    #    treats the account's current positions as the incumbents to hold.
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    account = client.get_account()
    equity = float(account.equity)
    positions = {pos.symbol: float(pos.qty) * float(pos.current_price)
                 for pos in client.get_all_positions()}
    print(f"Account ({'PAPER' if PAPER else 'LIVE'}): equity ${equity:,.2f}, "
          f"{len(positions)} open positions")

    # 3) Target weights as of the latest close
    if args.holistic:
        from src.strategies.weekly_holistic import compute_holistic_target_weights
        w = compute_holistic_target_weights(prices, bench["SPY"], top_k=args.top_k,
                                            weight_cap=args.weight_cap)
    else:
        # Rank buffering: hold an incumbent until it drops out of the top
        # buffer_mult*k, killing rank-10<->11 churn. --buffer-mult 1.0 disables.
        selector = BufferedSelector(args.buffer_mult)
        selector.held = [s for s in positions if s in prices.columns]
        w = compute_target_weights(prices, bench["SPY"], top_k=args.top_k,
                                   weight_cap=args.weight_cap,
                                   low_exposure=args.low_exposure,
                                   selector=selector)
    print(f"Signal date: {prices.index[-1].date()} | gross exposure {w.sum():.2f}")
    print((w * 100).round(2).to_string(), f"\nCash: {(1 - w.sum()) * 100:.2f}%\n")

    # 4) Orders (sells first)
    orders = build_orders(w.to_dict(), positions, equity)
    lines = [f"{now}: weekly momentum rebalance ({'EXECUTED' if args.execute else 'DRY RUN'})",
             f"Signal date {prices.index[-1].date()}, equity ${equity:,.2f}, "
             f"gross exposure {w.sum():.2f}", ""]
    for symbol, side, notional, close_all in orders:
        desc = f"{side.upper():4s} {'ALL' if close_all else f'${notional}'} {symbol}"
        if args.execute:
            try:
                if close_all:
                    qty = client.get_open_position(symbol).qty_available
                    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                             time_in_force=TimeInForce.DAY)
                else:
                    side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
                    req = MarketOrderRequest(symbol=symbol, notional=notional, side=side_enum,
                                             time_in_force=TimeInForce.DAY)
                client.submit_order(req)
                desc += "  [submitted]"
            except Exception as e:
                desc += f"  [ERROR: {e}]"
        print(desc)
        lines.append(f"- {desc}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to submit these orders.")
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
