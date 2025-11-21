# src/execution/live_tracker.py

import pandas as pd
from datetime import datetime, timezone

class LivePerformanceTracker:
    def __init__(self, initial_cash: float, initial_equity:float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.initial_equity = initial_equity
        self.equity_history = []
        self.trade_log = []

    def record_trade(self, timestamp, symbol, qty, side, price):
        """Update cash and log trade."""
        cost = qty * price
        if side.lower() == "buy":
            self.cash -= cost
        elif side.lower() == "sell":
            self.cash += cost

        self.trade_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "price": price
        })

    def update_equity(self, current_positions):
        """Estimate current equity based on positions + cash."""
        equity = self.cash
        for p in current_positions:
            equity += float(p.market_value)
        self.equity_history.append((datetime.now(timezone.utc), equity))
        return equity

    def print_status(self):
        if not self.equity_history:
            return
        times, equities = zip(*self.equity_history)
        latest_eq = equities[-1]
        ret = (latest_eq / self.initial_equity - 1) * 100
        max_eq = max(equities)
        # dd = (latest_eq - max_eq) / max_eq * 100

        print(f"\n [Live Performance]")
        print(f"  Equity      : ${latest_eq:.2f}")
        print(f"  Return      : {ret:.2f}%")
        # print(f"  Drawdown    : {dd:.2f}%")
        print(f"  Trades Made : {len(self.trade_log)}")
