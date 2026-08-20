# src/execution/orders.py
"""
Target-book -> order-list translation, shared by every live deployer.

Extracted from main/deploy_weekly_momentum.py so it can be unit-tested and so
multi-sleeve deployers reuse exactly the same ordering and dust rules as the
single-sleeve one.

The `managed` parameter is what makes running several sleeves against ONE
Alpaca account safe: `build_orders` liquidates any position it sees that is not
in `targets`, so a sleeve must only ever be shown the symbols it actually owns.
"""
from typing import Iterable, List, Optional, Tuple

MIN_ORDER_NOTIONAL = 1.0     # Alpaca minimum
MIN_TRADE_FRACTION = 0.005   # skip rebalance trades < 0.5% of equity (churn control)


def build_orders(
    targets: dict,
    current: dict,
    equity: float,
    managed: Optional[Iterable[str]] = None,
    min_order_notional: float = MIN_ORDER_NOTIONAL,
    min_trade_fraction: float = MIN_TRADE_FRACTION,
) -> List[Tuple[str, str, Optional[float], bool]]:
    """
    Orders as (symbol, side, notional_or_None, close_all: bool), sells first,
    largest deltas first. `targets` are weights (sum <= 1, remainder cash);
    positions held but not in targets are liquidated.

    `managed`, when given, restricts the symbol set: positions outside it are
    neither sized nor liquidated. This is how one sleeve is prevented from
    selling another sleeve's book when both share an account - and, critically,
    how a sleeve that FAILED to compute leaves its holdings untouched instead of
    having them read as "target 0" and liquidated. `managed=None` reproduces the
    original single-sleeve behavior exactly.
    """
    symbols = set(targets) | set(current)
    if managed is not None:
        managed = set(managed)
        stray = set(targets) - managed
        if stray:
            # A sleeve targeting outside its own universe is a config error, not
            # something to paper over by silently dropping the order.
            raise ValueError(f"targets outside managed set: {sorted(stray)}")
        symbols &= managed

    deltas = {}
    for s in symbols:
        target_notional = equity * targets.get(s, 0.0)
        deltas[s] = (target_notional, target_notional - current.get(s, 0.0))

    orders = []
    ordered = sorted(symbols, key=lambda s: (deltas[s][1] > 0, -abs(deltas[s][1])))
    for s in ordered:
        target_notional, delta = deltas[s]
        if abs(delta) < max(min_order_notional, min_trade_fraction * equity):
            continue
        if target_notional < min_order_notional and current.get(s, 0.0) > 0:
            orders.append((s, "sell", None, True))          # close position entirely
        elif delta < 0:
            orders.append((s, "sell", round(-delta, 2), False))
        else:
            orders.append((s, "buy", round(delta, 2), False))
    return orders
