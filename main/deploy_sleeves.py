# main/deploy_sleeves.py
"""
Live/paper deployment of the TWO-SLEEVE portfolio via Alpaca.

SAFE BY DEFAULT: running this prints the target portfolio and the orders it
WOULD submit (dry run). Add --execute to actually submit orders.

Usage (from the repo root, with .env containing API_KEY/API_SECRET/PAPER):

    python main/deploy_sleeves.py                      # dry run - always start here
    python main/deploy_sleeves.py --execute            # submit orders
    python main/deploy_sleeves.py --only momentum --allocation-momentum 1.0
                                                       # reproduces the old single-sleeve book

Architecture: one process, netted targets
-----------------------------------------
Both sleeves share ONE Alpaca account, so they cannot be deployed as two
independent jobs: build_orders liquidates any position it is shown that is not
in its target list, so each sleeve would sell the other's book every week. This
script instead reads the account once, computes each sleeve's weights against
its own allocated capital, nets them into a single account-level target vector,
and submits one set of orders - which also gives correct GLOBAL sells-before-buys.

The safety rule that makes netting sound: a sleeve that FAILS to compute has its
symbols removed from the managed set for that run. It is never represented as
"target 0", because that would read as "liquidate everything I hold" - one
yfinance hiccup would flatten the ETF book. See --only and the SKIPPED path.

Intended schedule: once a week after Friday's close (orders queue for Monday's
open) - see .github/workflows/deploying-weekly-momentum.yml.
"""
import argparse
import dataclasses
import os
import sys
from datetime import datetime, timedelta, timezone

import holidays
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.execution.orders import MIN_ORDER_NOTIONAL, build_orders
from src.strategies.sleeves import (DIVERSIFIER, MOMENTUM, SleeveConfig,
                                    managed_symbols, net_targets, validate_sleeves)
from src.strategies.weekly_momentum import (BufferedSelector, apply_no_trade_band,
                                            compute_target_weights,
                                            make_vol_target_exposure)
from main.backtest_weekly_momentum import fetch_close_matrix

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PAPER = os.getenv("PAPER", "True").strip().lower() in ("1", "true", "yes")

REPORT_PATH = "live_weekly_momentum.md"
MAX_STALE_DAYS = 5           # max consecutive missing prints before a name is dropped
MAX_DATA_AGE_DAYS = 4        # refuse to trade if the latest bar is older than this
HISTORY_DAYS = 600           # covers 252+21 momentum + the 200dma gate


@dataclasses.dataclass
class SleeveResult:
    """Outcome of one sleeve for one run. `status` gates everything downstream."""
    sleeve: SleeveConfig
    status: str                       # "ok" | "skipped"
    weights: pd.Series = None         # sleeve-relative weights (sum <= 1)
    managed: set = dataclasses.field(default_factory=set)
    reason: str = ""
    signal_date: object = None


def resolve_sleeves(only: str, alloc_momentum: float):
    """
    Apply CLI overrides to the sleeve registry, then validate.

    Returns (active, declared). `active` is the set of sleeves to RUN this
    invocation; `declared` is always the full registry and is what decides
    symbol OWNERSHIP.

    Those must not be the same tuple. Ownership is a property of the configured
    architecture, not of which sleeves happen to run today - if `--only momentum`
    narrowed the ownership map too, the residual momentum sleeve would stop
    seeing the ETFs as "claimed by the diversifier", absorb them as orphans, and
    liquidate the entire ETF book. Deriving `managed` from `declared` makes
    `--only` safe by construction.
    """
    mom, div = MOMENTUM, DIVERSIFIER
    if alloc_momentum is not None:
        mom = dataclasses.replace(mom, allocation=alloc_momentum)
        div = dataclasses.replace(div, allocation=round(1.0 - alloc_momentum, 10))
    declared = (mom, div)

    active = tuple(s for s in declared if s.allocation > 0)
    if only:
        active = tuple(s for s in active if s.name == only)
        if not active:
            raise ValueError(f"--only {only!r} selects no configured sleeve")

    # Validate the configured split, not the filtered subset.
    validate_sleeves(tuple(s for s in declared if s.allocation > 0))
    return active, declared


def compute_sleeve(sleeve, panel, spy, positions, all_sleeves, universe, equity,
                   liquidate_orphans=True) -> SleeveResult:
    """
    Target weights for one sleeve, relative to that sleeve's own capital.

    Any failure returns status="skipped" with an empty managed set, so the
    caller emits no orders at all for this sleeve's symbols.
    """
    try:
        managed = managed_symbols(sleeve, all_sleeves, universe, positions)
        if sleeve.residual and not liquidate_orphans:
            managed = set(universe)

        cols = [c for c in panel.columns if c in managed]
        if len(cols) < sleeve.top_k:
            raise ValueError(f"only {len(cols)} priced names, need {sleeve.top_k}")
        sub = panel[cols]

        selector = BufferedSelector(sleeve.buffer_mult)
        # Incumbents must be scoped to THIS sleeve, or the stock selector would be
        # seeded with the ETF book (and vice versa).
        selector.held = [s for s in positions if s in managed and s in sub.columns]

        exposure_fn = None
        if sleeve.target_vol and sleeve.target_vol > 0:
            exposure_fn = make_vol_target_exposure(
                target_vol=sleeve.target_vol, with_regime_gate=sleeve.use_dma_gate,
                low_exposure=sleeve.low_exposure)

        w = compute_target_weights(
            sub, spy, top_k=sleeve.top_k, weight_cap=sleeve.weight_cap,
            low_exposure=sleeve.low_exposure, selector=selector,
            exposure_fn=exposure_fn)
        if len(w) == 0:
            raise ValueError("no scoreable names (insufficient history?)")

        # No-trade band in SLEEVE weight space. run_walkforward applies it per
        # sleeve, so banding after netting would make the live band ~1.4x wider
        # for the 70% sleeve and silently diverge from the backtest.
        sleeve_equity = equity * sleeve.allocation
        current = pd.Series({s: v / sleeve_equity for s, v in positions.items()
                             if s in managed}, dtype=float)
        w = apply_no_trade_band(w, current, sleeve.min_trade_fraction)

        return SleeveResult(sleeve, "ok", w, managed,
                            signal_date=sub.index[-1].date())
    except Exception as e:
        return SleeveResult(sleeve, "skipped", None, set(), reason=str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="Actually submit orders.")
    p.add_argument("--only", choices=["momentum", "diversifier"], default=None,
                   help="Run one sleeve; the other's holdings are left untouched.")
    p.add_argument("--allocation-momentum", type=float, default=None,
                   help="Override the momentum allocation (diversifier gets the rest).")
    p.add_argument("--liquidate-orphans", dest="liquidate_orphans",
                   action="store_true", default=True,
                   help="Sell positions no sleeve's universe claims (default).")
    p.add_argument("--no-liquidate-orphans", dest="liquidate_orphans",
                   action="store_false",
                   help="Leave unclaimed positions alone.")
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500")
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    if now.date().weekday() >= 5 or now.date() in holidays.financial_holidays("NYSE"):
        print("Weekend/holiday: orders would just queue; exiting.")
        sys.exit(0)

    sleeves, declared = resolve_sleeves(args.only, args.allocation_momentum)
    print("Running: " + ", ".join(f"{s.name} {s.allocation:.0%}" for s in sleeves))
    idle = [s.name for s in declared if s not in sleeves and s.allocation > 0]
    if idle:
        print(f"Not running (holdings left untouched): {', '.join(idle)}")

    # 1) Universes and one shared price panel
    if args.universe == "sp500":
        from src.data.universe import fetch_sp500_symbols
        stock_universe = fetch_sp500_symbols()
    else:
        from src.data.universe import fetch_nasdaq_100_symbols
        stock_universe = fetch_nasdaq_100_symbols()

    universes = {}
    for s in sleeves:
        universes[s.name] = list(s.universe) if s.universe else stock_universe

    wanted = sorted(set().union(*universes.values()) | {"SPY"})
    start = now - timedelta(days=HISTORY_DAYS)
    prices = fetch_close_matrix(wanted, start, now)
    if "SPY" not in prices.columns:
        print("ABORT: no SPY data; cannot evaluate the regime gate.")
        sys.exit(1)

    # Bounded ffill only. An unbounded ffill lets a halted or delisted ticker
    # carry a flat price forward indefinitely, which keeps it selectable - and
    # buyable. Past a week of no prints, drop the name entirely.
    prices = prices.reindex(prices["SPY"].dropna().index).ffill(limit=MAX_STALE_DAYS)
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
    spy = prices["SPY"]

    # 2) Account state - read ONCE, before any weights are computed.
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

    # 3) Per-sleeve target weights
    results = []
    for s in sleeves:
        # `declared`, not `sleeves`: ownership comes from the configured
        # architecture, so --only cannot widen the residual sleeve's claim.
        r = compute_sleeve(s, prices, spy, positions, declared, universes[s.name],
                           equity, args.liquidate_orphans)
        results.append(r)
        if r.status == "ok":
            print(f"\n[{s.name}] alloc {s.allocation:.0%} "
                  f"(${equity * s.allocation:,.2f}) | signal {r.signal_date} | "
                  f"sleeve gross {r.weights.sum():.2f} -> "
                  f"account {r.weights.sum() * s.allocation:.3f}")
            print((r.weights * 100).round(2).to_string())
        else:
            print(f"\n[{s.name}] SKIPPED: {r.reason}  "
                  f"(its holdings will NOT be touched this run)")

    ok = [r for r in results if r.status == "ok"]
    if not ok:
        print("\nABORT: every sleeve failed; no orders.")
        sys.exit(1)

    # 4) Net into one account-level target book
    combined = net_targets({r.sleeve.name: r.weights for r in ok},
                           [r.sleeve for r in ok])
    managed_all = set().union(*[r.managed for r in ok])
    print(f"\nCombined account exposure {combined.sum():.2f} "
          f"({(1 - combined.sum()) * 100:.1f}% cash), "
          f"{len(managed_all)} managed symbols")

    # min_trade_fraction=0.0: the band was already applied per sleeve, in sleeve
    # weight space. Only dust filtering remains here.
    orders = build_orders(combined.to_dict(), positions, equity,
                          managed=managed_all, min_trade_fraction=0.0)

    # Orphans are positions no DECLARED universe claims, which the residual
    # sleeve absorbs and sells. That is intended for delistings and retired
    # strategies - but it is also what a typo in a sleeve's universe tuple looks
    # like, so never let it happen silently.
    claimed = set()
    for s in declared:
        if s.universe:
            claimed |= set(s.universe)
        else:
            claimed |= set(universes.get(s.name, []))
    orphans = {sym: positions[sym] for sym, side, _, _ in orders
               if side == "sell" and sym in positions and sym not in claimed}

    # 5) Report + submission. Keep the literal "weekly momentum rebalance" - the
    # workflow greps for it to build the email body.
    lines = [f"{now}: weekly momentum rebalance "
             f"({'EXECUTED' if args.execute else 'DRY RUN'}) - two-sleeve",
             f"Equity ${equity:,.2f}, combined exposure {combined.sum():.2f}", ""]
    for r in results:
        if r.status == "ok":
            lines.append(f"- {r.sleeve.name}: alloc {r.sleeve.allocation:.0%}, "
                         f"signal {r.signal_date}, sleeve gross {r.weights.sum():.2f}, "
                         f"account contribution {r.weights.sum() * r.sleeve.allocation:.3f}")
        else:
            lines.append(f"- {r.sleeve.name}: SKIPPED ({r.reason}) - holdings untouched")
    warnings = [f"{r.sleeve.name} skipped: {r.reason}"
                for r in results if r.status != "ok"]
    if orphans:
        total = sum(orphans.values())
        detail = ", ".join(f"{k} ${v:,.0f}" for k, v in sorted(orphans.items()))
        warnings.append(f"liquidating {len(orphans)} unclaimed position(s) "
                        f"worth ${total:,.0f} ({detail}) - these belong to no "
                        f"sleeve universe; check for a universe typo if unexpected")
        print(f"\nWARNING: liquidating {len(orphans)} unclaimed position(s) "
              f"worth ${total:,.0f}: {detail}")
    lines.append("")
    lines.append(f"WARNINGS: {'; '.join(warnings) if warnings else 'none'}")
    lines.append("")

    print()
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

    if not orders:
        print("(no orders - everything already within the no-trade band)")
        lines.append("- (no orders)")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to submit these orders.")
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
