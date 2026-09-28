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

Intended schedule: Saturday morning using the latest completed trading session
(normally Friday). Orders queue for the next market open - see
.github/workflows/deploying-weekly-momentum.yml.
"""
import argparse
import dataclasses
import os
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import holidays
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.execution.orders import MIN_ORDER_NOTIONAL, build_orders
from src.strategies.sleeves import (DIVERSIFIER, MOMENTUM, SleeveConfig,
                                    managed_symbols, net_targets, validate_sleeves)
from src.strategies.weekly_momentum import (BufferedSelector, apply_no_trade_band,
                                            compute_target_weights,
                                            make_vol_target_exposure,
                                            realized_portfolio_vol, regime_exposure)
from main.backtest_weekly_momentum import fetch_close_matrix

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PAPER = os.getenv("PAPER", "True").strip().lower() in ("1", "true", "yes")

REPORT_PATH = "live_weekly_momentum.md"
LATEST_REPORT_PATH = "latest_weekly_momentum.md"
MARKET_TZ = ZoneInfo("America/New_York")
MAX_STALE_DAYS = 5           # max consecutive missing prints before a name is dropped
MAX_DATA_AGE_DAYS = 4        # refuse to trade if the latest bar is older than this
HISTORY_DAYS = 600           # covers 252+21 momentum + the 200dma gate
PRICE_DOWNLOAD_ATTEMPTS = 3
PRICE_RETRY_SECONDS = 30


def latest_completed_session(now):
    """Use completed daily bars, including Friday for the Saturday job."""
    local = now.astimezone(MARKET_TZ)
    day = local.date() - timedelta(days=int(local.hour < 16))
    closed = holidays.financial_holidays("NYSE")
    while day.weekday() >= 5 or day in closed:
        day -= timedelta(days=1)
    return day


def write_report(lines, history_lines=None):
    text = "\n".join(lines) + "\n\n"
    with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(history_lines) + "\n\n" if history_lines is not None else text)


def fetch_completed_prices(wanted, start, signal_date):
    """Retry incomplete downloads before touching the account; never use stale SPY."""
    end = signal_date + timedelta(days=1)  # yfinance end is exclusive
    problem = ""
    for attempt in range(1, PRICE_DOWNLOAD_ATTEMPTS + 1):
        try:
            prices = fetch_close_matrix(wanted, start, end)
            # Do not let an unexpected later row become the signal session.
            if not prices.empty:
                prices = prices.loc[prices.index < pd.Timestamp(end)]
            spy = prices["SPY"].dropna() if "SPY" in prices else pd.Series(dtype=float)
            latest = spy.index[-1].date() if not spy.empty else None
            print(f"Price download {attempt}/{PRICE_DOWNLOAD_ATTEMPTS}: "
                  f"expected SPY session {signal_date}, latest received {latest}, "
                  f"{len(prices)} rows, {len(prices.columns)} symbols")
            if latest == signal_date:
                # Check the real SPY print BEFORE forward filling other symbols.
                return prices.reindex(spy.index).ffill(limit=MAX_STALE_DAYS)
            problem = (f"Missing completed SPY bar for {signal_date}; "
                       f"latest received: {latest}")
        except Exception as exc:
            problem = f"Price download failed: {type(exc).__name__}: {exc}"
            print(problem)
        if attempt < PRICE_DOWNLOAD_ATTEMPTS:
            delay = PRICE_RETRY_SECONDS * attempt
            print(f"Retrying the price panel in {delay}s; no orders submitted.")
            time.sleep(delay)
    raise RuntimeError(f"{problem}; exhausted {PRICE_DOWNLOAD_ATTEMPTS} attempts. "
                       "Refusing stale signals; no orders submitted.")


@dataclasses.dataclass
class SleeveResult:
    """Outcome of one sleeve for one run. `status` gates everything downstream."""
    sleeve: SleeveConfig
    status: str                       # "ok" | "skipped"
    weights: pd.Series = None         # sleeve-relative weights (sum <= 1)
    managed: set = dataclasses.field(default_factory=set)
    reason: str = ""
    signal_date: object = None
    risk: dict = dataclasses.field(default_factory=dict)


def portfolio_report(now, signal_date, equity, positions, combined, results, spy, execute):
    """Describe decision inputs and intended exposures, never imply filled orders."""
    managed = set().union(*(r.managed for r in results if r.status == "ok"))
    untouched = sum(value for symbol, value in positions.items() if symbol not in managed) / equity
    current = sum(positions.values()) / equity
    target = float(combined.sum()) + untouched
    lines = [f"Weekly rebalance | {now.astimezone(MARKET_TZ):%Y-%m-%d} | prices {signal_date}",
             f"Equity: ${equity:,.2f}",
             f"Invested exposure: {current:.1%} -> {target:.1%} target "
             f"({(target - current) * 100:+.1f} percentage points)",
             f"Implied cash at target: {1 - target:.1%}"]
    if untouched:
        lines.append(f"Includes {untouched:.1%} of equity in holdings left untouched.")
    px = spy.dropna()
    if len(px) >= 200:
        ma = float(px.iloc[-200:].mean())
        close = float(px.iloc[-1])
        lines += ["", f"Market trend: SPY ${close:,.2f}, "
                  f"{(close / ma - 1) * 100:+.1f}% vs 200-day average (${ma:,.2f}); "
                  f"{'above/at' if close >= ma else 'below'} trend threshold."]
    else:
        lines += ["", "Market trend: insufficient SPY history for the 200-day average."]
    lines += ["", "Sleeve exposures (% of account equity):"]
    for r in results:
        if r.status != "ok":
            lines.append(f"- {r.sleeve.name}: SKIPPED - {r.reason}; holdings untouched")
            continue
        before = sum(v for s, v in positions.items() if s in r.managed) / equity
        after = float(r.weights.sum()) * r.sleeve.allocation
        lines.append(f"- {r.sleeve.name}: {before:.1%} -> {after:.1%} target "
                     f"({(after - before) * 100:+.1f} percentage points)")
        risk = r.risk
        if "vol21" in risk:
            fast, slow = risk['vol21'], risk['vol63']
            fmt = lambda value: f"{value:.1%}" if math.isfinite(value) else "unavailable"
            estimate = max(fast, slow)  # same selection rule as vol_target_exposure
            if math.isfinite(estimate) and estimate > 0:
                window = '21-day' if fast >= slow else '63-day'
                state = 'above target; reduced exposure' if estimate > r.sleeve.target_vol else 'within target; no volatility reduction'
                detail = f"using {window} estimate, {state}"
            else:
                detail = "estimate unavailable/zero; strategy uses full exposure before trend gate"
            lines.append(f"  Basket volatility at full investment (annualized): 21d {fmt(fast)}, 63d {fmt(slow)} "
                         f"vs {r.sleeve.target_vol:.1%} target; {detail}.")
        if risk:
            gate = risk['gate']
            gate_text = (f"trend gate {'reducing exposure' if gate < 1 else 'not reducing exposure'} "
                         f"(x{gate:.2f})") if risk['gate_enabled'] else 'trend gate not applied'
            lines.append(f"  {gate_text}; model sleeve exposure {risk['model_exposure']:.1%}, "
                         f"after trade band {r.weights.sum():.1%}.")
    lines += ["", "Planned trades (dry run):" if not execute else "Order submissions (fills not yet confirmed):"]
    return lines


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

        # Capture inputs at sizing time, before the no-trade band changes weights.
        risk = {}
        sizing_fn = exposure_fn
        def measured_exposure(prices, benchmark, weights):
            exposure = (sizing_fn(prices, benchmark, weights) if sizing_fn is not None
                        else regime_exposure(benchmark, low_exposure=sleeve.low_exposure))
            gate_enabled = sleeve.use_dma_gate if sizing_fn is not None else True
            risk.update(gate_enabled=gate_enabled,
                        gate=regime_exposure(benchmark, low_exposure=sleeve.low_exposure) if gate_enabled else 1.0,
                        model_exposure=float(weights.sum() * exposure))
            if sizing_fn is not None:
                risk.update(vol21=realized_portfolio_vol(prices, weights, 21),
                            vol63=realized_portfolio_vol(prices, weights, 63))
            return exposure

        w = compute_target_weights(
            sub, spy, top_k=sleeve.top_k, weight_cap=sleeve.weight_cap,
            low_exposure=sleeve.low_exposure, selector=selector,
            exposure_fn=measured_exposure)
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
                            signal_date=sub.index[-1].date(), risk=risk)
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
    # Clear the current report before any operation can fail.
    with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"{now}: FAILED: rebalance did not complete. See this run's logs.\n")
    market_date = now.astimezone(MARKET_TZ).date()
    # Saturday is the weekly deployment day. Friday holidays roll the signal
    # back to the previous session via latest_completed_session below.
    if market_date.weekday() == 6 or (
            market_date.weekday() < 5 and market_date in holidays.financial_holidays("NYSE")):
        write_report([f"{now}: weekly momentum rebalance (SKIPPED)",
                      "Sunday/weekday market holiday in New York; no orders submitted."])
        sys.exit(1 if os.getenv("GITHUB_EVENT_NAME") == "schedule" else 0)

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
    signal_date = latest_completed_session(now)
    try:
        prices = fetch_completed_prices(wanted, start, signal_date)
    except RuntimeError as exc:
        write_report([f"{now}: weekly momentum rebalance (FAILED)", str(exc)])
        raise

    # Bounded ffill only. An unbounded ffill lets a halted or delisted ticker
    # carry a flat price forward indefinitely, which keeps it selectable - and
    # buyable. Past a week of no prints, drop the name entirely.
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

    # 5) Report + submission. Each invocation writes its own email body.
    lines = portfolio_report(now, signal_date, equity, positions, combined, results, spy, args.execute)
    warnings = []  # Skipped sleeves already carry their reason in the summary.
    if orphans:
        total = sum(orphans.values())
        detail = ", ".join(f"{k} ${v:,.0f}" for k, v in sorted(orphans.items()))
        warnings.append(f"liquidating {len(orphans)} unclaimed position(s) "
                        f"worth ${total:,.0f} ({detail}) - these belong to no "
                        f"sleeve universe; check for a universe typo if unexpected")
        print(f"\nWARNING: liquidating {len(orphans)} unclaimed position(s) "
              f"worth ${total:,.0f}: {detail}")
    if warnings:
        lines += ["", "WARNINGS: " + '; '.join(warnings), ""]
    history_lines = list(lines)

    print()
    submission_failed = False
    for symbol, side, notional, close_all in orders:
        desc = f"{side.upper():4s} {'ALL' if close_all else f'${notional}'} {symbol}"
        email_desc = desc
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
                order = client.submit_order(req)
                status = order.status.value
                desc += (f"  [id={order.id}; status={status}; "
                         f"filled_qty={order.filled_qty}; avg_fill_price={order.filled_avg_price}]")
                submission_failed = status in {"rejected", "canceled", "expired", "suspended"}
                if submission_failed:
                    email_desc += f" - FAILED: {status}"
            except Exception as e:
                desc += f"  [ERROR: {e}]"
                email_desc += f" - ERROR: {e}"
                submission_failed = True
        print(desc)
        lines.append(f"- {email_desc}")
        history_lines.append(f"- {desc}")
        if submission_failed:
            lines.append("FAILED: remaining orders stopped. Check broker orders before retrying.")
            history_lines.append(lines[-1])
            break

    if not orders:
        print("(no orders - everything already within the no-trade band)")
        lines.append("- No orders needed after trade-band and minimum-size filters.")
        history_lines.append(lines[-1])

    if not args.execute:
        print("\nDry run only. Re-run with --execute to submit these orders.")
    if args.execute:
        history_lines.append("Statuses above are at submission time; queued orders are not confirmed fills.")
    write_report(lines, history_lines=history_lines)
    if args.execute and (submission_failed or len(ok) != len(results)):
        sys.exit(1)


if __name__ == "__main__":
    main()
