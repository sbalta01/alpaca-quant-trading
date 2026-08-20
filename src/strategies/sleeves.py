# src/strategies/sleeves.py
"""
Sleeve registry: the capital-split layer of the portfolio.

Why sleeves exist
-----------------
The momentum book is a concentrated single-sector bet by construction - the
cross-sectional winners cluster (10 semiconductor names at 0.54 average pairwise
correlation is a typical week). Full-covariance vol targeting then divides gross
exposure down hard, so the account sits mostly in cash while still holding an
undiversified book.

Widening top_k does not fix it (measured: Sharpe 1.01 -> 0.79 from k=10 to k=40 -
going deeper into the ranking just dilutes the signal), and neither does merging
diversifying ETFs INTO the stock ranking (measured in REVIEW_findings_and_roadmap.md:
across 522 weeks no ETF ever entered the top-10, because a 15%-vol instrument cannot
win a momentum contest against 500 single-stock fat tails). Diversification has to
happen at the CAPITAL-SPLIT level: run the momentum book on part of the account and
an uncorrelated sleeve on the rest.

Measured 9y walk-forward at 10bps, 70/30:
    momentum only  CAGR 20.1%  Sharpe 1.08  MaxDD -24.9%
    70/30 combined CAGR 16.0%  Sharpe 1.13  MaxDD -20.3%   (sleeve corr 0.25)
It buys risk-adjusted return and drawdown with CAGR - not more return.

This module is deliberately pure: config dataclasses and pure functions, no
alpaca/dotenv/network imports and no module-level I/O, so it stays importable
under the CI workflow's minimal dependency set (pandas/numpy only). In
particular do NOT import src.execution.live_executor here - it reads
trades_info.json at import time.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class SleeveConfig:
    """One capital sleeve: an allocation, a universe, and its risk settings."""
    name: str
    allocation: float
    universe: Optional[Tuple[str, ...]] = None   # None = dynamic (index membership)
    residual: bool = False                       # owns positions no sleeve claims
    top_k: int = 10
    weight_cap: float = 0.20
    target_vol: float = 0.20
    buffer_mult: float = 1.5
    use_dma_gate: bool = True
    low_exposure: float = 0.4
    min_trade_fraction: float = 0.005


# The live momentum book, unchanged in every respect except that it now sizes
# against 70% of equity instead of 100%.
MOMENTUM = SleeveConfig(
    name="momentum",
    allocation=0.70,
    universe=None,          # resolved from the S&P 500 at run time
    residual=True,          # also owns any position no other sleeve claims
)

# Pure diversifiers: long-duration and intermediate Treasuries, gold, broad
# commodities, REITs. SPY/QQQ are deliberately EXCLUDED - a "diversifier" sleeve
# that can go 100% long US equity will do exactly that in a momentum-friendly
# regime, stacking beta on the stock book precisely when it should be offsetting
# it. Measured correlation to the momentum sleeve: 0.26 without SPY/QQQ, 0.63 with.
#
# weight_cap 0.60 is load-bearing, not arbitrary - see validate_sleeves.
# No 200dma gate: when equities roll over the right response for THIS sleeve is
# rotating into bonds/gold, not going to cash on an equity signal.
DIVERSIFIER = SleeveConfig(
    name="diversifier",
    allocation=0.30,
    universe=("TLT", "IEF", "GLD", "DBC", "VNQ"),
    residual=False,
    top_k=2,
    weight_cap=0.60,
    target_vol=0.10,
    use_dma_gate=False,
    low_exposure=1.0,
)

SLEEVES: Tuple[SleeveConfig, ...] = (MOMENTUM, DIVERSIFIER)


def validate_sleeves(sleeves: Sequence[SleeveConfig] = SLEEVES) -> None:
    """
    Fail loudly at startup on any misconfiguration that would misprice orders.

    Raises ValueError. Called before the account is touched.
    """
    if not sleeves:
        raise ValueError("no sleeves configured")

    names = [s.name for s in sleeves]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate sleeve names: {names}")

    total = sum(s.allocation for s in sleeves)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"allocations must sum to 1.0, got {total!r} ({names})")
    for s in sleeves:
        if not (0.0 < s.allocation <= 1.0):
            raise ValueError(f"sleeve {s.name!r} allocation {s.allocation} not in (0, 1]")

    residuals = [s.name for s in sleeves if s.residual]
    if len(residuals) > 1:
        raise ValueError(f"at most one residual sleeve, got {residuals}")

    # Overlapping universes would have two sleeves independently vol-target and
    # size the same name from different `held` state. Reject rather than net it.
    declared = [(s.name, set(s.universe)) for s in sleeves if s.universe]
    for i, (n1, u1) in enumerate(declared):
        for n2, u2 in declared[i + 1:]:
            shared = u1 & u2
            if shared:
                raise ValueError(
                    f"sleeves {n1!r} and {n2!r} share symbols: {sorted(shared)}")

    for s in sleeves:
        # top_k * weight_cap < 1 silently produces a SHORT book. inverse_vol_weights
        # caps then redistributes; when every name is over cap there is nothing to
        # redistribute into, the loop breaks, and the weights sum to top_k*cap < 1.
        # realized_portfolio_vol then normalizes (w / w.sum()) before estimating, so
        # vol_target_exposure computes a multiplier for a NORMALIZED book and applies
        # it to the short one - the vol target undershoots and nothing warns you.
        # Measured on the real ETF panel, top-2: cap 0.35 -> gross 0.505 vs 0.839.
        if s.top_k * s.weight_cap < 1.0:
            raise ValueError(
                f"sleeve {s.name!r}: top_k * weight_cap = "
                f"{s.top_k} * {s.weight_cap} = {s.top_k * s.weight_cap:.2f} < 1.0, "
                f"which silently under-invests the book and breaks vol targeting. "
                f"Raise weight_cap to at least {1.0 / s.top_k:.2f}.")


def managed_symbols(
    sleeve: SleeveConfig,
    sleeves: Sequence[SleeveConfig],
    resolved_universe: Iterable[str],
    positions: Iterable[str],
) -> set:
    """
    The symbol set `sleeve` is allowed to trade or liquidate.

    That is its own resolved universe, plus - for the residual sleeve - any held
    position that no sleeve's declared universe claims. Orphans (delisted names,
    leftovers from a retired strategy) must belong to someone or the `managed`
    gate would strand them in the account forever.
    """
    owned = set(resolved_universe)
    if sleeve.residual:
        claimed = set()
        for other in sleeves:
            if other.universe:
                claimed |= set(other.universe)
        owned |= {s for s in positions if s not in claimed}
    return owned


def net_targets(
    per_sleeve: Dict[str, pd.Series],
    sleeves: Sequence[SleeveConfig] = SLEEVES,
) -> pd.Series:
    """
    Scale each sleeve's weights by its allocation and net into one
    account-level target vector.

    Sleeve weights are relative to the sleeve's own capital, so a 0.85 gross
    book in a 0.30 sleeve contributes 0.255 of the account. Only sleeves present
    in `per_sleeve` contribute: a sleeve that failed to compute must be ABSENT
    here (and excluded from the managed set), never present with zero weights -
    zeros would read as "liquidate everything I hold".
    """
    alloc = {s.name: s.allocation for s in sleeves}
    parts = []
    for name, w in per_sleeve.items():
        if name not in alloc:
            raise ValueError(f"unknown sleeve {name!r}")
        if w is not None and len(w) > 0:
            parts.append(w * alloc[name])
    if not parts:
        return pd.Series(dtype=float)
    combined = pd.concat(parts).groupby(level=0).sum()
    return combined.sort_values(ascending=False)
