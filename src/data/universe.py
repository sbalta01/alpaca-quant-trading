# src/data/universe.py
"""
Robust universe fetching for the live weekly pipeline.

The Friday GitHub-Actions run used to depend on a single un-retried Wikipedia
scrape: one 503, one rate-limit, or one table-schema change and the live
rebalance failed outright (or worse, a silently-wrong parse could return a
handful of tickers and the deploy script would cheerfully trade a 4-name
portfolio). This module wraps the scrape with:

  1. Retries with exponential backoff - but only for errors where retrying can
     help (connection errors, timeouts, 5xx, 429). A 404 or a parse failure
     means the page changed; retrying is pointless and we fall through.
  2. Validation before trusting the result: plausible count (90-110), plausible
     ticker syntax, and high overlap with the last known-good list. Rejecting a
     bad parse matters more than surviving an outage.
  3. A committed CSV snapshot as the fallback. It must be a *committed* file,
     not a disk cache: Actions runners are ephemeral (fetch-depth: 1), so any
     ~/.cache entry is cold on every scheduled run. The workflow commits the
     refreshed snapshot back after each successful run, so it is never more
     than a week stale.
"""
import os
import re
import time
from datetime import datetime, timezone
from io import StringIO
from typing import List, Tuple

import pandas as pd
import requests

NASDAQ_100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0"}
_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(_DIR, "nasdaq100_snapshot.csv")
SP500_SNAPSHOT_PATH = os.path.join(_DIR, "sp500_snapshot.csv")

TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")
MIN_COUNT, MAX_COUNT = 90, 110          # NASDAQ-100 plausibility range
SP500_MIN, SP500_MAX = 490, 510
MIN_OVERLAP = 0.85          # Jaccard similarity vs the snapshot


class UniverseError(RuntimeError):
    """Raised when neither the live fetch nor the snapshot can supply a universe."""


def fetch_with_retry(url: str, *, headers: dict = None, timeout: float = 15.0,
                     retries: int = 3, backoff: float = 2.0) -> str:
    """GET with retries on transient failures only. Returns response text."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers or HEADERS, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = requests.HTTPError(f"HTTP {resp.status_code}")
            else:
                resp.raise_for_status()   # 4xx (not 429): permanent, don't retry
                return resp.text
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
        if attempt < retries - 1:
            wait = backoff ** attempt
            print(f"  fetch attempt {attempt + 1}/{retries} failed ({last_err}); "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)
    raise last_err


def parse_universe(html: str, ticker_col: str) -> List[str]:
    """Extract tickers from the constituents table (the one with `ticker_col`)."""
    tables = pd.read_html(StringIO(html))
    df = next(tbl for tbl in tables if ticker_col in tbl.columns)
    return [str(s).replace(".", "-") for s in df[ticker_col].tolist()]


def parse_nasdaq_100(html: str) -> List[str]:
    return parse_universe(html, "Ticker")


def validate_universe(symbols: List[str], reference: List[str] = None,
                      min_count: int = MIN_COUNT, max_count: int = MAX_COUNT) -> None:
    """Raise ValueError if `symbols` does not look like a plausible index list."""
    if not (min_count <= len(symbols) <= max_count):
        raise ValueError(f"implausible universe size: {len(symbols)} "
                         f"(expected {min_count}-{max_count})")
    bad = [s for s in symbols if not TICKER_RE.match(s)]
    if bad:
        raise ValueError(f"implausible tickers: {bad[:5]}")
    if reference:
        a, b = set(symbols), set(reference)
        jaccard = len(a & b) / len(a | b)
        if jaccard < MIN_OVERLAP:
            raise ValueError(f"only {jaccard:.0%} overlap with last known-good "
                             f"list (need >= {MIN_OVERLAP:.0%}) - suspect bad parse")


def load_snapshot(path: str = None) -> Tuple[List[str], datetime]:
    """Last known-good universe and when it was recorded.

    `path` resolves at call time (not def time) so tests and callers can
    repoint SNAPSHOT_PATH."""
    df = pd.read_csv(path or SNAPSHOT_PATH)
    as_of = pd.to_datetime(df["as_of"].iloc[0]).to_pydatetime()
    return df["ticker"].tolist(), as_of


def save_snapshot(symbols: List[str], path: str = None) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pd.DataFrame({"ticker": symbols, "as_of": stamp}).to_csv(
        path or SNAPSHOT_PATH, index=False)


def _fetch_index_symbols(url: str, ticker_col: str, snapshot_path: str,
                         min_count: int, max_count: int,
                         use_snapshot_fallback: bool = True,
                         refresh_snapshot: bool = True) -> List[str]:
    """
    Shared fetch pipeline: retry -> parse -> validate -> refresh snapshot,
    falling back to the committed snapshot on any failure.

    Never returns an unvalidated list: a syntactically-alive but semantically
    wrong scrape falls back to the snapshot just like an outage does.
    """
    try:
        snapshot, snap_date = load_snapshot(snapshot_path)
    except (FileNotFoundError, KeyError, IndexError):
        snapshot, snap_date = None, None

    try:
        symbols = parse_universe(fetch_with_retry(url), ticker_col)
        validate_universe(symbols, reference=snapshot,
                          min_count=min_count, max_count=max_count)
        if refresh_snapshot:
            try:
                save_snapshot(symbols, snapshot_path)
            except OSError as e:                     # read-only FS is non-fatal
                print(f"  (could not refresh snapshot: {e})")
        return symbols
    except Exception as e:
        if not (use_snapshot_fallback and snapshot):
            raise UniverseError(
                f"universe fetch failed ({e}) and no usable snapshot exists") from e
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - snap_date).days
        print(f"WARNING: live universe fetch failed ({e}); using committed "
              f"snapshot of {len(snapshot)} tickers from {snap_date.date()} "
              f"({age}d old)")
        return snapshot


def fetch_nasdaq_100_symbols(use_snapshot_fallback: bool = True,
                             refresh_snapshot: bool = True,
                             url: str = NASDAQ_100_URL) -> List[str]:
    """Current NASDAQ-100 membership: retry, validation, snapshot fallback."""
    return _fetch_index_symbols(url, "Ticker", SNAPSHOT_PATH, MIN_COUNT, MAX_COUNT,
                                use_snapshot_fallback, refresh_snapshot)


def fetch_sp500_symbols(use_snapshot_fallback: bool = True,
                        refresh_snapshot: bool = True,
                        url: str = SP500_URL) -> List[str]:
    """Current S&P 500 membership: retry, validation, snapshot fallback."""
    return _fetch_index_symbols(url, "Symbol", SP500_SNAPSHOT_PATH,
                                SP500_MIN, SP500_MAX,
                                use_snapshot_fallback, refresh_snapshot)
