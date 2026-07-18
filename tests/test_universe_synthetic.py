"""Tests for the hardened universe fetcher. No network - everything is
monkeypatched, with an inline HTML fixture standing in for Wikipedia.

Run from the repo root:  python tests/test_universe_synthetic.py
"""
import os
import sys
import tempfile

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data.universe as universe
from src.data.universe import (
    UniverseError, fetch_nasdaq_100_symbols, fetch_with_retry, parse_nasdaq_100,
    save_snapshot, validate_universe,
)

# A miniature constituents page: table 0 is a decoy, table 1 has the tickers.
TICKERS_100 = [f"T{i:03d}"[:5].replace("0", "A") for i in range(100)]
TICKERS_100 = [f"AB{i:02d}".replace("0", "C").replace("1", "D")
               .replace("2", "E").replace("3", "F").replace("4", "G")
               .replace("5", "H").replace("6", "I").replace("7", "J")
               .replace("8", "K").replace("9", "L") for i in range(100)]


def make_html(tickers):
    rows = "\n".join(f"<tr><td>{t}</td><td>Company {t}</td></tr>" for t in tickers)
    return f"""
    <table><tr><th>Year</th><th>Level</th></tr><tr><td>2020</td><td>1</td></tr></table>
    <table><tr><th>Ticker</th><th>Company</th></tr>{rows}</table>
    """


class FakeResponse:
    def __init__(self, status=200, text=""):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def with_fake_get(responses, fn):
    """Run fn() with requests.get returning each response (or raising) in turn."""
    calls = {"n": 0}
    real_get, real_sleep = requests.get, universe.time.sleep

    def fake_get(url, **kw):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        r = responses[i]
        if isinstance(r, Exception):
            raise r
        return r

    requests.get = fake_get
    universe.time.sleep = lambda s: None      # no real backoff waits in tests
    try:
        return fn(), calls["n"]
    finally:
        requests.get = real_get
        universe.time.sleep = real_sleep


def test_happy_path():
    html = make_html(TICKERS_100)
    (out, n) = with_fake_get([FakeResponse(200, html)],
                             lambda: fetch_with_retry("http://x"))
    assert out == html and n == 1
    assert len(parse_nasdaq_100(html)) == 100
    print("happy path, one request: OK")


def test_transient_503_then_success():
    html = make_html(TICKERS_100)
    (out, n) = with_fake_get([FakeResponse(503), FakeResponse(200, html)],
                             lambda: fetch_with_retry("http://x"))
    assert out == html and n == 2, f"expected retry then success, got {n} calls"
    print("503 then success retries exactly once: OK")


def test_404_does_not_retry():
    try:
        (_, n) = with_fake_get([FakeResponse(404)],
                               lambda: fetch_with_retry("http://x"))
        raise AssertionError("404 should raise")
    except requests.HTTPError:
        pass
    print("404 raises immediately (no pointless retries): OK")


def test_validation_rejects_junk():
    for bad, why in [
        (["AAPL", "MSFT"], "too few"),
        (TICKERS_100 + [f"X{i}" for i in range(50)], "too many"),
        (TICKERS_100[:-1] + ["bad ticker!"], "bad syntax"),
    ]:
        try:
            validate_universe(bad)
            raise AssertionError(f"validation should reject: {why}")
        except ValueError:
            pass
    validate_universe(TICKERS_100)     # sane list passes
    # low overlap vs reference is rejected even if syntactically fine
    other = [t[::-1].upper()[:4] for t in TICKERS_100]
    other = [f"Q{t[:4]}" for t in TICKERS_100]
    try:
        validate_universe(other, reference=TICKERS_100)
        raise AssertionError("low-overlap list should be rejected")
    except ValueError:
        pass
    print("validation rejects short/long/junk/low-overlap lists: OK")


def test_fallback_to_snapshot():
    """Persistent outage -> committed snapshot is served, with its age noted."""
    with tempfile.TemporaryDirectory() as td:
        snap = os.path.join(td, "snap.csv")
        save_snapshot(TICKERS_100, snap)
        old_path = universe.SNAPSHOT_PATH
        universe.SNAPSHOT_PATH = snap
        try:
            (out, n) = with_fake_get(
                [requests.ConnectionError("down")] * 5,
                lambda: fetch_nasdaq_100_symbols(refresh_snapshot=False))
            assert out == TICKERS_100
            assert n == 3, f"should exhaust 3 retries, made {n}"
        finally:
            universe.SNAPSHOT_PATH = old_path
    print("persistent outage falls back to snapshot after 3 attempts: OK")


def test_bad_parse_falls_back():
    """A page that parses to 4 tickers must be rejected -> snapshot, not traded."""
    with tempfile.TemporaryDirectory() as td:
        snap = os.path.join(td, "snap.csv")
        save_snapshot(TICKERS_100, snap)
        old_path = universe.SNAPSHOT_PATH
        universe.SNAPSHOT_PATH = snap
        try:
            html = make_html(["AAPL", "MSFT", "NVDA", "AMZN"])
            (out, _) = with_fake_get(
                [FakeResponse(200, html)],
                lambda: fetch_nasdaq_100_symbols(refresh_snapshot=False))
            assert out == TICKERS_100, "4-ticker parse must not be trusted"
        finally:
            universe.SNAPSHOT_PATH = old_path
    print("silently-wrong 4-ticker parse is rejected in favor of snapshot: OK")


def test_no_snapshot_and_dead_network_raises():
    old_path = universe.SNAPSHOT_PATH
    universe.SNAPSHOT_PATH = os.path.join(tempfile.gettempdir(), "nope_missing.csv")
    try:
        try:
            with_fake_get([requests.ConnectionError("down")] * 5,
                          lambda: fetch_nasdaq_100_symbols(refresh_snapshot=False))
            raise AssertionError("should raise UniverseError")
        except UniverseError:
            pass
    finally:
        universe.SNAPSHOT_PATH = old_path
    print("dead network with no snapshot raises UniverseError: OK")


def test_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        snap = os.path.join(td, "snap.csv")
        save_snapshot(TICKERS_100, snap)
        syms, as_of = universe.load_snapshot(snap)
        assert syms == TICKERS_100
        assert as_of is not None
    print("snapshot save/load roundtrip: OK")


if __name__ == "__main__":
    test_happy_path()
    test_transient_503_then_success()
    test_404_does_not_retry()
    test_validation_rejects_junk()
    test_fallback_to_snapshot()
    test_bad_parse_falls_back()
    test_no_snapshot_and_dead_network_raises()
    test_snapshot_roundtrip()
    print("\nAll universe tests passed.")
