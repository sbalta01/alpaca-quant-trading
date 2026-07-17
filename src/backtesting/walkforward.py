# src/backtesting/walkforward.py
"""
Walk-forward evaluation harness (fix-order item 3 in REVIEW_findings_and_roadmap).

Replaces the single 70/30 split with rolling refits: the model is refit every
`test_days`, each fold's positions are kept only for its out-of-sample segment,
and the segments are stitched into one continuous OOS track record. A
`purge_days` gap (>= the strategy's label horizon) separates each train window
from its test segment so overlapping-horizon labels cannot leak.

It reuses the existing Strategy.generate_signals interface unmodified:

* ML strategies (those with a `train_frac` attribute) split chronologically at
  `train_frac` internally. Per fold we slice the data to [slice_start, test_end)
  and set `train_frac` so the internal boundary lands at (test_start - purge).
  Strategies drop an unknown number of feature-warmup rows before splitting,
  which shifts the realized boundary; `warmup_buffer` is a conservative upper
  bound on those dropped rows, chosen so the realized train end can only
  UNDERSHOOT the intended boundary (slightly less training data - harmless),
  never overshoot into the purge gap (leak).

* Rule-based strategies (no `train_frac`) are causal by construction, so they
  run once over the full history; everything from the first test date onward is
  their out-of-sample window (the same window as ML strategies, for fair
  comparison).

Usage:
    harness = WalkForwardHarness(lambda: XGBoostRegressionStrategy(...), data,
                                 train_days=756, test_days=63, purge_days=10)
    results = harness.run()
    print(harness.performance(results))
"""
from typing import Callable, List

import numpy as np
import pandas as pd

from src.strategies.base_strategy import Strategy
from src.strategies.weekly_momentum import performance_metrics


class WalkForwardHarness:
    def __init__(
        self,
        strategy_factory: Callable[[], Strategy],
        data: pd.DataFrame,
        train_days: int = 756,      # ~3y of daily bars per fit
        test_days: int = 63,        # refit quarterly
        purge_days: int = None,     # default: strategy.horizon if present, else 5
        expanding: bool = True,     # False = rolling window of train_days
        warmup_buffer: int = 100,   # upper bound on feature-warmup rows dropped internally
        min_train_rows: int = 100,  # sanity floor after the buffer haircut
        cost_bps: float = 10.0,
        verbose: bool = True,
    ):
        self.strategy_factory = strategy_factory
        self.data = data
        self.train_days = train_days
        self.test_days = test_days
        self.expanding = expanding
        self.warmup_buffer = warmup_buffer
        self.min_train_rows = min_train_rows
        self.cost_bps = cost_bps
        self.verbose = verbose

        probe = strategy_factory()
        if probe.multi_symbol:
            raise NotImplementedError(
                "WalkForwardHarness supports single-symbol strategies; "
                "cross-sectional strategies have their own backtester "
                "(see src/strategies/weekly_momentum.py).")
        self.needs_fit = hasattr(probe, "train_frac")
        self.purge_days = purge_days if purge_days is not None else int(getattr(probe, "horizon", 5))
        self.folds_: List[dict] = []   # filled by run(); introspectable in tests

    # ------------------------------------------------------------------ folds
    def _fold_bounds(self, n_rows: int) -> List[dict]:
        """Row-position fold boundaries for one symbol."""
        first_test = self.train_days + self.purge_days
        if first_test + 5 > n_rows:
            raise ValueError(
                f"Not enough history: need > {first_test + 5} rows "
                f"(train_days + purge_days + 5), got {n_rows}")
        folds = []
        for test_start in range(first_test, n_rows, self.test_days):
            test_end = min(test_start + self.test_days, n_rows)
            if test_end - test_start < 5:   # skip a tiny tail fold
                break
            slice_start = 0 if self.expanding else max(
                0, test_start - self.purge_days - self.train_days)
            n_train = (test_start - self.purge_days) - slice_start
            n_slice = test_end - slice_start
            frac = (n_train - self.warmup_buffer) / (n_slice - self.warmup_buffer)
            if n_train - self.warmup_buffer < self.min_train_rows:
                raise ValueError(
                    f"train window too small: {n_train} rows minus warmup_buffer "
                    f"{self.warmup_buffer} < min_train_rows {self.min_train_rows}")
            folds.append(dict(slice_start=slice_start, test_start=test_start,
                              test_end=test_end, train_frac=frac))
        return folds

    # ------------------------------------------------------------------- run
    def _run_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        folds = self._fold_bounds(n)
        first_test = folds[0]["test_start"]
        position = pd.Series(0.0, index=df.index)

        if not self.needs_fit:
            # Causal rules: one pass over full history equals per-fold reruns.
            strat = self.strategy_factory()
            out = strat.generate_signals(df.copy())
            position.iloc[first_test:] = out["position"].iloc[first_test:].values
            if self.verbose:
                print(f"[{symbol}] rule-based strategy, single causal pass; "
                      f"OOS from {df.index[first_test].date()}")
        else:
            for k, f in enumerate(folds):
                sl = df.iloc[f["slice_start"]:f["test_end"]].copy()
                strat = self.strategy_factory()
                strat.train_frac = f["train_frac"]
                if self.verbose:
                    print(f"[{symbol}] fold {k + 1}/{len(folds)}: "
                          f"train {df.index[f['slice_start']].date()} -> "
                          f"{df.index[f['test_start'] - self.purge_days - 1].date()}, "
                          f"purge {self.purge_days}d, test "
                          f"{df.index[f['test_start']].date()} -> "
                          f"{df.index[f['test_end'] - 1].date()} "
                          f"(train_frac={f['train_frac']:.3f})")
                out = strat.generate_signals(sl)
                seg = out["position"].iloc[f["test_start"] - f["slice_start"]:]
                position.loc[seg.index] = seg.values
        self.folds_.append(dict(symbol=symbol, folds=folds))

        gross = (df["close"].pct_change() * position.shift(1)).fillna(0.0)
        trades = position.diff().abs()
        trades.iloc[0] = abs(position.iloc[0])
        res = pd.DataFrame(index=df.index)
        res["close"] = df["close"]
        res["position"] = position
        res["returns"] = gross - trades * self.cost_bps / 1e4
        res["test_mask"] = 0.0
        res.iloc[first_test:, res.columns.get_loc("test_mask")] = 1.0
        return res

    def run(self) -> pd.DataFrame:
        """Returns MultiIndex (symbol, timestamp) frame with stitched OOS
        positions, net returns, and a test_mask marking the OOS window."""
        self.folds_ = []
        results = []
        for symbol, sub in self.data.groupby(level="symbol"):
            res = self._run_symbol(symbol, sub.droplevel("symbol"))
            res["symbol"] = symbol
            results.append(res)
        final = pd.concat(results).set_index("symbol", append=True)
        return final.reorder_levels(["symbol", final.index.names[0]]).sort_index()

    # ----------------------------------------------------------- performance
    def performance(self, results: pd.DataFrame) -> pd.DataFrame:
        """OOS-only metrics per symbol, strategy vs buy-and-hold of the same
        symbol over the identical stitched window, plus equal-weight aggregate."""
        rows = {}
        strat_rets, bench_rets = [], []
        for symbol, sub in results.groupby(level="symbol"):
            sub = sub.droplevel("symbol")
            oos = sub[sub["test_mask"] == 1.0]
            r = oos["returns"]
            bh = oos["close"].pct_change().fillna(0.0)
            rows[(symbol, "Strategy")] = performance_metrics(r)
            rows[(symbol, "Buy&Hold")] = performance_metrics(bh)
            strat_rets.append(r)
            bench_rets.append(bh)
        if len(strat_rets) > 1:
            ew = pd.concat(strat_rets, axis=1).mean(axis=1)
            ew_bh = pd.concat(bench_rets, axis=1).mean(axis=1)
            rows[("ALL (equal-weight)", "Strategy")] = performance_metrics(ew)
            rows[("ALL (equal-weight)", "Buy&Hold")] = performance_metrics(ew_bh)
        return pd.DataFrame(rows).T
