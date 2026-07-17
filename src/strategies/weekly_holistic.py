# src/strategies/weekly_holistic.py
"""
Holistic weekly method: all six layers of REVIEW_findings_and_roadmap section 4.

Layers 1, 5 and the simple 200dma gate live in weekly_momentum.py and are
reused here. This module adds the remaining layers, each individually
toggleable so it can be admitted only if it improves out-of-sample results:

  Layer 2 - Short-term reversal : last-`rev_window`-day return, z-scored
            cross-sectionally and NEGATIVELY weighted into the signal score
            (dampens buying into spikes; ~uncorrelated with momentum).
  Layer 3 - ML relative-return ranker : ONE pooled cross-sectional XGBoost
            model (not per-symbol). Stationary features only, z-scored within
            each weekly cross-section; target = next-week return percentile
            rank within the universe. Trained walk-forward on past weekly
            cross-sections whose targets are fully realized, refit every
            `ml_refit_every` rebalances. Its rank is averaged with the
            layer-1/2 rank; set ml_weight=0 to remove it.
  Layer 4 - Regime gate : three independent health checks - SPY >= 200dma,
            a 3-state GaussianHMM on SPY [return, vol] (bear = lowest-mean
            state) fit walk-forward, and a cross-sectional turbulence index
            (Mahalanobis distance, trailing window) vs its own trailing
            90th percentile. Gross exposure = 1.0 / 0.65 / 0.3 for
            0 / 1 / >=2 unhealthy checks.

With use_reversal=False, ml_weight=0 and gate="simple",
run_walkforward_holistic reproduces weekly_momentum.run_walkforward exactly
(regression-tested in tests/test_weekly_holistic_synthetic.py).

Everything uses data up to and including the signal date only - no lookahead.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from src.strategies.weekly_momentum import (
    inverse_vol_weights, momentum_scores, regime_exposure, weekly_rebalance_dates,
)


# --------------------------------------------------------------- Layer 2
def reversal_scores(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Cross-sectional z-score of the last-`window`-day return (row t uses
    data up to t). Caller applies the NEGATIVE weight."""
    rev = prices / prices.shift(window) - 1.0
    return rev.sub(rev.mean(axis=1), axis=0).div(rev.std(axis=1) + 1e-12, axis=0)


def signal_scores(
    prices: pd.DataFrame,
    lookbacks: Tuple[int, ...] = (252, 126),
    skip: int = 21,
    use_reversal: bool = True,
    rev_window: int = 5,
    rev_weight: float = 0.25,
) -> pd.DataFrame:
    """Layer 1 momentum z, optionally blended with layer 2 (negative) reversal z."""
    score = momentum_scores(prices, lookbacks, skip)
    if use_reversal:
        score = (1.0 - rev_weight) * score - rev_weight * reversal_scores(prices, rev_window)
    return score


# --------------------------------------------------------------- Layer 3
FEATURE_BUILDERS = {
    "r1w": lambda px, rets: px / px.shift(5) - 1.0,
    "r1m": lambda px, rets: px / px.shift(21) - 1.0,
    "r3m": lambda px, rets: px / px.shift(63) - 1.0,
    "r6m_skip": lambda px, rets: px.shift(21) / px.shift(126) - 1.0,
    "r12m_skip": lambda px, rets: px.shift(21) / px.shift(252) - 1.0,
    "vol21": lambda px, rets: rets.rolling(21).std(),
    "vol63": lambda px, rets: rets.rolling(63).std(),
    "px_sma50": lambda px, rets: px / px.rolling(50).mean() - 1.0,
    "px_sma200": lambda px, rets: px / px.rolling(200).mean() - 1.0,
    "rsi14": lambda px, rets: _rsi_wide(px, 14),
}


def _rsi_wide(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def build_feature_panel(prices: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Long panel indexed (date, ticker): stationary features z-scored within each
    weekly cross-section, plus 'target' = NEXT-rebalance return percentile rank
    in [0, 1] (NaN for the latest date, whose target is not yet realized).
    Row (t, s) uses prices up to t only; the target uses (t, next rebalance].
    """
    rets = prices.pct_change()
    feats = {}
    for name, fn in FEATURE_BUILDERS.items():
        wide = fn(prices, rets).reindex(rebal_dates)
        feats[name] = wide.sub(wide.mean(axis=1), axis=0).div(wide.std(axis=1) + 1e-12, axis=0)

    px_rebal = prices.reindex(rebal_dates)
    fwd_ret = px_rebal.shift(-1) / px_rebal - 1.0          # return over (t, t+1 rebal]
    target = fwd_ret.rank(axis=1, pct=True)

    panel = pd.concat({**feats, "target": target}, axis=1)  # cols: (feature, ticker)
    panel = panel.stack(future_stack=True)                  # index: (date, ticker)
    return panel.dropna(subset=list(FEATURE_BUILDERS))


class CrossSectionalRanker:
    """Pooled walk-forward XGBoost ranker over weekly cross-sections."""

    def __init__(self, refit_every: int = 4, min_train_weeks: int = 104,
                 random_state: int = 42):
        self.refit_every = refit_every
        self.min_train_weeks = min_train_weeks
        self.random_state = random_state
        self.model = None
        self._fits = 0
        self._calls = 0
        self.last_train_date = None      # introspectable in tests

    def scores_for(self, panel: pd.DataFrame, date: pd.Timestamp,
                   prev_date: pd.Timestamp) -> pd.Series:
        """
        Cross-sectional z-scored predictions for `date`, or an empty Series if
        there is not yet enough history. Training uses only cross-sections
        dated <= prev_date, whose targets are realized by `date` - the newest
        usable information with zero lookahead.
        """
        from xgboost import XGBRegressor

        train = panel[(panel.index.get_level_values(0) <= prev_date)
                      & panel["target"].notna()]
        n_weeks = train.index.get_level_values(0).nunique()
        if n_weeks < self.min_train_weeks:
            self._calls += 1
            return pd.Series(dtype=float)

        if self.model is None or self._calls % self.refit_every == 0:
            X = train[list(FEATURE_BUILDERS)]
            self.model = XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, verbosity=0, n_jobs=-1)
            self.model.fit(X, train["target"])
            self._fits += 1
            self.last_train_date = train.index.get_level_values(0).max()
        self._calls += 1

        try:
            today = panel.xs(date, level=0)
        except KeyError:
            return pd.Series(dtype=float)
        pred = pd.Series(self.model.predict(today[list(FEATURE_BUILDERS)]),
                         index=today.index)
        return (pred - pred.mean()) / (pred.std() + 1e-12)


# --------------------------------------------------------------- Layer 4
def turbulence_at(prices: pd.DataFrame, date: pd.Timestamp, window: int = 252) -> float:
    """Mahalanobis distance of `date`'s cross-section of daily returns from the
    trailing `window`-day mean/cov (both ending the previous day - causal)."""
    rets = prices.loc[:date].pct_change(fill_method=None)
    if len(rets) <= window:
        return 0.0
    hist = rets.iloc[-window - 1:-1]
    x = rets.iloc[-1]
    cols = hist.columns[hist.notna().all() & x.notna()]
    if len(cols) < 2:
        return 0.0
    delta = (x[cols] - hist[cols].mean()).values
    inv_cov = np.linalg.pinv(hist[cols].cov().values)
    return float(delta @ inv_cov @ delta)


class RegimeGate:
    """Three walk-forward health checks -> gross exposure in {0.3, 0.65, 1.0}."""

    def __init__(self, ma_window: int = 200, hmm_states: int = 3,
                 hmm_refit_every: int = 4, hmm_min_history: int = 504,
                 turb_window: int = 252, turb_pct: float = 0.90,
                 turb_min_obs: int = 52, random_state: int = 42):
        self.ma_window = ma_window
        self.hmm_states = hmm_states
        self.hmm_refit_every = hmm_refit_every
        self.hmm_min_history = hmm_min_history
        self.turb_window = turb_window
        self.turb_pct = turb_pct
        self.turb_min_obs = turb_min_obs
        self.random_state = random_state
        self._hmm = None
        self._hmm_mu = self._hmm_sd = None
        self._bear_state = None
        self._calls = 0
        self.turb_history = []           # turbulence at past rebalance dates
        self.last_checks = {}            # introspectable

    def _hmm_features(self, bench: pd.Series) -> pd.DataFrame:
        ret = bench.pct_change()
        return pd.DataFrame({"ret": ret, "vol": ret.rolling(21).std()}).dropna()

    def _hmm_unhealthy(self, bench: pd.Series) -> bool:
        from hmmlearn.hmm import GaussianHMM

        X = self._hmm_features(bench)
        if len(X) < self.hmm_min_history:
            return False
        if self._hmm is None or self._calls % self.hmm_refit_every == 0:
            self._hmm_mu, self._hmm_sd = X.mean(), X.std() + 1e-12
            Xs = ((X - self._hmm_mu) / self._hmm_sd).values
            self._hmm = GaussianHMM(n_components=self.hmm_states,
                                    covariance_type="full", n_iter=200,
                                    random_state=self.random_state)
            self._hmm.fit(Xs)
            self._bear_state = int(np.argmin(self._hmm.means_[:, 0]))
        Xs = ((X - self._hmm_mu) / self._hmm_sd).values
        post = self._hmm.predict_proba(Xs)[-1]
        return int(np.argmax(post)) == self._bear_state

    def exposure(self, bench: pd.Series, prices: pd.DataFrame,
                 date: pd.Timestamp) -> float:
        ma_bad = regime_exposure(bench, self.ma_window, low_exposure=0.0) == 0.0
        hmm_bad = self._hmm_unhealthy(bench)
        turb = turbulence_at(prices, date, self.turb_window)
        if len(self.turb_history) >= self.turb_min_obs:
            turb_bad = turb > np.percentile(self.turb_history, self.turb_pct * 100)
        else:
            turb_bad = False
        self.turb_history.append(turb)
        self._calls += 1
        self.last_checks = {"ma_bad": ma_bad, "hmm_bad": hmm_bad, "turb_bad": turb_bad}
        n_bad = int(ma_bad) + int(hmm_bad) + int(turb_bad)
        return 1.0 if n_bad == 0 else (0.65 if n_bad == 1 else 0.3)


# --------------------------------------------------------- combined method
def compute_holistic_weights(
    prices: pd.DataFrame,
    score_today: pd.Series,
    ml_scores: pd.Series,
    ml_weight: float,
    exposure: float,
    top_k: int = 10,
    vol_window: int = 63,
    weight_cap: float = 0.20,
) -> pd.Series:
    """Rank-average base and ML scores, take top_k, size inverse-vol, scale by
    the regime exposure. Weights sum to <= 1; remainder is cash."""
    score_today = score_today.dropna()
    if score_today.empty:
        return pd.Series(dtype=float)
    if ml_weight > 0 and len(ml_scores) > 0:
        base_rank = score_today.rank(pct=True)
        ml_rank = ml_scores.reindex(score_today.index).rank(pct=True)
        combined = (1 - ml_weight) * base_rank + ml_weight * ml_rank.fillna(base_rank)
    else:
        combined = score_today
    members = combined.nlargest(top_k).index
    w = inverse_vol_weights(prices, members, vol_window, weight_cap)
    return (w * exposure).sort_values(ascending=False)


def compute_holistic_target_weights(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    top_k: int = 10,
    vol_window: int = 63,
    weight_cap: float = 0.20,
    use_reversal: bool = True,
    rev_window: int = 5,
    rev_weight: float = 0.25,
    ml_weight: float = 0.5,
    ml_min_train_weeks: int = 104,
    turb_warmup_weeks: int = 104,
    random_state: int = 42,
) -> pd.Series:
    """
    One-shot live/paper counterpart of weekly_momentum.compute_target_weights
    for the FULL holistic method: target weights as of prices.index[-1], using
    only data up to that date. Needs ~3.5y of history for the ML layer to be
    active (it degrades gracefully to layers 1-2 + gate below that).
    """
    rebals = weekly_rebalance_dates(prices.index)
    date = rebals[-1]
    score_today = signal_scores(prices, use_reversal=use_reversal,
                                rev_window=rev_window,
                                rev_weight=rev_weight).loc[date]

    ml_scores = pd.Series(dtype=float)
    if ml_weight > 0 and len(rebals) >= 2:
        panel = build_feature_panel(prices, rebals)
        ranker = CrossSectionalRanker(refit_every=1,
                                      min_train_weeks=ml_min_train_weeks,
                                      random_state=random_state)
        ml_scores = ranker.scores_for(panel, date, rebals[-2])

    gate = RegimeGate(random_state=random_state)
    # seed the turbulence percentile with trailing rebalance-date history
    for d in rebals[-(turb_warmup_weeks + 1):-1]:
        gate.turb_history.append(turbulence_at(prices, d, gate.turb_window))
    expo = gate.exposure(benchmark.loc[:date], prices.loc[:date], date)

    return compute_holistic_weights(prices.loc[:date], score_today, ml_scores,
                                    ml_weight, expo, top_k, vol_window, weight_cap)


def run_walkforward_holistic(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    top_k: int = 10,
    lookbacks: Tuple[int, ...] = (252, 126),
    skip: int = 21,
    vol_window: int = 63,
    weight_cap: float = 0.20,
    # layer 2
    use_reversal: bool = True,
    rev_window: int = 5,
    rev_weight: float = 0.25,
    # layer 3
    ml_weight: float = 0.5,
    ml_refit_every: int = 4,
    ml_min_train_weeks: int = 104,
    # layer 4: "simple" = 200dma only (original), "hmm" = full gate, None = always 1.0
    gate: str = "simple",
    ma_window: int = 200,
    low_exposure: float = 0.4,
    turb_window: int = 252,
    turb_pct: float = 0.90,
    cost_bps: float = 0.0,
    warmup: int = None,
    random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Walk-forward weekly backtest of the holistic method. Same conventions as
    weekly_momentum.run_walkforward: weights decided at rebalance close t apply
    from the next trading day; cost_bps charged on turnover.

    Returns dict with 'daily_returns', 'weights', 'turnover', 'exposure',
    'ml_active_from' (first date the ML layer contributed, or None).
    """
    warmup = warmup if warmup is not None else max(max(lookbacks) + skip, ma_window) + 5
    if warmup >= len(prices.index):
        raise ValueError(
            f"Not enough history: need > {warmup} rows for warm-up, got {len(prices.index)}")
    daily_rets = prices.pct_change().fillna(0.0)
    all_rebals = weekly_rebalance_dates(prices.index)
    rebal_dates = [d for d in all_rebals if d >= prices.index[warmup]]

    scores = signal_scores(prices, lookbacks, skip, use_reversal, rev_window, rev_weight)
    use_ml = ml_weight > 0
    ranker = panel = None
    if use_ml:
        ranker = CrossSectionalRanker(ml_refit_every, ml_min_train_weeks, random_state)
        panel = build_feature_panel(prices, all_rebals)
    regime = RegimeGate(ma_window=ma_window, turb_window=turb_window,
                        turb_pct=turb_pct,
                        random_state=random_state) if gate == "hmm" else None

    w_current = pd.Series(dtype=float)
    port_ret = pd.Series(0.0, index=prices.index)
    weights_hist, turnover_hist, expo_hist = {}, {}, {}
    ml_active_from = None

    rebal_set = set(rebal_dates)
    rebal_pos = {d: i for i, d in enumerate(all_rebals)}
    pending_weights = None
    for date in prices.index:
        if pending_weights is not None:
            w_current = pending_weights
            pending_weights = None
        if len(w_current) > 0:
            port_ret.loc[date] = float((daily_rets.loc[date, w_current.index] * w_current).sum())
        if date in rebal_set:
            hist = prices.loc[:date]
            ml_scores = pd.Series(dtype=float)
            if use_ml:
                pos = rebal_pos[date]
                prev = all_rebals[pos - 1] if pos > 0 else date
                ml_scores = ranker.scores_for(panel, date, prev)
                if len(ml_scores) > 0 and ml_active_from is None:
                    ml_active_from = date
            if regime is not None:
                expo = regime.exposure(benchmark.loc[:date], hist, date)
            elif gate == "simple":
                expo = regime_exposure(benchmark.loc[:date], ma_window, low_exposure)
            else:
                expo = 1.0
            w_new = compute_holistic_weights(
                hist, scores.loc[date], ml_scores, ml_weight if use_ml else 0.0,
                expo, top_k, vol_window, weight_cap)
            all_names = w_new.index.union(w_current.index)
            turnover = float(
                (w_new.reindex(all_names, fill_value=0.0)
                 - w_current.reindex(all_names, fill_value=0.0)).abs().sum())
            port_ret.loc[date] -= turnover * cost_bps / 1e4
            weights_hist[date] = w_new
            turnover_hist[date] = turnover
            expo_hist[date] = float(w_new.sum())
            pending_weights = w_new
            if verbose:
                print(f"{date.date()}: expo={expo:.2f} turnover={turnover:.2f} "
                      f"ml={'on' if len(ml_scores) > 0 else 'off'}")

    return {
        "daily_returns": port_ret,
        "weights": weights_hist,
        "turnover": pd.Series(turnover_hist),
        "exposure": pd.Series(expo_hist),
        "ml_active_from": ml_active_from,
    }
