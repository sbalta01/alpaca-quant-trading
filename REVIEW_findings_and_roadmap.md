# Repository Review: Findings and Roadmap

> **Status update:** Fix-order items 1, 2, 7 and 8 plus the section-1 bug list are now implemented (engine costs + test-window metrics, label-leak fixes, hardcoded-override removal, TradingEnv reward, RegimeSwitching kwargs, Bollinger/RollingWindow position columns, live-executor sell sizing, PAPER parsing, rebalance renormalization + sells-first, EVT return type, data-source flags replacing the deliberate-exception hacks). Regression tests: `python tests/test_engine_synthetic.py`. Still open: walk-forward harness (fix-order 3-4) and the weekly method layers (5-6).

Scope: full read of `src/`, `main/`, workflows, and live logs; mechanics verified in a sandbox with synthetic data (network-restricted, so no live-data backtests — sklearn/torch strategies were verified by inspection, the engine and rule-based strategies by execution).

**What is sound:** the P&L timing in `BacktestEngine` (`position.shift(1)`) is correct — no lookahead in the engine itself (verified numerically). The separation strategies/engine/execution is good, the benchmark-on-test-window idea is right, and the walk-forward CV (`TimeSeriesSplit`) inside model selection is correct. The live rebalance pipeline works end-to-end.

---

## 1. Correctness bugs

| # | Where | Bug |
|---|-------|-----|
| 1 | `src/env/trading_env.py:104-109` | **RL reward is broken.** After `t += 1`, `cur_val` is priced at `prices[self.t-1]` — the *same* bar the trade happened on. Reward = −transaction costs, always. Verified numerically: on a tape rising 10%/step with full allocation, rewards were −1.00 and −0.00 while true P&L was +100 and +110. The CLSTM-PPO agent is trained to minimize trading, nothing else. Any RL results are meaningless until this is fixed (`cur_val` must use `prices[self.t]`). |
| 2 | `regime_switching_factor_ML.py:71-78` | Passes `short_ma`, `long_ma`, `angle_threshold_deg` to `HybridAdaBoostFilterStrategy.__init__`, which doesn't accept them → `TypeError` on construction. The strategy cannot currently run. Also: bull and bear regimes dispatch to the *same* model, so the regime layer only zeroes out "sideways" days. |
| 3 | `bollinger_mean_reversion.py`, `rolling_window_ML.py` | Emit no `position` column → `KeyError` in `BacktestEngine._run_single`. Their `signal` is also a level (+1/0/−1), while the engine treats `signal` as a trade diff. Incompatible with the current engine. |
| 4 | `live_executor.py:100-107` | SELL submits `notional=cash_per_trade` (fixed $) instead of the actual position size. If the position appreciated you leave a residual; if it fell, the order can fail. Sells should use held `qty`. |
| 5 | `live_executor.py` / `rebalance_portfolio.py` | `PAPER = os.getenv("PAPER")` is a *string*; `TradingClient(paper="False")` is truthy → always paper mode. Fails safe, but the flag does nothing. Parse it explicitly. |
| 6 | `monte_carlo.py:146-148` | `compute_evt_var_cvar` returns a **dict** in the `k < 5` branch but a **tuple** otherwise; callers index `evt[0]` → `KeyError` when tail data is thin. |
| 7 | `rebalance_portfolio.py:57` | `optimized_portfolio.loc[mask] = 0.0` zeroes entire rows (all columns), and weights are **not renormalized** after small positions are dropped → targets sum to <1, a slice of equity silently stays wherever it was. |
| 8 | `rebalance_portfolio.py` | Orders are submitted in weight order, not sells-first → buys can hit insufficient buying power (your own log shows a $10k XEL buy submitted before most sells). Also `total_portfolio_notional` counts only symbols in the target list, so cash and delisted holdings are excluded from sizing. |
| 9 | `momentum_ranking_adaboost_ML.py:29-37` | `dropna(subset=["target"])` only — rolling features keep NaNs in the first ~26 rows, which `StandardScaler`/`RFECV` will reject. Likely crashes on fresh data. |
| 10 | `adaboost_ML.py:133`, `xgboost_regression_ML.py:251`, `lstm_event_technical_ML.py:591` | `feat = feat.ffill().dropna()` runs **after** the target column is added → the last `horizon` rows get their unknowable labels forward-filled from the previous row and are used in training. Fabricated labels, exactly where the model is used live. Drop those rows from training instead. |
| 11 | Repo-wide | Deliberate exceptions to force the Yahoo path (`raise ValueError("Chose not to use Alpaca data")`, the `lmkdscsa` NameError in `live_executor.py:140`) plus bare `except:` everywhere. In `backtest_executor.py` a bare except silently swaps the benchmark to the *full* period if anything in the control branch fails — you can't tell which comparison you're looking at. Replace with an explicit `data_source` flag and narrow exceptions. |
| 12 | `hybrid_adaboost_filter_ML.py:51` | `df.drop(columns=["signal","position"])` without `inplace`/reassignment — dead code. |

## 2. Why the backtests haven't been trustworthy

These are the methodological issues that most plausibly explain "research hub, but backtests not successful / not reliable":

**No transaction costs or slippage.** The engine assumes zero cost, full-notional fills at close. Verified impact: for a low-churn MACD (9 round trips / 8y) drag at 10 bps is ~1.8% — tolerable; for anything daily-rebalanced (MomentumRanking top-k churns every day) it is fatal. Add a per-side bps cost to `_run_single` (one line: `ret -= position.diff().abs() * cost_bps/1e4`) before believing any result.

**Metrics are computed over the wrong window.** ML strategies hold position 0 during the train fraction, but `performance()` computes Sharpe/Sortino over the full series. Verified: a test-only Sharpe of −1.07 reports as −0.58 (scales by ~√test_frac). Positive Sharpes get deflated, negative ones flattered, and every strategy with different `train_frac` is on a different scale. Compute metrics on the test mask only.

**Single 70/30 split.** Every ML strategy is judged on one test window. One window = one draw; a strategy that looks good on 2021-2025 tech tape is mostly measuring beta. Move to rolling walk-forward (refit every N months, stitch out-of-sample segments, with a `horizon`-sized purge gap between train and test).

**Lookahead in several places.** Pair trading selects pairs by correlation + cointegration computed over the *entire* sample including the test period; `TradingEnv` turbulence threshold is a full-sample percentile; `remove_outliers` uses full-sample quantiles. Pair trading additionally `clip(0,1)`s away the short leg — without both legs it is not a spread trade and the "cointegration" edge is gone.

**Non-stationary features.** AdaBoost/LSTM feature sets include raw `close`, SMA/EMA levels, cumulative OBV, and a VWAP anchored to the start of the data window. Tree splits and scaler statistics learned on 2018 price levels are meaningless in 2025. Use returns, ratios (close/SMA − 1), z-scores, and rolling-window VWAP.

**Weak target choice.** Predicting sign of ΔMA(d) is mostly predicting the smoother, not tradable returns — accuracy looks fine, P&L doesn't follow. The event-classification target in LSTMEventTechnical (return > threshold over horizon) is better; the cross-sectional *relative* return rank (below) is better still.

**The portfolio optimizer is an error maximizer.** The Monte-Carlo GBM sizing feeds `logrets.mean()` from ~2y of daily data into a return-maximizing objective. Sample means of daily equity returns have enormous estimation error, so weights chase noise/recent momentum. The GBM simulation itself adds nothing over closed-form (terminal returns are lognormal by construction — you're simulating what you could compute analytically); simulation would only earn its keep with bootstrapped/fat-tailed returns. The +24.4% live result since 2025-11 is real but should be attributed against an equal-weight QQQ benchmark over the same window before crediting the optimizer.

**Hardcoded overrides.** `adaboost_ML.py` sets `prob_positive_threshold = 0.7` and `adjust_threshold = False` inside `generate_signals`, silently overriding whatever you configure. Several thresholds appear tuned by eye against the test window — that is test-set leakage via the researcher.

**Live retraining every poll.** `live_executor` calls `fit_and_save` inside the polling loop — with `interval_seconds=60` that's a full LSTM ensemble retrain every minute. Retrain on a schedule (weekly), load otherwise.

**Engine accounting inconsistency (minor).** `final_equity` sums per-symbol silos (fixed $10k each, never rebalanced) while `total_returns` takes the cross-symbol mean (implies daily equal-weight rebalancing). The two disagree; pick one convention.

## 3. Which algorithms fit your actual goal

Goal: reliable retail framework — weekly signals or low-frequency Alpaca execution. Ranked by (evidence in your own repo) × (robustness of the underlying effect) × (fit to weekly cadence):

**Keep and promote:**
- **Monthly/weekly portfolio rebalance pipeline** (`rebalance_portfolio.py`) — already live and operationally proven. This is your production skeleton. Fix bugs #7/#8, replace the return-chasing objective (below).
- **Cross-sectional momentum ranking** (the *idea* behind `MomentumRankingAdaBoost`) — cross-sectional momentum is the best-documented equity anomaly and is exactly a weekly-cadence, long-only-friendly strategy. But the AdaBoost layer adds fragility without demonstrated edge: start with plain 12-1 and 6-1 momentum ranks.
- **Trend/regime filters** (`MovingAverage`, `MACD`, the HMM in RegimeSwitch) — weak as standalone signals (MACD lost money even gross of costs on a synthetic driftless tape, as expected), valuable as *exposure gates*.

**Keep as research, don't deploy:**
- **LSTMEventTechnical** — most engineering effort, decent design (walk-forward CV, bagging, focal loss), but heavy, fragile, single-name, and its edge is unproven once costs and test-window metrics are fixed. Re-evaluate under walk-forward after fixes.
- **XGBoostRegression** — same, but it's the best candidate for the ML sleeve below if re-targeted to *cross-sectional* ranking.

**Park:**
- **Pair trading** — needs shorts, borrow costs, rolling pair selection, and a hedge ratio; currently none of these. A real rewrite, and the payoff for retail (with short fees) is modest.
- **CLSTM-PPO** — reward function is broken (bug #1); even fixed, RL is the least sample-efficient path to a weekly retail signal. Lowest priority.

## 4. A holistic weekly method (combining independent information)

The principle: combine *decorrelated* information sources, each simple enough to trust, and let none of them size positions alone. All of this reuses code you already have.

**Cadence:** weekly, one GitHub-Actions run after Friday close (signals) + Monday open execution. Universe: NASDAQ-100 or S&P 500 (you already fetch both).

**Layer 1 — Cross-sectional momentum score** (new, small): 12-1-month and 6-1-month total returns, z-scored across the universe each week. This is the anchor sleeve; it needs no model fitting.

**Layer 2 — Short-term reversal score** (new, small): last-5-day return, *negatively* weighted. Weekly reversal is roughly uncorrelated with momentum and dampens buying into spikes.

**Layer 3 — ML relative-return ranker** (adapt `XGBoostRegression`): one pooled cross-sectional model (not per-symbol), stationary features only (returns, close/SMA ratios, vol, RSI, plus your VIX/Fama-French macros from `attach_factors`), target = next-week return *rank* within the universe. Trained walk-forward, refit monthly. Its output is a third score, rank-averaged with layers 1-2 — if it adds nothing out-of-sample, its weight goes to zero and you still have a working system.

**Layer 4 — Regime gate** (reuse `RegimeSwitch` HMM + turbulence in `tools.py`): fit the HMM walk-forward on SPY returns/vol; combine with a simple SPY-above-200dma check and your turbulence index. Output is a gross-exposure scalar in {0.3, 0.65, 1.0}, not per-stock signals. This is where MACD/MA logic earns its keep.

**Layer 5 — Portfolio construction** (adapt `monte_carlo.py`): take the top-k (~15-25) by combined rank. Drop the expected-return term from the optimizer entirely; size by inverse volatility (or minimum variance with Ledoit-Wolf shrinkage — `sklearn.covariance.LedoitWolf`, already a dependency), cap any name at ~10%, renormalize after threshold-dropping (fixes bug #7). Keep VaR/CVaR as a *veto/report* (your EVT machinery is nice for this), not as the objective.

**Layer 6 — Execution** (adapt `rebalance_portfolio.py`): sells before buys, qty-based closes for full exits, minimum-trade band (skip rebalance trades <0.5% of equity to kill churn), log everything to the markdown report as you do now.

**Validation protocol before real money:** engine with 10 bps/side costs; walk-forward 2015→today with purged splits; metrics on out-of-sample weeks only; benchmarks = equal-weight universe and QQQ buy-and-hold; then ≥3 months paper trading with the same code path as live (you already have paper trading — use it as the final test, not GitHub-log spot checks).

## 5. Suggested fix order

1. Add transaction costs to `BacktestEngine` and restrict metrics to the test mask (two small changes — every subsequent decision depends on them).
2. Fix label ffill leak (#10) and the hardcoded threshold overrides.
3. Build the walk-forward evaluation harness.
4. Re-evaluate existing strategies under it; expect most single-name ML results to shrink — that's the point.
5. Build layers 1-2 + 5-6 of the weekly method (no ML — deployable quickly on the proven rebalance skeleton).
6. Add layers 3-4, each admitted only if it improves out-of-sample results.
7. Fix live-executor sell sizing (#4) and rebalance sequencing (#8) before the next live run.
8. Fix `TradingEnv` reward only if/when you return to RL.
