# Deployment Guide

How to run this repo yourself, for backtesting and for live/paper trading. The recommended production path is the **weekly momentum portfolio** (`src/strategies/weekly_momentum.py`) deployed through **GitHub Actions on a Friday-evening schedule**, exactly like your existing monthly rebalance job. Reasons: it needs no GPU/torch/sklearn so CI runs are fast and cheap, it has no model files to version, every decision is reproducible from the day's closes, and it reuses the Alpaca execution pattern you already trust.

## 0. One-time setup (local)

```bash
cd alpaca-quant-trading
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create `.env` in the repo root (never commit it):

```env
API_KEY=your_alpaca_key
API_SECRET=your_alpaca_secret
PAPER=True
```

Get the keys at alpaca.markets → your dashboard → API Keys. Paper and live have *different* keys; start with paper. `PAPER` is now parsed as a real boolean — `False` means real money.

## 1. Backtesting

**Weekly momentum (the deployable strategy):**

```bash
python main/backtest_weekly_momentum.py                    # NASDAQ-100, 10y, zero costs
python main/backtest_weekly_momentum.py --cost-bps 10      # realistic-cost variant - run this too
python main/backtest_weekly_momentum.py --top-k 12 --years 8
python main/backtest_weekly_momentum.py --no-regime-gate   # ablation: is the gate helping?
python main/backtest_weekly_momentum.py --tickers AAPL MSFT NVDA AMZN GOOG META AVGO ...
```

It prints: metrics table (Strategy vs equal-weight universe vs QQQ vs SPY over the identical window), yearly returns, annualized turnover with its cost implication, and **the exact portfolio it would buy today**. Defaults now match live: rank buffering (`--buffer-mult 1.5`, hold a name until it exits the top 15) and the live 0.5% no-trade band — together they halve turnover (19x -> ~10x/yr) at unchanged Sharpe. `--buffer-mult 1.0 --min-trade-fraction 0` recovers the old behavior.

**Diagnostics** (run these before trusting any config change):

```bash
python main/sensitivity_weekly_momentum.py --null-test   # is the signal real, or concentration luck?
python main/sensitivity_weekly_momentum.py --oat         # parameter sensitivity + weekday robustness
```

The 2026-07 diagnostic verdict: the momentum signal's CAGR contribution is unambiguous (100th percentile vs 100 turnover-matched random-selection portfolios), its Sharpe contribution is ~90th percentile (real but modest - the signal picks higher-vol names), and no single parameter change moves Sharpe below ~1.1. Only ~9% of parameter variants beat the equal-weight universe on Sharpe: this strategy is a return-enhancer, not a risk-adjusted-return enhancer. The whole history is walk-forward (signals never see the future), but remember the universe is today's index membership, so absolute numbers are survivorship-flattered — the fairest read is Strategy vs the equal-weight benchmark of the *same* universe.

**Holistic weekly method (layers 2-4 on top of the momentum core):** compares the deployed method against variants adding short-term reversal, a pooled cross-sectional ML ranker, and an HMM+turbulence regime gate — each admitted only if it improves out-of-sample results:

```bash
python main/backtest_weekly_holistic.py                    # NASDAQ-100, 10y, zero cost
python main/backtest_weekly_holistic.py --cost-bps 10      # realistic-cost variant
```

**Single-name strategies (research stack):** edit the symbol/strategy block in `main/backtest.py`, then:

```bash
python main/backtest.py
```

The engine now charges `cost_bps=10` per side by default and computes Sharpe/Sortino on the out-of-sample window only. Pass `cost_bps=0` to `run_backtest_strategy(...)` if you want the old frictionless numbers.

**Walk-forward re-evaluation (research stack):** judges a strategy only on stitched out-of-sample segments, refit every quarter with a purge gap — the honest way to read any ML result here:

```bash
python main/walkforward_eval.py --strategy macd --symbols AAPL MSFT SPY
python main/walkforward_eval.py --strategy xgboost --symbols AAPL --fast   # no Optuna/RFECV
python main/walkforward_eval.py --strategy xgboost --symbols AAPL          # full research config (slow)
```

**Regression tests** (no network needed): `python tests/test_engine_synthetic.py`, `python tests/test_walkforward_synthetic.py`, `python tests/test_weekly_holistic_synthetic.py`

## 2. Live / paper trading — manual first

Always dry-run first. It prints the target portfolio and every order it *would* place, touching nothing:

```bash
python main/deploy_weekly_momentum.py            # dry run
python main/deploy_weekly_momentum.py --execute  # submits orders (paper while PAPER=True)
```

`--holistic` switches to the full six-layer method (reversal + ML ranker + HMM/turbulence gate). As of the 2026-07 comparison the extra layers cut volatility and drawdown but cost too much return and turnover to be admitted (Sharpe 1.07 vs 1.23 at 10 bps) - keep the default. Re-run `main/backtest_weekly_holistic.py` before ever flipping it, and paper-trade the switch like any new strategy.

Orders are market DAY orders: run after the close and they queue for the next open. Sells are submitted before buys; dropped names are liquidated by share quantity; trades under 0.5% of equity are skipped to control churn. Each run appends to `live_weekly_momentum.md`.

Hardening (2026-07): the universe scrape now retries with backoff, validates the parse (count, ticker syntax, overlap with the last known-good list), and falls back to the committed `src/data/nasdaq100_snapshot.csv` on failure - a Wikipedia outage can no longer fail (or worse, mis-trade) the Friday run. Prices use a bounded `ffill(limit=5)` so a halted/delisted name drops out instead of being carried at a flat price, and the run aborts if the latest bar is more than 4 days old. Rank buffering (`--buffer-mult`, default 1.5) treats your actual account positions as incumbents.

## 3. Live / paper trading — scheduled (recommended)

The workflow `.github/workflows/deploying-weekly-momentum.yml` runs every Friday 21:30 UTC (after US close; fills happen at Monday's open).

One-time GitHub setup:
1. Push the repo to GitHub (private recommended).
2. Repo → Settings → Secrets and variables → Actions → add `API_KEY` and `API_SECRET` (paper keys first).
3. The workflow has `PAPER: True` hardcoded — flip it only when you're ready for real money.
4. Actions tab → enable workflows → you can trigger a first run manually via "Run workflow" (workflow_dispatch) to check the logs.
5. Each run commits its report (`live_weekly_momentum.md`) back to the repo, so your trade history is versioned.

Your existing monthly Monte-Carlo rebalance (`deploying-rebalance-portfolio.yml`) can keep running in parallel — they're separate sleeves; just be aware they share the same Alpaca account equity unless you split accounts.

## 4. Go-live checklist

Paper trade the weekly job for **at least 8-12 weeks** and only then consider real money, verifying: fills arrive Monday open at sane prices; live weekly returns are within the backtest's typical weekly range; turnover matches the backtest estimate; the regime gate state matches what you'd compute by hand. Then: create live API keys, replace the two GitHub secrets, set `PAPER: False` in the workflow, and start with a small fraction of intended capital. Keep position caps (20%/name) and the exposure gate on.

## 5. Maintenance

Weekly: skim the committed report (holdings, exposure, errors). Monthly: re-run the backtest to confirm live tracking; check for delisted/renamed tickers in the universe fetch. Quarterly: re-run `--cost-bps 10` and the `--no-regime-gate` ablation to confirm the assumptions still hold. If GitHub's Wikipedia fetch for NASDAQ-100 membership breaks, pin the universe with `--tickers` (both scripts accept it).

---
*This repository is research/engineering tooling, not investment advice. Markets can and will behave unlike any backtest; never deploy money you cannot afford to lose.*
