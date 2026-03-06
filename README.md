# Quantitative Trading Framework
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

End-to-end quantitative trading framework for researching, backtesting, and deploying systematic strategies across equities.

## Project Context

This project is built to bridge research and execution:

- Strategy R&D for rule-based, machine-learning, and reinforcement-learning models.
- Unified backtesting with benchmark comparison.
- Live/paper-trading execution and monthly-style portfolio rebalancing.
- Risk-aware sizing using Monte Carlo simulation with VaR and CVaR controls.

The stack is Python-first and integrates `alpaca-py`, `yfinance`, `scikit-learn`, `xgboost`, `torch`, and `stable-baselines3`.

## Core Strengths

- Modular architecture: all strategies follow a shared interface in `src/strategies/base_strategy.py`.
- Broad strategy coverage: technical, statistical, supervised ML, deep learning, and CLSTM+PPO RL workflows.
- Practical execution pipeline: backtest and live executors are separated but reuse common components.
- Benchmarking discipline: strategy outputs are compared against buy-and-hold in the same period.
- Rich performance diagnostics: Max Drawdown, CAGR, Sharpe, Sortino, Calmar, Turnover, Fitness, and trade-level metrics.
- Portfolio risk engine: Monte Carlo optimization with empirical and EVT-style tail risk reporting.
- Multi-market flexibility: supports US tickers and Yahoo-formatted international tickers (for example `.DE`, `.MC`).

## Backtesting

- Final equity and total profit vs buy-and-hold benchmark.
- Risk-adjusted metrics (Sharpe, Sortino, Calmar).
- Drawdown and turnover-based fitness diagnostics.
- Optional ML classification metrics when applicable (F1, Accuracy, Precision, Recall).

## Repository Layout

```text
alpaca-quant-trading/
|-- main/
|   |-- backtest.py
|   |-- deploy.py
|   |-- rebalance_portfolio.py
|   |-- run_monte_carlo.py
|   |-- train_clstm_ppo.py
|   `-- backtest_clstm_ppo.py
|-- src/
|   |-- backtesting/
|   |-- data/
|   |-- env/
|   |-- execution/
|   |-- strategies/
|   `-- utils/
|-- tests/
|-- models/
|-- requirements.txt
`-- README.md
```

## Quick Start

```bash
git clone https://github.com/sbalta01/alpaca-quant-trading.git
cd alpaca-quant-trading
python -m venv .venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set credentials in `.env` for Alpaca-enabled workflows:

```env
API_KEY=...
API_SECRET=...
PAPER=True
```

## Typical Workflows

Run a single-strategy backtest:

```bash
python main/backtest.py
```

Run live/paper strategy execution:

```bash
python main/deploy.py
```

Run portfolio risk simulation and sizing:

```bash
python main/run_monte_carlo.py
```

Run portfolio rebalance job:

```bash
python main/rebalance_portfolio.py
```

## Notes

- This repository is for research and engineering purposes; it is not investment advice.
- Live trading requires careful validation, risk limits, and broker/API configuration before production use.
