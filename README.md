# Quantitative Trading Framework

A modular Python framework for **developing, backtesting, and deploying systematic trading strategies**.  
Built for experimentation with machine learning, technical indicators, and production-ready execution pipelines.

Supports:

- **LSTM-based event forecasting (Skorch / PyTorch)**
- **XGBoost & ensemble models**
- **Rule-based technical strategies (MACD, trend filters, RSI, ADX, SMA/EMA)**
- **Backtesting, evaluation, and Alpaca live deployment**

Designed for **fast iteration, reproducibility, and live integration**.

---

## Repository Structure

```

alpaca-quant-trading/
├── main/
│   ├── backtest.py                # Full backtest engine
│   ├── deploy.py                  # Live deployment loop
│   ├── rebalance_portfolio.py     # Portfolio optimizer
│   ├── visualize_stock_data.py    # OHLC + indicators plotter
│   └── get_fundamentals.py        # API-based fundamentals
├── src/
│   ├── strategies/
│   │   ├── base_strategy.py          # Abstract Strategy API
│   │   ├── lstm_event_strategy.py    # LSTM binary classifier
│   │   ├── lstm_regression.py        # LSTM regression forecaster
│   │   ├── xgboost_regression.py     # Gradient boosting regression
│   │   └── macd_strategy.py          # Rule-based MACD strategy
│   ├── utils/
│   │   ├── tools.py                  # Feature engineering (SMA, EMA, RSI…)
│   │   └── metrics.py                # Sharpe, drawdown, hit-rate
│   └── ...
├── requirements_frozen.txt
└── README.md

````

---

## Installation

```bash
git clone https://github.com/your-org/quant-trading.git
cd quant-trading

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

### GPU Support

```bash
pip install torch
```

### For Live Trading

```bash
pip install alpaca-py
```

---

## Strategy Architecture

All strategies inherit from:

```
src/strategies/base_strategy.py
```

Required method:

```python
generate_signals(data: pd.DataFrame) -> pd.DataFrame
```

Returned columns include:

| Column              | Description           |
| ------------------- | --------------------- |
| `position`          | 1 = long, 0 = flat    |
| `signal`            | entry/exit markers    |
| `y_pred` / `y_prob` | model outputs         |
| `features...`       | engineered indicators |

Optional lifecycle methods:

```python
fit_and_save(data, path)
load(path)
predict_next(recent_data)
```

This enables **offline training + live inference**.

---

## Backtesting

Example run:

```bash
python main/backtest.py
```

Features:

* Sharpe / Sortino / MDD scoring
* Trade logs + signal overlay plots
* Reproducible train-val-test splits

Outputs:

* Equity curve
* Performance metrics
* Prediction diagnostics

---

## Live Deployment

Run in paper trading mode:

```bash
python main/deploy.py
```

Execution loop includes:

* Live data polling
* Incremental prediction
* Risk-aware sizing & order routing

Real-market mode:

```bash
export API_KEY=...
export API_SECRET=...
python main/deploy.py
```

---

## Feature Engineering

Key helpers (`src/utils/tools.py`):

* SMA / EMA / WMA
* RSI / ADX / CCI / MACD
* Volatility measures & rolling windows
* Event-driven features for ML

Easily extended to:

* Fundamentals
* Options signals
* Alternative data (news, sentiment)

---

## Risk Management

Built-in protections:

* Stop-loss / take-profit
* Max exposure constraints
* Volatility-aware sizing
* Drawdown ceilings