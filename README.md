<!-- README.md -->

# Quant Trading Framework

A lightweight, extensible Python framework for strategy development, backtesting, and live execution.

## 📂 Project Structure
    alpaca-quant-trading/
    ├── src/
    │ ├── config.py
    │ ├── data/
    │ ├── strategies/
    │ │ ├── init.py
    │ │ ├── base_strategy.py
    │ │ ├── moving_average.py
    │ │ └── rsi.py
    │ ├── backtesting/
    │ ├── execution/
    │ └── utils/
    ├── examples/
    │ ├── backtest_moving_avg.py
    │ └── live_run_moving_avg.py
    ├── tests/
    ├── .gitignore
    ├── requirements.txt
    └── README.md



## ⚙️ Installation

1. **Clone** this repo and `cd` into it:
   ```bash
   git clone https://github.com/sbalta01/alpaca-quant-trading
   cd alpaca-quant-trading
2. Virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate    # Windows: venv\Scripts\activate
3. Install dependencies
    ```bash
    pip install -r requirements.txt
