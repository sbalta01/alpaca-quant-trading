<!-- README.md -->

# Quantitative Trading Framework

A unified framework for developing, backtesting, and deploying systematic trading strategies in Python.  
Leverages PyTorch (via Skorch), scikit‑learn, XGBoost, Optuna, and pandas to build feature‑rich LSTM classifiers/regressors, ensemble learners, and rule‑based technical strategies.

---

<<<<<<< Updated upstream
<<<<<<< Updated upstream
## 📂 Repository Structure
All code to be deployed, either locally or through Alpaca, is located inside main/ folder.
src/ contains all modules: strategies, utils, backtesting engine, execution engine, data fetching, etc.

=======
## Repository Structure
>>>>>>> Stashed changes
=======
## Repository Structure
>>>>>>> Stashed changes
    alpaca-quant-trading/
    ├── main/
    │ ├── backtest.py
    │ ├── deploy.py
    │ ├── rebalance_portfolio.py
    │ ├── visualize_stock_data.py
    │ └── get_fundamentals.py
    ├── src/
    │ ├── strategies/
    │ │ ├── base_strategy.py # Abstract Strategy interface
    │ │ ├── lstm_event_strategy.py # “Big‑move” LSTM classifier strategy
    │ │ ├── lstm_regression.py # LSTM regressor strategy
    │ │ ├── xgboost_regression.py # XGBoost‑based regression strategy
    │ │ └── macd_strategy.py # Rule‑based MACD strategy
    │ ├── utils/
    │ │ ├── tools.py # Feature‑engineering helpers (sma, ema, rsi, adx…)
    │ │ └── metrics.py # Custom scoring (e.g. Sharpe scorer)
    │ └── … # Data loaders, backtest engines, etc.
    ├── requirements_frozen.txt # Pinned Python dependencies
    └── README.md 

---
## Installation

```bash
git clone https://github.com/your‑org/quant‑trading.git
cd quant‑trading
python3 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

```
- For GPU support: install torch with CUDA
- For live deployment, ensure ```joblib``` and market‑data API clients (e.g. ```alpaca-py```) are installed

---
## Strategy interface
All strategies inherit from Strategy in src/strategies/base_strategy.py and implement:

- generate_signals(data: pd.DataFrame) → pd.DataFrame

    - Compute features, fits/models if needed, returns a DataFrame with:

        - position (0 = flat, 1 = long)

        - signal (entry/exit markers)

        - diagnostic columns (e.g. y_pred, y_prob, indicators)

- (Optional) fit_and_save(data, path) & load(path)

- (Optional) predict_next(recent_data) for live one‑step‑ahead forecasts.
