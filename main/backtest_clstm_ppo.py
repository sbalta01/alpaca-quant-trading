import numpy as np
import pandas as pd
from datetime import datetime
from alpaca.data.timeframe import TimeFrame

from stable_baselines3 import PPO

from src.data.data_loader   import fetch_alpaca_data
from src.env.trading_env    import TradingEnv
from src.execution.backtest_executor import run_backtest_strategy

def backtest_rl(
    model_path: str,
    symbols,
    start: datetime,
    end:   datetime,
    tech_cols,
    macro_cols,
    initial_cash=1e6,
    transaction_cost=0.001,
    turbulence_col="turbulence",
    turbulence_pct=0.9
):
    # 1) Fetch and prepare data
    df = fetch_alpaca_data(symbols, start, end, timeframe=TimeFrame.Day)
    # assume df already has columns ['close'] + tech_cols + macro_cols + turbulence_col

    # 2) Build the env
    env_kwargs = dict(
        price_df          = df,
        tech_cols         = tech_cols,
        macro_cols        = macro_cols,
        initial_cash      = initial_cash,
        transaction_cost  = transaction_cost,
        turbulence_col    = turbulence_col,
        turbulence_percentile = turbulence_pct
    )
    env = TradingEnv(**env_kwargs)

    # 3) Load trained RL model
    model = PPO.load(model_path)

    # 4) Roll out one episode to collect actions & equity
    obs, _ = env.reset()
    done = False

    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        # record timestamp, portfolio value
        ts   = df.index.get_level_values("timestamp")[env.t - 1]
        pv   = info["portfolio_value"]
        records.append((symbols, ts, pv))

    # 5) Build a results DataFrame matching backtester format
    # Here we assume single symbol; for multiple, you'd unstack appropriately
    idx = pd.DatetimeIndex([r[1] for r in records], name="timestamp")
    equity = pd.Series([r[2] for r in records], index=idx, name="equity")

    # assemble into a MultiIndex if multiple symbols
    if isinstance(symbols, (list, tuple)):
        mi = pd.MultiIndex.from_product([symbols, equity.index], names=["symbol","timestamp"])
        equity = pd.Series(
            np.repeat(equity.values[None,:], len(symbols), axis=0).ravel(),
            index=mi, name="equity"
        )

    # 6) Compute P&L, returns
    df_out = equity.to_frame().sort_index()
    df_out["returns"] = df_out.groupby(level="symbol")["equity"].pct_change().fillna(0.0)

    # 7) Print performance (reuse your executor’s metrics)
    # You could call run_backtest_strategy, but here we already have the time series
    from src.backtesting.backtester import BacktestEngine
    # We need dummy signals DataFrame for BacktestEngine API:
    # Insert a +1 signal whenever equity steps up from zero position.
    df_out["signal"] = np.nan
    df_out["signal"].iloc[0] = 1.0
    df_out["signal"].iloc[1:] = (df_out["equity"].values[1:] != df_out["equity"].values[:-1]).astype(float)
    # position & cash will be reconstructed by engine, but we can skip to performance
    engine = BacktestEngine(strategy=None, data=df_out, initial_cash_per_stock=initial_cash)
    perf = engine.performance(df_out)
    print("RL Backtest Performance:", perf)

    return df_out

if __name__ == "__main__":
    symbols   = ["AAPL","MSFT","GOOG"]
    start     = datetime(2018,1,1)
    end       = datetime(2022,1,1)
    tech_cols = ["macd","rsi","cci","adx"]
    macro_cols= ["VIX","EURUSD"]

    results = backtest_rl(
        model_path      = "clstm_ppo_stock.zip",
        symbols         = symbols,
        start           = start,
        end             = end,
        tech_cols       = tech_cols,
        macro_cols      = macro_cols
    )

    print(results.groupby(level="symbol")["equity"].last())