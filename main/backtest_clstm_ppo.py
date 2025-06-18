import numpy as np
import pandas as pd
from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from main.train_clstm_ppo import add_technicals
from src.data.data_loader import attach_factors
from src.env.trading_env import TradingEnv

import matplotlib.pyplot as plt


def backtest_rl(
    model_path: str,
    symbols,
    start: datetime,
    end:   datetime,
    timeframe: TimeFrame,
    tech_cols,
    macro_cols,
    initial_cash=100,
    transaction_cost=0.001,
    turbulence_col="turbulence",
    turbulence_pct=0.9
):
    
    if timeframe.unit_value == TimeFrameUnit.Month:
        print('Timeframe set to Month')
        timeframe_yahoo = '1mo'
    elif timeframe.unit_value == TimeFrameUnit.Week:
        print('Timeframe set to Week')
        timeframe_yahoo = '1wk'
    elif timeframe.unit_value == TimeFrameUnit.Day:
        print('Timeframe set to Day')
        timeframe_yahoo = '1d'
    elif timeframe.unit_value == TimeFrameUnit.Hour:
        print('Timeframe set to Hour')
        timeframe_yahoo = '1h'
    elif timeframe.unit_value == TimeFrameUnit.Minute:
        print('Timeframe set to Minute')
        timeframe_yahoo = '1m'
    try:
        from src.data.data_loader import fetch_alpaca_data as fetch_data
        jkandc
        print('USING ALPACA DATA')
    except:
        from src.data.data_loader import fetch_yahoo_data as fetch_data
        print('USING YAHOO DATA')
        timeframe = timeframe_yahoo
        feed = None

    # 1) Fetch and prepare data
    df = fetch_data(symbols, start, end, timeframe=timeframe)
    df = attach_factors(df, timeframe=timeframe_yahoo)
    df = add_technicals(df)

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

    # 3) Load trained RL model
    model = RecurrentPPO.load(model_path)

    env = DummyVecEnv([lambda: TradingEnv(**env_kwargs)])
    # seq_len = model.policy.features_extractor_class.seq_len
    seq_len = 30
    env = VecFrameStack(env, n_stack=seq_len)


    # 4) Roll out one episode to collect actions & equity
    obs = env.reset()
    done = False

    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        done = done[0]
        info = info[0]
        # record timestamp, portfolio value
        ts   = df.index.get_level_values("timestamp")[env.envs[0].t - 1]
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

    # 7) Compute performance metrics
    perf_summary = {}
    for symbol in df_out.index.get_level_values(0).unique():
        data = df_out.xs(symbol, level="symbol")

        total_return = data["equity"].iloc[-1] / data["equity"].iloc[0] - 1
        periods_per_year = 252 if timeframe_yahoo == "1d" else 12
        cagr = (1 + total_return) ** (periods_per_year / len(data)) - 1

        sharpe = (
            data["returns"].mean() / data["returns"].std()
        ) * np.sqrt(periods_per_year) if data["returns"].std() > 0 else 0.0

        roll_max = data["equity"].cummax()
        dd = (data["equity"] - roll_max) / roll_max
        max_dd = dd.min()

        perf_summary[symbol] = {
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd
        }

    print("\n=== Performance Summary ===")
    for sym, metrics in perf_summary.items():
        print(f"{sym}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2%}")

    # 8) Plot equity curves
    for symbol in df_out.index.get_level_values(0).unique():
        data = df_out.xs(symbol, level="symbol")
        data["equity"].plot(title=f"{symbol} - Equity Curve", figsize=(10, 4))
        plt.ylabel("Equity ($)")
        plt.xlabel("Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 9) Plot signals (1 = position change)
    for symbol in df_out.index.get_level_values(0).unique():
        data = df_out.xs(symbol, level="symbol")
        data["signal"] = 0
        data["signal"].iloc[1:] = (data["equity"].diff().iloc[1:] != 0).astype(int)
        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(data.index, data["equity"], label="Equity", color="blue")
        ax1.set_ylabel("Equity", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_title(f"{symbol} - Signals on Equity Curve")

        # mark signals
        signals = data[data["signal"] == 1]
        ax1.scatter(signals.index, signals["equity"], marker="o", color="red", label="Signal")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_out

if __name__ == "__main__":
    symbols   = ["AAPL","MSFT","GOOG"]
    start     = datetime(2022,1,1)
    end       = datetime(2025,6,1)
    timeframe = TimeFrame.Day
    tech_cols  = ["close","macd","rsi","cci","adx"]
    macro_cols = ["VIX","EURUSD", "DFF"]

    results = backtest_rl(
        model_path      = "clstm_ppo_stock.zip",
        symbols         = symbols,
        start           = start,
        end             = end,
        timeframe       = timeframe,
        tech_cols       = tech_cols,
        macro_cols      = macro_cols
    )

    print(results.groupby(level="symbol")["equity"].last())