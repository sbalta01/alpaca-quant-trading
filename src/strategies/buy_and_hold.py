# src/strategies/buy_and_hold.py

import pandas as pd
from src.strategies.base_strategy import Strategy

class BuyAndHoldStrategy(Strategy):
    """
    Control group strategy: buy all-in at the first bar, sell everything at the last bar.
    Emits:
      - signal = +1 on first timestamp
      - signal = -1 on last timestamp
      - signal =  0 elsewhere
    The BacktestEngine will then build the position column for share counts.
    """
    name = "BuyAndHold"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Generate raw positions
        df['position'] = 1.0
        df.iloc[0, df.columns.get_loc("position")] = 0.0
        df.iloc[-1, df.columns.get_loc("position")] = 0.0

        # Generate trading orders: +1 for a buy, -1 for a sell
        df['signal'] = df['position'].diff().fillna(0)

        return df
    
    # def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    #     df = data.copy()
    #     # initialize
    #     df["signal"] = 0.0
    #     # buy at first bar
    #     df.iloc[0, df.columns.get_loc("signal")] = 1.0
    #     # sell at last bar
    #     df.iloc[-1, df.columns.get_loc("signal")] = -1.0
    #     return df