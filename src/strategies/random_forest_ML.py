# src/strategies/random_forest.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.strategies.base_strategy import Strategy
from src.utils.indicators import sma, rsi

class RandomForestStrategy(Strategy):
    """
    ML‐based strategy based on a random forest classifier:
      - Features: lagged returns, SMA deviation, RSI
      - Model: RandomForestClassifier
      - Train on the first `train_frac` of data, predict on the remainder
      - Generates:
         signal = +1 if model predicts positive next‐day return AND not currently long
         signal = -1 if model predicts negative return AND currently long
         signal = 0 otherwise
    """
    name = "Random_Forest"

    def __init__(self, train_frac: float = 0.7, 
                 n_estimators: int = 100, random_state: int = 42):
        self.train_frac = train_frac
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # 1) Compute features
        df["ret1"] = df["close"].pct_change(1)
        df["ret2"] = df["close"].pct_change(2)
        df["sma20"] = sma(df["close"], 20)
        df["p_sma_dev"] = (df["close"] - df["sma20"]) / df["sma20"]
        df["rsi14"] = rsi(df["close"], 14)
        df = df.dropna()

        # 2) Labels: 1 if next-day return > 0, else 0
        df["label"] = (df["close"].shift(-1).pct_change(fill_method=None) > 0).astype(int)
        df = df.dropna()

        # 3) Split train / test
        split = int(len(df) * self.train_frac)
        train = df.iloc[:split]
        test  = df.iloc[split:]

        features = ["ret1", "ret2", "p_sma_dev", "rsi14"]
        X_train = train[features]
        y_train = train["label"]
        X_test  = test[features]

        # 4) Train model & predict
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        # 5) Build signals series aligned with test index
        signals = pd.Series(0, index=df.index)
        # we only generate signals in test set
        prev_pos = 0
        for idx, pred in zip(test.index, preds):
            if pred == 1 and prev_pos == 0:
                signals.at[idx] = 1   # enter long
                prev_pos = 1
            elif pred == 0 and prev_pos == 1:
                signals.at[idx] = -1  # exit long
                prev_pos = 0
            else:
                signals.at[idx] = 0

        df["signal"] = signals
        return df[["close","signal"]]
