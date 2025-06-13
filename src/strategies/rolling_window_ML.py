# src/strategies/rolling_window_ML.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from src.strategies.base_strategy import Strategy
from src.utils.tools import sma, ema, rsi

class RollingWindowStrategy(Strategy):
    """
    Rolling-window ML strategy:
      - Features: lagged returns, momentum, volatility, RSI, MACD, OBV
      - Model: GradientBoostingClassifier inside a Pipeline
      - Rolling retrain: for each bar t in test set, train on the previous `window` bars
      - Predict next-bar direction → generate +1/-1 signals
    """
    name = "Rolling_Window"

    def __init__(self,
                 train_window: int = 252,
                 retrain_every: int = 5,
                 **gb_kwargs):
        """
        Parameters
        ----------
        train_window  : lookback size (bars) for rolling training
        retrain_every : retrain model every N bars to save time
        gb_kwargs     : passed to GradientBoostingClassifier
        """
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.gb_kwargs = gb_kwargs

        # build pipeline
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(**self.gb_kwargs))
        ])

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns to price DataFrame."""
        df = df.copy()
        df["ret1"] = df["close"].pct_change(1)
        df["ret5"] = df["close"].pct_change(5)
        df["mom10"] = df["close"].pct_change(10)
        df["vol20"] = df["close"].rolling(20).std() / df["close"].rolling(20).mean()
        df["rsi14"] = rsi(df["close"], 14)
        # MACD: EMA(12) - EMA(26)
        df["ema12"] = ema(df["close"], 12)
        df["ema26"] = ema(df["close"], 26)
        df["macd"]  = df["ema12"] - df["ema26"]
        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df_feat = self._compute_features(df)

        # Prepare storage
        signals = pd.Series(0.0, index=df_feat.index)

        last_retrain = None
        model = None

        # Loop through test points: start after initial window
        for i in range(self.train_window, len(df_feat) - 1):
            if last_retrain is None or (i - last_retrain) >= self.retrain_every:
                # retrain model on [i - train_window : i)
                train_slice = df_feat.iloc[i - self.train_window : i]
                X_train = train_slice.drop(columns=["open","high","low","close","volume"])
                # label: next bar up/down
                y_train = (train_slice["close"].shift(-1).loc[train_slice.index] 
                           > train_slice["close"]).astype(int)
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("gb", GradientBoostingClassifier(**self.gb_kwargs))
                ])
                model.fit(X_train, y_train)
                last_retrain = i

            # predict on bar i
            X_pred = df_feat.drop(columns=["open","high","low","close","volume"]).iloc[[i]]
            pred = model.predict(X_pred)[0]  # 1=up, 0=down
            ts_next = df_feat.index[i + 1]

            # signal at next bar
            if pred == 1:
                signals.at[ts_next] = 1.0
            else:
                signals.at[ts_next] = -1.0

        # assemble results
        out = df.copy()
        out["signal"] = signals.reindex(out.index).fillna(0.0)
        return out