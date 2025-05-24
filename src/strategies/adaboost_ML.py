# src/strategies/adaboost_ma_predictor.py

import pandas as pd
import numpy as np
from typing import List
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.strategies.base_strategy import Strategy
from src.utils.indicators import sma, ema, rsi

class AdaBoostStrategy(Strategy):
    """
    Predict the sign of ΔMA(d) = MA(d)_{t+1} - MA(d)_t using AdaBoost + GridSearchCV.
    Features: 32 from your table (price, volume, MA, EMA, RSI, OBV, ROC, MACD, Stoch, CCI).
    Target: sign of ΔMA(d) for d in {5,10,20}.
    """
    name = "AdaBoost"

    def __init__(
        self,
        d: int = 5,
        train_frac: float = 0.7,
        cv_splits: int = 5,
        param_grid: dict = None,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        d          : int
            MA window to predict (5, 10, or 20).
        train_frac : float
            Fraction of data to train on.
        cv_splits  : int
            Number of folds for time-series CV.
        param_grid : dict
            Grid for GridSearchCV. Defaults to
            {'clf__n_estimators':[50,100], 'clf__learning_rate':[0.5,1.0]}.
        """
        if d not in (5,10,20):
            raise ValueError("d must be one of 5, 10, 20")
        self.d = d
        self.train_frac = train_frac
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.param_grid = param_grid or {
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.5, 1.0]
        }

        # build pipeline: scale → AdaBoost
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', AdaBoostClassifier(random_state=self.random_state))
        ])

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 32 features from your table."""
        df = df.copy()
        # Basic price & lag
        df['open'] = df['open']
        df['high'] = df['high']
        df['low']  = df['low']
        df['close'] = df['close']
        df['close_1'] = df['close'].shift(1)
        df['close_inc'] = df['close'] - df['close_1']

        # Volume features
        df['volume'] = df['volume']
        df['volume_1'] = df['volume'].shift(1)
        df['volume_inc'] = df['volume'] - df['volume_1']

        # Moving Averages
        for w in (5,10,20):
            df[f'MA{w}'] = sma(df['close'], w)
            df[f'MA{w}_1'] = df[f'MA{w}'].shift(1)
            df[f'MA{w}_inc'] = df[f'MA{w}'] - df[f'MA{w}_1']

        # EMAs
        for w in (5,10,20):
            df[f'EMA{w}'] = ema(df['close'], w)

        # RSI
        df['RSI'] = rsi(df['close'], 12)

        # OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # ROC
        for w in (5,10,20):
            df[f'ROC{w}'] = df['close'].pct_change(w) * 100

        # MACD (12,26) + Signal(9)
        df['EMA12'] = ema(df['close'], 12)
        df['EMA26'] = ema(df['close'], 26)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACDsignal'] = ema(df['MACD'], 9)
        df['MACDhist']   = df['MACD'] - df['MACDsignal']

        # Stochastic Oscillator %K(3), %D(3)
        low_min  = df['low'].rolling(3).min()
        high_max = df['high'].rolling(3).max()
        df['slowk'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['slowd'] = df['slowk'].rolling(3).mean()

        # CCI(10)
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(10).mean()
        mean_dev = tp.rolling(10).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI'] = (tp - ma_tp) / (0.015 * mean_dev)

        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # 1) Build features & target
        feat = self._compute_features(df)
        # target = sign of ΔMA(d)
        feat['target'] = np.sign(
            feat[f'MA{self.d}'].shift(-1) - feat[f'MA{self.d}']
        )
        feat = feat.dropna()
        feat['target'] = feat['target'].astype(int)

        # 2) Split train/test
        split = int(len(feat) * self.train_frac)
        train = feat.iloc[:split]
        test  = feat.iloc[split:]

        X_train = train.drop(columns=['open','high','low','close','volume','target'])
        y_train = train['target']
        X_test  = test.drop(columns=['open','high','low','close','volume','target'])

        # 3) GridSearchCV with time-series split
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        gs = GridSearchCV(
            self.pipeline,
            param_grid=self.param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        # 4) Predict & build signals
        preds = best.predict(X_test)
        signals = pd.Series(0.0, index=feat.index)
        prev_pos = 0
        for idx, pred in zip(test.index, preds):
            if pred == 1 and prev_pos == 0:
                signals.at[idx] = 1.0
                prev_pos = 1
            elif pred == -1 and prev_pos == 1:
                signals.at[idx] = -1.0
                prev_pos = 0
            else:
                signals.at[idx] = 0.0

        # 5) Merge back to full df
        out = df.copy()
        out['signal'] = signals.reindex(df.index).fillna(0.0)
        return out