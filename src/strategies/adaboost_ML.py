# src/strategies/adaboost_ML.py

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, r2_score

from src.strategies.base_strategy import Strategy
from src.utils.indicators import sma, ema, rsi

class AdaBoostStrategy(Strategy):
    """
    Predict the sign of ΔMA(d) = MA(d)_{t+1} - MA(d)_t using AdaBoost + GridSearchCV.
    Features: 32 from your table (price, volume, MA, EMA, RSI, OBV, ROC, MACD, Stoch, CCI).
    Target: sign of ΔMA(d) for d in {5,10,20}.
    """
    name = "AdaBoost"
    train_val_frac: float
    val_ratio: float

    def __init__(
        self,
        d: int = 5,
        train_val_frac: float = 0.7,
        val_ratio: float = 0.25,
        cv_splits: int = 5,
        param_grid: Dict[str, Any] = None,
        random_state: int = 42,
        ratio_outliers:float = 1.5
    ):
        """
        Parameters
        ----------
        d          : int
            MA window to predict (5, 10, or 20).
        train_val_frac : float
            Fraction of data to train on.
        cv_splits  : int
            Number of folds for time-series CV.
        param_grid : dict
            Grid for GridSearchCV. Defaults to
            {'clf__n_estimators':[50,100], 'clf__learning_rate':[0.5,1.0]}.
        """
        if d not in (5, 10, 20):
            raise ValueError("d must be one of 5, 10, 20")
        self.d = d
        self.train_val_frac = train_val_frac
        self.val_ratio = val_ratio
        self.cv_splits = cv_splits
        self.param_grid = param_grid or {
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.5, 1.0]
        }
        self.random_state = random_state
        self.ratio_outliers = ratio_outliers

        # Pipeline: scaler → RFE(Ridge) → AdaBoost
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('rfe', RFE(Ridge(random_state=self.random_state), n_features_to_select=32)),
            ('clf', AdaBoostClassifier(random_state=self.random_state))
        ])

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where any feature is outside ratio*IQR."""
        clean = df.copy()
        for col in df.columns:
            q1 = clean[col].quantile(0.25)
            q3 = clean[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - self.ratio_outliers * iqr, q3 + self.ratio_outliers * iqr
            clean = clean[(clean[col] >= lo) & (clean[col] <= hi)]
        return clean

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

        # 2) Remove outliers
        feat = self._remove_outliers(feat)

        # 3) Split train / val / test (no shuffle)
        X = feat.drop(columns=['open','high','low','close','volume','target'])
        y = feat['target']

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, train_size=self.train_val_frac, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_ratio, shuffle=False
        )

        # 4) Nested CV Grid Search on train+val
        X_tune = pd.concat([X_train, X_val])
        y_tune = pd.concat([y_train, y_val])
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        gs = GridSearchCV(
            self.pipeline,
            param_grid=self.param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        gs.fit(X_tune, y_tune)
        best = gs.best_estimator_

        # 5) Evaluate on test
        y_pred = best.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)

        # 6) Generate signals: only in test period
        signals = pd.Series(0.0, index=feat.index)
        idxs = list(X_test.index)
        prev_pos = 0
        for idx, pred in zip(idxs, y_pred):
            if pred == 1 and prev_pos == 0:
                signals.at[idx] = 1.0
                prev_pos = 1
            elif pred == -1 and prev_pos == 1:
                signals.at[idx] = -1.0
                prev_pos = 0
            else:
                signals.at[idx] = 0.0

        # 7) Merge back to full
        out = df.copy()
        out['signal'] = signals.reindex(df.index).fillna(0.0)
        return out