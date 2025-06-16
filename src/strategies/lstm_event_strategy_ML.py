# src/strategies/lstm_event_strategy.py

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, recall_score, r2_score

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from torch.optim import Adam

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pykalman import KalmanFilter

from src.strategies.base_strategy import Strategy
from src.utils.tools import sma, ema, rsi

# ──────────────────────────────────────────────────────────────────────────────

class ARIMAGARCHKalmanTransformer(BaseEstimator, TransformerMixin):
    """Add ARIMA residuals, GARCH vol forecast & Kalman trend to your DataFrame."""
    def __init__(self, arima_order=(1,0,1), garch_p=1, garch_q=1):
        self.arima_order = arima_order
        self.garch_p = garch_p
        self.garch_q = garch_q

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # 1) log returns
        ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        # 2) ARIMA residuals
        ar = ARIMA(ret, order=self.arima_order).fit(disp=False)
        df['arima_resid'] = ar.resid
        # 3) GARCH( p, q ) vol forecast (in same units as ret)
        g = arch_model(ret * 100, p=self.garch_p, q=self.garch_q).fit(disp='off')
        fcast = g.forecast(horizon=1).variance.iloc[-1, 0]
        df['garch_vol'] = np.sqrt(fcast) / 100.0
        # 4) simple Kalman trend
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
        state_means, _ = kf.em(ret.values).filter(ret.values)
        df['kf_trend'] = state_means.flatten()
        return df.dropna()

# ──────────────────────────────────────────────────────────────────────────────

class TechnicalTransformer(BaseEstimator, TransformerMixin):
    """Add a few technical indicators via `src.utils.indicators`."""
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df['sma5']  = sma(df['close'],  5)
        df['ema10'] = ema(df['close'], 10)
        df['rsi14'] = rsi(df['close'], 14)
        return df.dropna()

# ──────────────────────────────────────────────────────────────────────────────

class LSTMClassifierModule(nn.Module):
    """Simple 1-layer LSTM → Dropout → Dense(sigmoid) classifier."""
    def __init__(self, n_features, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)
        self.act  = nn.Sigmoid()

    def forward(self, X):
        # X: (batch, seq_len, features)
        out, _ = self.lstm(X)
        h = out[:, -1, :]           # last time-step
        h = self.drop(h)
        return self.act(self.fc(h)).squeeze()

# ──────────────────────────────────────────────────────────────────────────────

class LSTMEventStrategy(Strategy):
    """
    Predict and signal “big-move” days: next-N-day log-return > threshold.
    Uses ARIMA/GARCH/Kalman + technicals + (optional) macro columns,
    a PyTorch LSTM classifier via skorch, RFECV feature-selection,
    and forward-rolling CV for hyperparameter search.
    """
    name = "LSTMEvent"
    multi_symbol = False

    def __init__(
        self,
        horizon: int  = 5,
        threshold: float = 0.02,
        train_frac = 0.7,
        arima_order=(1,0,1),
        garch_p: int = 1,
        garch_q: int = 1,
        cv_splits: int = 5,
        rfecv_step: float = 0.1,
        lstm_hidden: int = 32,
        lstm_dropout: float = 0.2,
        random_state: int = 42
    ):
        self.horizon  = horizon
        self.threshold= threshold
        self.cv_splits= cv_splits
        self.random_state = random_state
        self.train_frac = train_frac

        # recall-scorer for RFECV and CV
        self.recall_scorer = make_scorer(recall_score)

        # skorch LSTM wrapper
        net = NeuralNetClassifier(
            module              = LSTMClassifierModule,
            module__n_features  = None,  # auto-inferred
            module__hidden_size = lstm_hidden,
            module__dropout     = lstm_dropout,
            max_epochs          = 10,
            lr                  = 1e-3,
            optimizer           = Adam,
            criterion           = nn.BCELoss,
            batch_size          = 32,
            train_split         = None,   # we do CV externally
            iterator_train__shuffle = False,
            verbose             = 0,
            device              = 'cpu'
        )

        # Full pipeline
        self.pipeline = Pipeline([
            ('arima_garch_kf', ARIMAGARCHKalmanTransformer(
                arima_order=arima_order,
                garch_p=garch_p, garch_q=garch_q
            )),
            ('tech',       TechnicalTransformer()),
            ('scale',      StandardScaler()),
            ('select',     RFECV(
                estimator=net,
                step=rfecv_step,
                cv=TimeSeriesSplit(n_splits=cv_splits),
                scoring=self.recall_scorer,
                n_jobs=-1
            )),
            ('clf',        net)
        ])

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # 1) Label events: forward log-return over horizon
        target = (np.log(df['close'].shift(-self.horizon) / df['close']))
        df['event'] = (target > self.threshold).astype(int)

        # 2) Build sequences X, y and aligned timestamps
        feat = df.dropna().copy()
        X_seqs, y_labels, times = [], [], []
        for i in range(len(feat) - self.horizon):
            win = feat.iloc[i : i + self.horizon]
            X_seqs.append(win.drop(columns=['event']).values)
            y_labels.append(feat['event'].iloc[i + self.horizon])
            times.append(feat.index[i + self.horizon])

        X = np.stack(X_seqs)      # shape (n_samples, horizon, n_features)
        y = np.array(y_labels)    # shape (n_samples,)

        # 3) Split train / test
        n_samples = len(X)
        split_i   = int(n_samples * self.train_frac)

        X_train, y_train = X[:split_i], y[:split_i]
        X_test,  y_test  = X[split_i:], y[split_i:]
        times_train, times_test       = times[:split_i], times[split_i:]

        # 4) Forward-rolling CV + RandomizedSearch for RFECV+LSTM
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        search = RandomizedSearchCV(
            estimator = self.pipeline,
            param_distributions = {
                # RFECV step size
                'select__step':       [0.05, 0.1, 0.2],
                # LSTM params
                'clf__max_epochs':    [5, 10, 20],
                'clf__module__hidden_size': [16, 32, 64],
                'clf__module__dropout':     [0.1, 0.2, 0.3]
            },
            cv = tscv,
            scoring = self.recall_scorer,
            n_iter  = 10,
            n_jobs  = 1,
            random_state = self.random_state
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        # 5) Final predictions on test history
        y_pred = best_pipe.predict(X_test)
        
        r2_train = r2_score(y_train, best_pipe.predict(X_train))
        r2_test = r2_score(y_test, y_pred)
        print(f"[{self.name}] R² score. Train: {r2_train:.3f}. Test: {r2_test:.3f}")
        
        # Directional accuracy: how often sign(pred) == sign(true)
        sign_test = np.sign(y_test)
        sign_pred = np.sign(y_pred)
        dir_acc = (sign_test == sign_pred).mean()

        print(f"[{self.name}] average directional accuracy (Test set): {dir_acc:.3f}")

        # 6) Build a signal series (only +1 entries; exits via backtest logic)
        positions = pd.Series(np.nan, index=feat.index)
        positions.at[times_train[0]] = 0.0
        y_pred_series = pd.Series(0.0, index=feat.index)
        y_test_series = pd.Series(0.0, index=feat.index)
        test_mask = pd.Series(0.0, index=feat.index)
        position = 0
        days_left = 0
        for t, pred, test in zip(times_test, y_pred, y_test):
            if days_left > 0:
                days_left -= 1
            else:
                if position == 0 and pred == 1:
                    position = 1
                    days_left = self.horizon - 1
                elif position == 1 and pred == 0.0:
                    position = 0 #exit to flat
            positions.at[t] = position
            y_pred_series.at[t] = pred
            y_test_series.at[t] = test
            test_mask.at[t] = 1.0
        print('Max prediction', y_pred_series.max())
        # 7) Merge signals into df
        out = df.copy()
        out["position"] = positions
        out["position"] = out["position"].ffill().bfill().clip(0,1) #clip(0,1) for no short. ffill for missing timestamps in test (because of holidays or outliers removal eg). bfill (afterwards) for position = 0 before test. 
        out['signal'] = out['position'].diff().fillna(0.0).clip(-1,1) #fillna for the first signal
        
        out["y_test"] = y_test_series.reindex(df.index).fillna(0.0)
        out["y_pred"] = y_pred_series.reindex(df.index).fillna(0.0)
        out["test_mask"] = test_mask.reindex(df.index).fillna(0.0)
        return out