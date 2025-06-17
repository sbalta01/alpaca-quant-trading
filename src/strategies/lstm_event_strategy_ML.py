# src/strategies/lstm_event_strategy.py

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, recall_score, roc_auc_score

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from torch.optim import Adam

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pykalman import KalmanFilter

from src.strategies.base_strategy import Strategy
from src.utils.tools import sma, ema, rsi

# ─── ARIMA/GARCH/Kalman ────────────────────────────────────────────────────────────────────
class ARIMAGARCHKalmanTransformer(BaseEstimator, TransformerMixin):
    """Add ARIMA residuals, GARCH vol forecast & Kalman trend to your DataFrame."""
    def __init__(self, arima_order=(1,0,1), garch_p=1, garch_q=1):
        self.arima_order = arima_order
        self.garch_p = garch_p
        self.garch_q = garch_q

    def fit(self, X, y=None):
        df = X.copy()
        ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        self.ar_model_ = ARIMA(ret, order=self.arima_order).fit(method_kwargs={'disp': False})
        self.garch_model_ = arch_model(ret * 100, p=self.garch_p, q=self.garch_q).fit(disp='off')
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
        self.kf_em_ = kf.em(ret.values)
        self.kf_trend_ = self.kf_em_.filter(ret.values)[0].flatten()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['arima_resid'] = self.ar_model_.resid
        fcast = self.garch_model_.forecast(horizon=1).variance.iloc[-1, 0]
        df['garch_vol'] = np.sqrt(fcast) / 100.0
        df['kf_trend'] = self.kf_trend_
        return df


# ─── TechnicalTransformer unchanged except dropna removal ──────────────────────────
class TechnicalTransformer(BaseEstimator, TransformerMixin):
    """Add a few technical indicators via `src.utils.indicators`."""
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df['sma5']  = sma(df['close'],  5)
        df['ema10'] = ema(df['close'], 10)
        df['rsi14'] = rsi(df['close'], 14)
        return df


# ─── LSTM module unchanged ───────────────────────────────────────────────────────
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
        out, _ = self.lstm(X)
        h = out[:, -1, :]
        return self.act(self.fc(self.drop(h))).squeeze()

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
        arima_order = (1,0,1),
        garch_p: int = 1,
        garch_q: int = 1,
        cv_splits: int = 5,
        rfecv_step: float = 0.1,
        lstm_hidden: int = 32,
        lstm_dropout: float = 0.2,
        random_state: int = 42
    ):
        self.horizon   = horizon
        self.threshold = threshold
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.train_frac = train_frac

        self.recall_scorer = make_scorer(recall_score)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net = NeuralNetClassifier(
            module              = LSTMClassifierModule,
            module__n_features  = None,
            module__hidden_size = lstm_hidden,
            module__dropout     = lstm_dropout,
            max_epochs          = 10,
            lr                  = 1e-3,
            optimizer           = Adam,
            criterion           = nn.BCELoss,
            batch_size          = 32,
            train_split         = None,
            iterator_train__shuffle = False,
            verbose             = 0,
            device              = device
        )

        # Precompute expensive features once, then use simpler pipeline
        self.feature_transform = Pipeline([ 
            ('arima_garch_kf', ARIMAGARCHKalmanTransformer(
                arima_order=arima_order,
                garch_p=garch_p, garch_q=garch_q
            )),
            ('tech', TechnicalTransformer()),
            ('scale', StandardScaler())
        ])

        self.pipeline = Pipeline([
            ('select', RFECV(
                estimator=net,
                step=rfecv_step,
                cv=TimeSeriesSplit(n_splits=cv_splits),
                scoring=self.recall_scorer,
                n_jobs=-1
            )),
            ('clf', net)
        ])

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        target = np.log(df['close'].shift(-self.horizon) / df['close'])
        df['event'] = (target > self.threshold).astype(int)

        feat = df.copy().dropna()
        X_feat = self.feature_transform.fit_transform(feat)
        y_feat = feat['event'].values

        X_seqs, y_labels, times = [], [], []
        for i in range(len(X_feat) - self.horizon):
            win = X_feat[i : i + self.horizon]
            X_seqs.append(win)
            y_labels.append(y_feat[i + self.horizon])
            times.append(feat.index[i + self.horizon])

        X = np.stack(X_seqs)
        y = np.array(y_labels)

        split_i = int(len(X) * self.train_frac)
        X_train, y_train = X[:split_i], y[:split_i]
        X_test,  y_test  = X[split_i:], y[split_i:]
        times_train, times_test = times[:split_i], times[split_i:]

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions={
                'select__step': [0.05, 0.1, 0.2],
                'clf__max_epochs': [5, 10, 20],
                'clf__module__hidden_size': [16, 32, 64],
                'clf__module__dropout': [0.1, 0.2, 0.3]
            },
            cv=tscv,
            scoring=self.recall_scorer,
            n_iter=10,
            n_jobs=1,
            random_state=self.random_state
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:,1]

        # 5) Final predictions on test history
        rec_train = recall_score(y_train, best_pipe.predict(X_train))
        rec_test  = recall_score(y_test,  y_pred)                     
        auc_test  = roc_auc_score(y_test, y_prob)
        print(f"[{self.name}] Recall. Train: {rec_train:.3f}. Test: {rec_test:.3f}")
        print(f"[{self.name}] ROC AUC (Test): {auc_test:.3f}")

        train_acc = search.score(X_train, y_train)
        test_acc  = search.score(X_test,  y_test)
        print(f"Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

        # 6) Build a signal series
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