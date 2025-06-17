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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Compute all features _only_ on the given slice
        df = X.copy().reset_index(drop=True)                        
        ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)  

        # ARIMA residuals on this slice
        ar_model = ARIMA(ret, order=self.arima_order)\
                       .fit(method_kwargs={'disp': False})          
        df['arima_resid'] = ar_model.resid                         

        # GARCH volatility forecast on this slice
        garch = arch_model(ret * 100, p=self.garch_p, q=self.garch_q)\
                     .fit(disp='off')                              
        fcast = garch.forecast(horizon=1).variance.iloc[-1, 0]     
        df['garch_vol'] = np.sqrt(fcast) / 100.0                   

        # Kalman trend on this slice
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])  
        kf_em = kf.em(ret.values)                                              
        trend = kf_em.filter(ret.values)[0].flatten()                           
        df['kf_trend'] = trend                                                  

        return df.dropna()                            


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
        return df.dropna()


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
        # self.act  = nn.Sigmoid()

    def forward(self, X):
        out, _ = self.lstm(X)
        h = out[:, -1, :]
        h = self.drop(h)
        # return self.act(self.fc(h)).squeeze()
        return self.fc(h).squeeze(-1)

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

        self.net = NeuralNetClassifier(
            module              = LSTMClassifierModule,
            module__hidden_size = lstm_hidden,
            module__dropout     = lstm_dropout,
            max_epochs          = 10,
            lr                  = 1e-3,
            optimizer           = Adam,
            criterion           = nn.BCEWithLogitsLoss,
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
            ('clf', self.net)
        ])

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        target = np.log(df['close'].shift(-self.horizon) / df['close'])
        df['event'] = (target > self.threshold).astype(int)

        feat = df.copy().dropna()
        split_i = int(len(feat) * self.train_frac)                       
        feat_train = feat.iloc[:split_i].copy()                          
        feat_test  = feat.iloc[split_i:].copy()                          

        self.feature_transform.fit(feat_train.drop(columns=['event']))                           
        X_train_full = self.feature_transform.transform(feat_train.drop(columns=['event']))
        X_test_full  = self.feature_transform.transform(feat_test.drop(columns=['event']))

        y_train_full = feat_train['event'].values                        
        y_test_full  = feat_test['event'].values                         
        times_train_full = list(feat_train.index)                        
        times_test_full  = list(feat_test.index)                         

        def make_sequences(X_arr, y_arr, times_arr):
            X_seqs, y_labels, times = [], [], []
            for i in range(len(X_arr) - self.horizon):
                X_seqs.append(X_arr[i : i + self.horizon])
                y_labels.append(y_arr[i + self.horizon])
                times.append(times_arr[i + self.horizon])
            return np.stack(X_seqs), np.array(y_labels), times

        X_train, y_train, times_train = make_sequences(
            X_train_full, y_train_full, times_train_full)             
        X_test,  y_test,  times_test  = make_sequences(
            X_test_full,  y_test_full,  times_test_full)              

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        n_feats = X_train.shape[2]
        self.net.set_params(module__n_features=n_feats)

        # Hyperparam search on LSTM
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions={
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
        auc_test  = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC (Test): {auc_test:.3f}")

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
        # 7) Merge signals into df
        out = df.copy()
        out["position"] = positions
        out["position"] = out["position"].ffill().bfill().clip(0,1) #clip(0,1) for no short. ffill for missing timestamps in test (because of holidays or outliers removal eg). bfill (afterwards) for position = 0 before test. 
        out['signal'] = out['position'].diff().fillna(0.0).clip(-1,1) #fillna for the first signal
        
        out["y_test"] = y_test_series.reindex(df.index).fillna(0.0)
        out["y_pred"] = y_pred_series.reindex(df.index).fillna(0.0)
        out["test_mask"] = test_mask.reindex(df.index).fillna(0.0)
        return out