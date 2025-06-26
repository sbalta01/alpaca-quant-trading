# src/strategies/lstm_event_strategy.py

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from torch.optim import Adam

from src.strategies.base_strategy import Strategy
from src.utils.tools import adx, compute_turbulence_single_symbol, sma, ema, rsi

class FeatureAttention(nn.Module):
    """
    Learns a per-feature attention weight vector for each time step.
    Input:  X of shape (batch, seq_len, n_features)
    Output: X_att of same shape, where X_att[:,:,i] = alpha_i · X[:,:,i]
    """
    def __init__(self, n_features: int):
        super().__init__()
        # one scalar score per feature
        self.score = nn.Parameter(torch.zeros(n_features))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, T, F]
        # produce a normalized weight vector α of shape [F]
        alpha = self.softmax(self.score)
        # broadcast to [B, T, F]
        return X * alpha

class LSTMWithFeatureAttention(nn.Module):
    """Simple 1-layer LSTM → Dropout → Dense(sigmoid) classifier with
    feature-level attention before the LSTM.
    """
    def __init__(self, n_features, hidden_size=32, dropout=0.2):
        super().__init__()
        self.attn   = FeatureAttention(n_features)
        self.lstm   = nn.LSTM(input_size=n_features,
                              hidden_size=hidden_size,
                              batch_first=True)
        self.drop   = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, X):
        # 1) apply feature attention
        X = self.attn(X)                   # [B, T, F] → weighted

        # 2) run through the LSTM
        out, _ = self.lstm(X)              # out: [B, T, hidden]

        # 3) classify off the final hidden state
        h = out[:, -1, :]                  # [B, hidden]
        h = self.drop(h)
        return self.output(h).squeeze(-1)

# ──────────────────────────────────────────────────────────────────────────────

class TimeSeriesScaler(TransformerMixin, BaseEstimator):
    """
    Wraps any 2D scaler (e.g. StandardScaler) so it will
    scale each feature across both the sample and time axes.
    Expects X of shape (n_samples, seq_len, n_features).
    """
    def __init__(self, scaler=None):
        # allow passing in a custom scaler like MinMaxScaler if you want
        self.scaler = scaler or StandardScaler()

    def fit(self, X, y=None):
        # X: (n_samples, seq_len, n_features)
        n_s, t, n_f = X.shape
        # flatten time into samples
        flat = X.reshape(-1, n_f)  # shape (n_s * t, n_f)
        self.scaler.fit(flat)
        return self

    def transform(self, X):
        n_s, t, n_f = X.shape
        flat = X.reshape(-1, n_f)
        flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(n_s, t, n_f)

# ──────────────────────────────────────────────────────────────────────────────

class LSTMEventTechnicalStrategy(Strategy):
    """
    Predict and signal “big-move” days: next-N-day log-return > threshold.
    Uses ARIMA/GARCH/Kalman + technicals + (optional) macro columns,
    a PyTorch LSTM classifier via skorch, RFECV feature-selection,
    and forward-rolling CV for hyperparameter search.
    """
    name = "LSTMEventTechnical"
    multi_symbol = False

    def __init__(
        self,
        horizon: int  = 5,
        threshold: float = 0.02,
        train_frac = 0.7,
        cv_splits: int = 5,
        random_state: int = 42
    ):
        self.horizon   = horizon
        self.threshold = np.log(1 + threshold)
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.train_frac = train_frac

        # Reproducibility
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if torch.cuda.is_available() and False: #Disabled
            device = 'cuda'
            print("Current device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            device = 'cpu'
            print("Current device name: ", device)

        self.net = NeuralNetClassifier(
            module              = LSTMWithFeatureAttention,
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

        self.pipeline = Pipeline([
            ('scale', TimeSeriesScaler(StandardScaler())),
            ('clf', self.net)
        ])
        
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = df.index

        # --- Calendar data with cyclical embedding
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek   / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek   / 7)
        df["mo_sin"]  = np.sin(2 * np.pi * (idx.month - 1) / 12)
        df["mo_cos"]  = np.cos(2 * np.pi * (idx.month - 1) / 12)

        # --- Price & lag features ---
        df['open'] = df['open']
        df['high'] = df['high']
        df['low']  = df['low']
        df['close'] = df['close']

        df['logret'] = np.log(df['close'] / df['close'].shift(1))
        df[f'vol{self.horizon}'] = df['logret'].rolling(self.horizon).std()
        df[f'logret{self.horizon}'] = np.log(df['close'] / df['close'].shift(self.horizon))


        # --- Volume features ---
        df['volume'] = df['volume']
        df['volume_1'] = df['volume'].shift(1)
        df['volume_inc'] = df['volume'] - df['volume_1']

        # --- Moving Averages & their diffs ---
        for w in (5,10,20,50):
            df[f'sma{w}']    = sma(df['close'], w)
            df[f'sma{w}_1'] = df[f'sma{w}'].shift(1)
            df[f'sma{w}_inc'] = df[f'sma{w}'] - df[f'sma{w}_1']
            df[f'ema{w}']    = ema(df['close'], w)
            df[f'ema{w}_1'] = df[f'ema{w}'].shift(1)
            df[f'ema{w}_inc'] = df[f'ema{w}'] - df[f'ema{w}_1']

        # --- RSI(14) ---
        df['RSI14'] = rsi(df['close'], 14)

        # --- OBV ---
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # --- ROC ---
        for w in (5,10,20):
            df[f'ROC{w}'] = df['close'].pct_change(w)

        # --- MACD + Signal(9) + hist ---
        df['ema12'] = ema(df['close'], 12)
        df['ema26'] = ema(df['close'], 26)
        df['macd']  = df['ema12'] - df['ema26']
        df['macd_sig']  = ema(df['macd'], 9)
        df['macd_hist'] = df['macd'] - df['macd_sig']

        # --- Stochastic %K(3), %D(3) ---
        low3  = df['low'].rolling(3).min()
        high3 = df['high'].rolling(3).max()
        df['sto_k'] = 100 * (df['close'] - low3) / (high3 - low3)
        df['sto_d'] = df['sto_k'].rolling(3).mean()

        # --- CCI(10) ---
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(10).mean()
        md = tp.rolling(10).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI10'] = (tp - ma_tp) / (0.015 * md)

        # --- ADX(14) ---
        df["adx"] = adx(df["high"], df["low"], df["close"], window=14)

        # --- Ichimoku Cloud ---
        # Conversion line (9), Base line (26), Leading Span A/B (26/52)
        high9  = df['high'].rolling(9).max()
        low9   = df['low'].rolling(9).min()
        df['ichimoku_conv'] = (high9 + low9) / 2
        high26 = df['high'].rolling(26).max()
        low26  = df['low'].rolling(26).min()
        df['ichimoku_base'] = (high26 + low26) / 2
        df['ichimoku_span_a'] = ((df['ichimoku_conv'] + df['ichimoku_base'])/2).shift(26)
        high52 = df['high'].rolling(52).max()
        low52  = df['low'].rolling(52).min()
        df['ichimoku_span_b'] = ((high52 + low52)/2).shift(26)
        df['turbulence'] = compute_turbulence_single_symbol(df, window=252)
        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()       

        # 1) Compute features + target (next-bar return)
        feat = self._compute_features(df)
        target = np.log(feat['close'].shift(-self.horizon) / feat['close'])
        feat['event'] = (target > self.threshold).dropna().astype(int)

        # 3) Split train / test
        X = feat.drop(columns=['event'])            
        y = feat['event']

        split_index = int(len(feat) * self.train_frac)
        X_train_full = X.iloc[:split_index]
        y_train_full = y.iloc[:split_index]
        X_test_full = X.iloc[split_index:]
        y_test_full = y.iloc[split_index:]

        times_train_full = list(X_train_full.index)
        times_test_full  = list(X_test_full.index)

        def make_sequences(X_arr, y_arr, times_arr):
            X_seqs, y_labels, times = [], [], []
            for i in range(len(X_arr) - self.horizon):
                X_seqs.append(X_arr[i : i + self.horizon])
                y_labels.append(y_arr.iloc[i + self.horizon])
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
        search = self.pipeline
        search.set_params(clf__module__n_features=n_feats,
                          clf__module__hidden_size=n_feats,
                          clf__module__dropout     = 0.1,
                          )
        
        
        # from torch import tensor
        # ratio = (len(y_train) - y_train.sum()) / y_train.sum() #Negative/Positive
        # pos_weight = tensor([ratio])
        # self.net.set_params(criterion__pos_weight=pos_weight)
        
        search.fit(X_train, y_train)
        best_pipe = search

        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:,1]

        # 5) Final predictions on test history
        auc_test  = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC (Test): {auc_test:.3f}")

        train_acc = search.score(X_train, y_train)
        test_acc  = search.score(X_test,  y_test)
        print(f"Train score: {train_acc:.3f}, Test score: {test_acc:.3f}")

        # 6) Build a signal series
        positions = pd.Series(np.nan, index=df.index)
        positions.at[times_train[0]] = 0.0
        y_pred_series = pd.Series(0.0, index=df.index)
        y_test_series = pd.Series(0.0, index=df.index)
        test_mask = pd.Series(0.0, index=df.index)
        position = 0
        days_left = 0
        prob_threshold = 0.5
        for t, pred, prob, test in zip(times_test, y_pred, y_prob, y_test):
            # print(pred)
            if days_left > 0:
                days_left -= 1
                if prob >= prob_threshold:
                    days_left += 1
            else:
                if position == 0 and prob >= prob_threshold:
                    position = 1
                    days_left = self.horizon - 1
                elif position == 1 and prob < prob_threshold:
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