# src/strategies/lstm_event_strategy.py

import logging
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import f_classif

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EarlyStopping
import optuna
from torch import tensor
import io
from contextlib import redirect_stdout, redirect_stderr

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


from src.strategies.base_strategy import Strategy
from src.utils.tools import adx, compute_turbulence_single_symbol, sma, ema, rsi, threshold_adjust



class DynamicFeatureAttention(nn.Module):
    """
    For each time-step t, computes an attention weight over features:
      alpha_t = softmax( W2 · tanh(W1 x_t + b1) + b2 )
    where x_t ∈ R^F is the feature-vector at time t.
    """
    def __init__(self, n_features, attn_dim=64):
        super().__init__()
        self.W1 = nn.Linear(n_features, attn_dim, bias=True)
        self.W2 = nn.Linear(attn_dim, n_features, bias=True)
        self.ln = nn.LayerNorm(n_features)

    def forward(self, X):
        # X: [batch, seq_len, features]
        B, T, F_ = X.shape
        # flatten to [B*T, F]
        x_flat = X.view(B * T, F_)
        h       = torch.tanh(self.W1(x_flat))        # [B*T, attn_dim]
        scores  = self.W2(h)                         # [B*T, F]
        scores = scores / math.sqrt(F_)
        alpha       = F.softmax(scores, dim=-1)          # attention per feature
        # re-shape alpha --> [B, T, F]
        alpha = alpha.view(B, T, F_)
        # weighted input
        # return X * alpha
        X_att = X * alpha.view(B, T, F_)
        # add residual + norm
        return self.ln(X + X_att)


class TemporalAttention(nn.Module):
    """
    Given LSTM outputs H \in R^{B*T*H}, computes an attention
    over the time dimension and returns a single context vector:
      beta = softmax( v^T tanh(W H + b) )
      context = sigma_t beta_t H_t
    """
    def __init__(self, hidden_size, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(hidden_size, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, H):
        # H: [B, T, H]
        B, T, Hdim = H.shape
        # flatten to [B*T, H]
        h_flat = H.contiguous().view(B * T, Hdim)
        u      = torch.tanh(self.W(h_flat))             # [B*T, attn_dim]
        e      = self.v(u).view(B, T)                   # [B, T]
        beta      = F.softmax(e, dim=1).unsqueeze(-1)       # [B, T, 1]
        # context: weighted sum over time
        context = (H * beta).sum(dim=1)                     # [B, H]
        # return context
        # residual from last LSTM hidden
        res = H[:, -1, :]
        return self.ln(context + res)


class AttentionLSTMClassifier(nn.Module):
    """
    1) Dynamic feature-level attention per time step
    2) LSTM over the attended inputs
    3) Temporal attention over the LSTM outputs
    4) Final Dense --> logit
    """
    def __init__(self, n_features, with_feature_attn: bool, hidden_size=32, dropout=0.2, attn_dim = 16,):
        super().__init__()
        self.feat_attn = DynamicFeatureAttention(n_features, attn_dim=attn_dim)
        self.lstm      = nn.LSTM(
                                input_size=n_features,
                                hidden_size=hidden_size,
                                num_layers  = 2,           # two stacked layers 
                                bidirectional = True,      # bidirectional 
                                batch_first=True
                                )
        # self.temp_attn = TemporalAttention(hidden_size, attn_dim=attn_dim)
        self.temp_attn = TemporalAttention(hidden_size * 2, attn_dim=attn_dim) 
        self.drop      = nn.Dropout(dropout)
        # self.out       = nn.Linear(hidden_size, 1)
        self.out   = nn.Linear(hidden_size * 2, 1) # bidirectional doubles the hidden dimension 

        self.with_feature_attn = with_feature_attn

    def forward(self, X):
        self.lstm.flatten_parameters()
        if self.with_feature_attn:
            # 1) feature attention
            X_att   = self.feat_attn(X)                   # [B, T, F]
            # 2) LSTM
            H, _     = self.lstm(X_att)                   # H: [B, T, hidden]
            # 3) temporal attention
            context = self.temp_attn(H)                   # [B, hidden]
            # 4) dropout & final logit
            c = self.drop(context)
            return self.out(c).squeeze(-1)                # [B]
        else: #No attention layers
            H, _     = self.lstm(X)                   # H: [B, T, hidden]
            h = H[:, -1, :]                  # [B, hidden] #Returns sequences = False (Keep the last value of each sequence only, for steps after LSTM)
            h = self.drop(h)                 # [B, 1]
            return self.out(h).squeeze(-1) # [B]


class TimeSeriesScaler(TransformerMixin, BaseEstimator):
    """
    Wraps any 2D scaler (e.g. StandardScaler) so it will
    scale each feature across both the sample and time axes.
    Expects X of shape (n_samples, seq_len, n_features).
    """
    def __init__(self, scaler=None):
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


class FocalLoss(nn.BCEWithLogitsLoss):
    """
    Focal Loss for binary classification.
    """
    def __init__(self, pos_weight: float= 1.0, alpha: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.pos_weight = tensor([pos_weight])
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = super().forward(logits, targets)

        # p_t = {p if y=1; 1-p if y=0}
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)

        # focal weighting factor
        focal_factor = self.alpha * (1 - p_t) ** self.gamma

        loss = focal_factor * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        

class SequenceBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple bootstrap-bagging ensemble for sequence data (3D X).
    
    Parameters
    ----------
    base_estimator : estimator
        Any object implementing fit(X, y) and predict_proba(X).
    n_estimators : int
        Number of bootstrap members.
    max_samples : float or int, default=1.0
        If float in (0,1], fraction of samples to draw for each bootstrap.
        If int, number of samples.
    n_jobs : int, default=1
        Parallel jobs for fitting / predicting.
    random_state : int or None
        Seed for reproducible bootstraps.
    """
    def __init__(
        self,
        base_estimator,
        n_estimators: int = 10,
        max_samples=1.0,
        n_jobs: int = 1,
        random_state=None
    ):
        self.base_estimator = base_estimator
        self.n_estimators   = n_estimators
        self.max_samples    = max_samples
        self.n_jobs         = n_jobs
        self.random_state   = random_state

    def fit(self, X, y):
        """
        Fit n_estimators copies of base_estimator on bootstrap samples.
        
        X: array-like, shape (n_samples, seq_len, n_features)
        y: array-like, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = len(X)
        rng = np.random.RandomState(self.random_state)

        # determine how many per bootstrap
        if isinstance(self.max_samples, float):
            m = int(self.max_samples * n_samples)
        else:
            m = int(self.max_samples)

        def _fit_one(seed):
            # each job has its own RNG
            r = np.random.RandomState(seed)
            torch.manual_seed(seed)
            idx = r.randint(0, n_samples, size=m)
            idx = np.sort(idx)
            clone_est = clone(self.base_estimator)
            clone_est.fit(X[idx], y[idx])
            return clone_est

        seeds = rng.randint(0, 2**16 - 1, size=self.n_estimators)
        self.estimators_ = [_fit_one(int(s)) for s in seeds]
        return self

    def predict_proba(self, X):
        """
        Average the predicted probabilities from each bootstrap member.
        """
        X = np.asarray(X)
        # collect shape: (n_estimators, n_samples, n_classes)
        all_probs = [est.predict_proba(X) for est in self.estimators_]
        # mean over estimators --> (n_samples, n_classes)
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        """
        Majority vote on predict_proba (threshold at 0.5).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    

class SequenceSelectKBest(BaseEstimator, TransformerMixin):
    """
    Univariate feature selection for 3D sequence data.
    
    During fit:
      - Collapses X to 2D via last-timestep (or mean over time) 
      - Runs your score_func (e.g. f_classif) against y
      - Stores the top-k feature indices.
    
    During transform:
      - Just selects those k feature columns at every time step
        so your output stays 3D: (n_samples, seq_len, k).
    """
    def __init__(self, score_func=f_classif, k=50, use_last_step: bool = True):
        self.score_func  = score_func
        self.k           = k
        self.use_last_step = use_last_step

    def fit(self, X, y):
        # X: (n_samples, seq_len, n_features)
        if self.use_last_step:
            # score on just the final time step
            X2d = X[:, -1, :]     # shape (n_samples, n_features)
        else:
            # or collapse by time-average
            X2d = X.mean(axis=1)  # shape (n_samples, n_features)

        # run the univariate test
        scores, _ = self.score_func(X2d, y)
        # pick top-k
        self.support_ = np.argsort(scores)[::-1][: self.k]
        return self

    def transform(self, X):
        return X[:, :, self.support_]

    def get_support(self):
        # mimic sklearn API
        return self.support_







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
        n_models: int = 1,
        bootstrap: float = 0.8,
        random_state: int = 42,
        sequences_length: int = None,
        prob_positive_threshold: float = 0.5,
        with_hyperparam_fit: bool = True,
        with_feature_attn: bool = True,
        with_pos_weight: bool = True,
        adjust_threshold: bool = False,
    ):
        self.horizon   = horizon
        self.threshold = np.log(1 + threshold)
        self.cv_splits = cv_splits
        self.n_models = n_models
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.train_frac = train_frac
        self.sequences_length = sequences_length if sequences_length is not None else horizon
        self.prob_positive_threshold = prob_positive_threshold
        self.with_hyperparam_fit = with_hyperparam_fit
        self.with_feature_attn = with_feature_attn
        self.with_pos_weight = with_pos_weight
        self.adjust_threshold = adjust_threshold

        if torch.cuda.is_available(): #Disabled
                self.device = 'cuda'
                print("Current device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
                self.device = 'cpu'
                print("Current device name: ", self.device)
        
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical & Ichimoku features from price DataFrame.
        Expects df has columns: ['open','high','low','close','volume'].
        """
        df = df.copy()

        # --- Price & lag features ---
        df['logret'] = np.log(df['close'] / df['close'].shift(1))
        df[f'logret{self.horizon}'] = np.log(df['close'] / df['close'].shift(self.horizon))


        # --- Volume features ---
        df['volume'] = df['volume']
        df['volume_inc'] = df['volume'] - df['volume'].shift(1)

        # --- Moving Averages & their diffs ---
        for w in (5,20,50):
            df[f'sma{w}']    = sma(df['close'], w)
            df[f'ema{w}']    = ema(df['close'], w)

        # --- RSI(14) ---
        df['RSI14'] = rsi(df['close'], 14)

        # --- OBV ---
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # --- MACD + Signal(9) + hist ---
        df['macd']  = ema(df['close'], 12) - ema(df['close'], 26)
        df['macd_hist'] = df['macd'] - ema(df['macd'], 9)

        # --- Stochastic %K(3), %D(3) ---
        low3  = df['low'].rolling(3).min()
        high3 = df['high'].rolling(3).max()
        df['sto_k'] = 100 * (df['close'] - low3) / (high3 - low3)
        df['sto_d'] = df['sto_k'].rolling(3).mean()

        # --- ADX(14) ---
        df["adx_14"] = adx(df["high"], df["low"], df["close"], window=14)

        # Price‑Volume clusters: rolling VWAP deviation
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_dev'] = (df['close'] - vwap) / vwap


        # # Bollinger Band Width
        # bb_mid   = df['close'].rolling(20).mean()
        # bb_std   = df['close'].rolling(20).std()
        # bb_upper = bb_mid + 2 * bb_std
        # bb_lower = bb_mid - 2 * bb_std
        # df['bb_width'] = (bb_upper - bb_lower) / bb_mid

        # # Average True Range (ATR)
        # high_low    = df['high'] - df['low']
        # high_close1 = np.abs(df['high'] - df['close'].shift(1))
        # low_close1  = np.abs(df['low']  - df['close'].shift(1))
        # tr    = pd.concat([high_low, high_close1, low_close1], axis=1).max(axis=1)
        # df['atr_14'] = tr.rolling(14).mean()




        # from scipy.signal import argrelextrema
        # # convert to numpy for extrema
        # highs = df['high'].values
        # lows  = df['low'].values

        # # 5‑bar lookback/lookahead
        # swing_high_idx = argrelextrema(highs, np.greater_equal, order=5)[0]
        # swing_low_idx  = argrelextrema(lows,  np.less_equal,    order=5)[0]

        # # mark swing points
        # df['swing_high'] = 0
        # df.loc[df.index[swing_high_idx], 'swing_high'] = 1
        # df['swing_low']  = 0
        # df.loc[df.index[swing_low_idx],  'swing_low']  = 1

        # # rolling count of last N swings
        # df['recent_swings'] = df['swing_high'].rolling(50).sum() + df['swing_low'].rolling(50).sum()



        # # Mark prior high/low
        # prev_high = df['high'].rolling(20).max().shift(1)
        # prev_low  = df['low'].rolling(20).min().shift(1)

        # # Boolean for break
        # df['break_up']   = (df['close'] > prev_high).astype(int)
        # df['break_down'] = (df['close'] < prev_low).astype(int)



        # # Volume spikes
        # df['vol_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)


        # # Trend regime: slope of 20‑period SMA
        # df['slope_20'] = np.arctan((df['sma20'] - df['sma20'].shift(1)))  # angle

        # # Consolidation vs trending: ratio of range to ATR
        # df['range_to_atr'] = df['high'] - df['low'] / df['atr_14']






        
        # df = df.drop(columns=['open','high','low','close','volume'])
        df = df.drop(columns=['open','high','low'])
        return df

    def hyperparameter_fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        n_feats = X_train.shape[2]
        if len(np.unique(y_train)) < 2:
            raise ValueError("One class is not represented within the entire training set")
        naive_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() #0's/1's
        def objective(trial):
            # 1) Suggest hyperparameters
            hidden_size = trial.suggest_int("hidden_size", n_feats-n_feats//2, n_feats + n_feats//2, log=False)
            attn_dim    = trial.suggest_int("attn_dim",    n_feats-n_feats//2, n_feats + n_feats//2, log=False)
            dropout     = trial.suggest_float("dropout",    0.3, 0.5)
            lr          = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
            batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
            max_epochs  = trial.suggest_int("max_epochs",  5, 20)
            pos_weight  = trial.suggest_float("pos_weight",  min(1,naive_pos_weight-naive_pos_weight//2), naive_pos_weight+naive_pos_weight//2)
            alpha       = trial.suggest_float("alpha",  0.1, 2)
            gamma       = trial.suggest_float("gamma",  0.25, 2)
            
            # 2) Build a skorch net with LSTM Classifier
            net = NeuralNetClassifier(
                module              = AttentionLSTMClassifier,
                module__n_features  = n_feats,            
                module__hidden_size = hidden_size,
                module__attn_dim    = attn_dim,
                module__dropout     = dropout,
                module__with_feature_attn = self.with_feature_attn,
                
                criterion           = FocalLoss(pos_weight = pos_weight, alpha = alpha, gamma=gamma), ##Equivalent to weighted BCE with logits when alpha = 1.0, gamma = 0.0
                optimizer           = Adam,
                optimizer__weight_decay=1e-4, #L2 regularization
                lr                  = lr,
                
                batch_size          = batch_size,
                max_epochs          = max_epochs,
            
                train_split         = None,
                iterator_train__shuffle = False,
                callbacks = [
                        ('early_stop',
                            EarlyStopping(
                                    monitor='train_loss', 
                                    patience=3, 
                                    threshold=1e-4
                                    )),
                        # ('lr_scheduler', 
                        #     LRScheduler(
                        #             policy = 'CosineAnnealingLR',
                        #             T_max  = 20
                        #         )),
                    ],
                device              = self.device,
            )

            pipe = Pipeline([
                ('scale', TimeSeriesScaler(StandardScaler())),
                ('clf', net)
            ])
            
            # 3) time-series CV
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            aucs = []
            f = io.StringIO()
            for train_idx, valid_idx in tscv.split(X_train):
                X_train_CV, y_train_CV = X_train[train_idx], y_train[train_idx]
                X_val_CV, y_val_CV     = X_train[valid_idx], y_train[valid_idx]

                # If either split has only one class, skip this trial
                if len(np.unique(y_train_CV)) < 2 or len(np.unique(y_val_CV)) < 2:
                    print("Either train or test has only one class, skip this trial")
                    raise optuna.TrialPruned()
        
                with redirect_stdout(f), redirect_stderr(f):
                    pipe.fit(X_train_CV, y_train_CV)
                
                y_pred_CV = pipe.predict(X_val_CV)
                y_prob_CV = pipe.predict_proba(X_val_CV)[:,1]

                if len(np.unique(y_pred_CV)) < 2:
                    print("Only one class has been predicted, skip this trial")
                    raise optuna.TrialPruned()
                
                # aucs.append(roc_auc_score(y_val_CV, y_prob_CV))
                aucs.append(precision_score(y_val_CV, y_pred_CV))
                # aucs.append(recall_score(y_val_CV, y_pred_CV))
                
                # allow Optuna to prune bad trials early
                trial.report(np.mean(aucs), len(aucs))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(aucs)

        study = optuna.create_study(
            direction   = "maximize",
            sampler     = optuna.samplers.TPESampler(),   # Bayesian
            pruner      = optuna.pruners.MedianPruner()   # early stopping
        )
        study.optimize(objective, n_trials=25, timeout=3600,show_progress_bar=True,)

        return study.best_params

    def generate_signals(self, data: pd.DataFrame, fit: bool = True,) -> pd.DataFrame:
        df = data.copy()       

        # 1) Compute features + target (next-bar return)
        target = np.log(df['close'].shift(-self.horizon) / df['close'])
        df['event'] = (target > self.threshold).astype(int) #This will not affect the calculation of any feature, thus no leakage
        feat = self._compute_features(df)
        
        #The order (1st event, then features, then dropping na) prevents any misalignment
        feat = feat.ffill().dropna() #Ffill so that we dont lose last rows to dropping Nas.

        # 3) Split train / test
        X = feat.drop(columns=['event']) #Remove data leakage
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
            for i in range(len(X_arr) - self.sequences_length):
                X_seqs.append(X_arr[i : i + self.sequences_length])
                y_labels.append(y_arr.iloc[i + self.sequences_length])
                times.append(times_arr[i + self.sequences_length])
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
        print("Number of features:", n_feats)
        # n_feats = 45

        if self.with_hyperparam_fit:
            try:
                best_params = self.hyperparameter_fit(X_train, y_train)
                print("Best hyperparameters:", best_params)
            except ValueError:
                logging.warning(f"No successful trials; skipping best-parameter step.")
                best_params = {"hidden_size":n_feats,
                           "attn_dim":16,
                           "dropout":0.3,
                           "lr":1e-3,
                           "batch_size":32,
                           "max_epochs":10,
                           "alpha":0.25,
                           "gamma":1,
                           }
                if self.with_pos_weight:
                    best_params['pos_weight'] = (len(y_train) - y_train.sum()) / y_train.sum() #0's/1's
                else:
                    best_params['pos_weight'] = 1.0
        else:
            best_params = {"hidden_size":n_feats,
                           "attn_dim":64,
                           "dropout":0.3,
                           "lr":1e-3,
                           "batch_size":32,
                           "max_epochs":10,
                           "alpha":0.25,
                           "gamma":1,
                           }
            print("No hyperparameter fitting")
            if self.with_pos_weight:
                best_params['pos_weight'] = (len(y_train) - y_train.sum()) / y_train.sum() #0's/1's
            else:
                best_params['pos_weight'] = 1.0
        
        self.net = NeuralNetClassifier(
            module              = AttentionLSTMClassifier,
            module__n_features  = n_feats,
            module__hidden_size = best_params["hidden_size"],
            module__attn_dim    = best_params["attn_dim"],
            module__dropout     = best_params["dropout"],
            module__with_feature_attn = self.with_feature_attn,

            criterion           = FocalLoss(pos_weight = best_params['pos_weight'], alpha = best_params['alpha'], gamma=best_params['gamma']), ##Equivalent to weighted BCE with logits when alpha = 1.0, gamma = 0.0
            optimizer           = Adam,
            optimizer__weight_decay=1e-4, #L2 regularization
            lr                  = best_params["lr"],

            batch_size          = best_params["batch_size"],
            max_epochs          = best_params["max_epochs"],
            
            iterator_train__shuffle = False,
            train_split         = None,       # we’re doing final train on *all* data
            callbacks = [
                    ('early_stop',
                        EarlyStopping(
                                monitor='train_loss', 
                                patience=3, 
                                threshold=1e-4
                                )),
                    # ('lr_scheduler', 
                        # LRScheduler(
                        #         policy = 'CosineAnnealingLR',
                        #         T_max  = 20
                        #     )),
                ],
            verbose             = 0,
            device              = self.device
        )

        if len(np.unique(y_train)) < 2:
            raise Exception("One class is not represented within the entire training set")
        
        self.pipeline = Pipeline([
            ('scale', TimeSeriesScaler(StandardScaler())),
            # ('select', SequenceSelectKBest(score_func=f_classif, k=n_feats)),
            ('clf', self.net)
        ])

        self.ensemble = SequenceBaggingClassifier(
            base_estimator=self.pipeline,
            n_estimators=self.n_models,
            max_samples=self.bootstrap,      # each bag sees max_samples% bootstrap
            random_state=self.random_state
        )
        if fit:
            self.final_model = self.ensemble
            self.final_model.fit(X_train, y_train)
        else:
            pass
        
        y_pred = self.final_model.predict(X_test)
        y_prob = self.final_model.predict_proba(X_test)[:, 1]
        y_prob_train = self.final_model.predict_proba(X_train)[:, 1]


        # 5) Final predictions on test history
        prec_train = precision_score(y_train, self.final_model.predict(X_train))
        prec_test = precision_score(y_test, y_pred)
        print(f"Precision (Train): {prec_train:.3f}, Precision (Test): {prec_test:.3f}")

        acc_train = accuracy_score(y_train, self.final_model.predict(X_train))
        acc_test = accuracy_score(y_test, y_pred)
        print(f"Accuracy (Train): {acc_train:.3f}, Accuracy (Test): {acc_test:.3f}")

        auc_train  = roc_auc_score(y_train, y_prob_train)
        auc_test  = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC (Train): {auc_train:.3f}, ROC AUC (Test): {auc_test:.3f}")

        # 6) Build a signal series
        positions = pd.Series(np.nan, index=df.index)
        positions.at[times_train[0]] = 0.0
        y_pred_series = pd.Series(0.0, index=df.index)
        y_test_series = pd.Series(0.0, index=df.index)
        test_mask = pd.Series(0.0, index=df.index)
        position = 0
        days_left = 0
        
        adjusted_prob_positive_threshold = threshold_adjust(feat['logret'], horizon = self.horizon, base_threshold = 0.5, max_shift=0.2) if self.adjust_threshold else pd.Series(self.prob_positive_threshold, index = times_test)
        
        #Smooth the probabilities to avoid single-day flops
        y_prob = pd.Series(y_prob).rolling(3).mean()

        for t, prob, test in zip(times_test, y_prob, y_test):
            pred = 0
            if days_left > 0:
                days_left -= 1
                if prob >= adjusted_prob_positive_threshold.at[t]:
                    days_left += 1
                    pred = 1
            else:
                if position == 0 and prob >= adjusted_prob_positive_threshold.at[t]:
                    position = 1
                    pred = 1
                    days_left = self.horizon - 1
                elif position == 1 and prob < adjusted_prob_positive_threshold.at[t]:
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
    
    def fit_and_save(self, data: pd.DataFrame, model_path: str):
        """
        Train on all data up through 'yesterday', then serialize the fitted pipeline.
        """
        df = data.copy()       

        # 1) Compute features + target (next-bar return)
        target = np.log(df['close'].shift(-self.horizon) / df['close'])
        df['event'] = (target > self.threshold).astype(int) #This will not affect the calculation of any feature, thus no leakage
        feat = self._compute_features(df)
        
        #The order (1st event, then features, then dropping na) prevents any misalignment
        feat = feat.ffill().dropna() #Ffill so that we dont lose last rows to dropping Nas.

        # 3) Split train / test
        X_full = feat.drop(columns=['event']) #Remove data leakage
        y_full = feat['event']

        def make_sequences_X_y(X_arr, y_arr):
            X_seqs, y_labels = [], []
            for i in range(len(X_arr) - self.sequences_length):
                X_seqs.append(X_arr[i : i + self.sequences_length])
                y_labels.append(y_arr.iloc[i + self.sequences_length])
            return np.stack(X_seqs), np.array(y_labels)
        X, y = make_sequences_X_y(X_full,y_full)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        n_feats = X.shape[2]

        if self.with_hyperparam_fit:
            try:
                best_params = self.hyperparameter_fit(X, y)
                print("Best hyperparameters:", best_params)
            except ValueError:
                logging.warning(f"No successful trials; skipping best-parameter step.")
                best_params = {"hidden_size":n_feats,
                           "attn_dim":64,
                           "dropout":0.3,
                           "lr":1e-3,
                           "batch_size":32,
                           "max_epochs":10,
                           "alpha":0.25,
                           "gamma":1,
                           }
                if self.with_pos_weight:
                    best_params['pos_weight'] = (len(y) - y.sum()) / y.sum() #0's/1's
                else:
                    best_params['pos_weight'] = 1.0
        else:
            best_params = {"hidden_size":n_feats,
                           "attn_dim":64,
                           "dropout":0.3,
                           "lr":1e-3,
                           "batch_size":32,
                           "max_epochs":10,
                           "alpha":0.25,
                           "gamma":1,
                           }
            print("No hyperparameter fitting")
            if self.with_pos_weight:
                best_params['pos_weight'] = (len(y) - y.sum()) / y.sum() #0's/1's
            else:
                best_params['pos_weight'] = 1.0
        
        self.net = NeuralNetClassifier(
            module              = AttentionLSTMClassifier,
            module__n_features  = n_feats,
            module__hidden_size = best_params["hidden_size"],
            module__attn_dim    = best_params["attn_dim"],
            module__dropout     = best_params["dropout"],
            module__with_feature_attn = self.with_feature_attn,

            criterion           = FocalLoss(pos_weight = best_params['pos_weight'], alpha = best_params['alpha'], gamma=best_params['gamma']), ##Equivalent to weighted BCE with logits when alpha = 1.0, gamma = 0.0
            optimizer           = Adam,
            optimizer__weight_decay=1e-4, #L2 regularization
            lr                  = best_params["lr"],

            batch_size          = best_params["batch_size"],
            max_epochs          = best_params["max_epochs"],
            
            iterator_train__shuffle = False,
            train_split         = None,       # we’re doing final train on *all* data
            callbacks = [
                    ('early_stop',
                        EarlyStopping(
                                monitor='train_loss', 
                                patience=3, 
                                threshold=1e-4
                                )),
                    # ('lr_scheduler', 
                        # LRScheduler(
                        #         policy = 'CosineAnnealingLR',
                        #         T_max  = 20
                        #     )),
                ],
            verbose             = 0,
            device              = self.device
        )

        if len(np.unique(y)) < 2:
            raise Exception("One class is not represented within the entire training set")
        
        self.pipeline = Pipeline([
            ('scale', TimeSeriesScaler(StandardScaler())),
            ('clf', self.net)
        ])

        self.ensemble = SequenceBaggingClassifier(
            base_estimator=self.pipeline,
            n_estimators=self.n_models,
            max_samples=self.bootstrap,      # each bag sees max_samples% bootstrap
            random_state=self.random_state
        )
        self.final_model = self.ensemble
        self.final_model.fit(X, y)
        
        y_prob = self.final_model.predict_proba(X)[:, 1]

        # Final metrics on history
        prec_train = precision_score(y, self.final_model.predict(X))
        print(f"Precision (Train): {prec_train:.3f}")

        acc_train = accuracy_score(y, self.final_model.predict(X))
        print(f"Accuracy (Train): {acc_train:.3f}")

        auc_train  = roc_auc_score(y, y_prob)
        print(f"ROC AUC (Train): {auc_train:.3f}")

        # 3) Save:
        p = Path(model_path)
        p.parent.mkdir(exist_ok=True, parents=False)
        joblib.dump(self.final_model, str(p))
        print(f"Saved model to {p}")

    def load(self, model_path: str):
        """Load a pipeline that was trained via .fit_and_save(...)"""
        self.final_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

    def predict_next(self, data: pd.DataFrame) -> float:
        """
        recent_data: DataFrame of exactly `sequences_length` rows,
                     in chronological order, up through yesterday's close.
        Returns: probability of an event (buy) over the next horizon.
        """
        smoothing_window = 3
        # 1) Compute features on the full window
        feat = self._compute_features(data)
        feat = feat.dropna()
        feat_last = feat.iloc[-self.sequences_length - 1 - smoothing_window + 1:] 
        time_last = list(feat_last.index)

        def make_sequences_X_times(X_arr, times_arr):
                X_seqs, times = [], []
                for i in range(len(X_arr) - self.sequences_length):
                    X_seqs.append(X_arr[i : i + self.sequences_length])
                    times.append(times_arr[i + self.sequences_length])
                return np.stack(X_seqs), times
        
        X, times = make_sequences_X_times(feat_last, times_arr=time_last) #X shape: (smoothing_window, seq_len, n_feats)
        X = X.astype(np.float32)
        time = times[-1]

        # 2) Predict
        probs = self.final_model.predict_proba(X)[:,1]

        #Smooth the probabilities to avoid single-day flops
        probs = pd.Series(probs).rolling(smoothing_window).mean()
        prob = probs.iloc[-1]

        adjusted_prob_positive_threshold = threshold_adjust(feat['logret'], horizon = self.horizon, base_threshold = 0.5, max_shift=0.2) if self.adjust_threshold else pd.Series(self.prob_positive_threshold, index = times)
        position = (prob >= adjusted_prob_positive_threshold.at[time]).astype(np.int32)
        return position, time