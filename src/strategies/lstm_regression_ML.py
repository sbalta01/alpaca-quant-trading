# src/strategies/lstm_regression.py

import logging
import math
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import f_classif

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler, EarlyStopping
import optuna
import io
from contextlib import redirect_stdout, redirect_stderr

from src.strategies.lstm_event_technical_ML import DynamicFeatureAttention, TemporalAttention, TimeSeriesScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


from src.strategies.base_strategy import Strategy
from src.utils.tools import adx, compute_turbulence_single_symbol, sma, ema, rsi

class AttentionLSTMRegressor(nn.Module):
    """
    1) Dynamic feature-level attention per time step
    2) LSTM over the attended inputs
    3) Temporal attention over the LSTM outputs
    4) Final Dense --> logit
    """
    def __init__(self, n_features, with_feature_attn: bool, hidden_size=128, dropout=0.2, attn_dim = 64,):
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
            h = H[:, -1, :]                  # [B, hidden]
            h = self.drop(h)
            return self.out(h).squeeze(-1)
        

class SequenceBaggingRegressor(BaseEstimator, RegressorMixin):
    """
    A simple bootstrap-bagging ensemble for sequence data (3D X).
    
    Parameters
    ----------
    base_estimator : estimator
        Any object implementing fit(X, y) and predict(X).
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

    def predict(self, X):
        """
        Average the predictions from each bootstrap member.
        """
        X = np.asarray(X)
        # collect shape: (n_estimators, n_samples, n_classes)
        all_preds = [est.predict(X) for est in self.estimators_]
        # mean over estimators --> (n_samples, n_classes)
        return np.mean(all_preds, axis=0)


class LSTMRegressionStrategy(Strategy):
    """
    Predict and signal next-N-day log-return.
    Uses technicals + (optional) macro columns,
    a PyTorch LSTM regressor via skorch, RFECV feature-selection,
    and forward-rolling CV for hyperparameter search.
    """
    name = "LSTMRegression"
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
        with_hyperparam_fit: bool = True,
        with_feature_attn: bool = True,
    ):
        self.horizon   = horizon
        self.threshold = np.log(1 + threshold)
        self.cv_splits = cv_splits
        self.n_models = n_models
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.train_frac = train_frac
        self.sequences_length = sequences_length if sequences_length is not None else horizon
        self.with_hyperparam_fit = with_hyperparam_fit
        self.with_feature_attn = with_feature_attn

        if torch.cuda.is_available(): #Disabled
                self.device = 'cuda'
                print("Current device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
                self.device = 'cpu'
                print("Current device name: ", self.device)
        
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = df.index

        # --- Calendar data with cyclical embedding
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek   / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek   / 7)
        df["mo_sin"]  = np.sin(2 * np.pi * (idx.month - 1) / 12)
        df["mo_cos"]  = np.cos(2 * np.pi * (idx.month - 1) / 12)

        # --- Price & lag features ---
        df['logret'] = np.log(df['close'] / df['close'].shift(1))
        df[f'vol{self.horizon}'] = df['logret'].rolling(self.horizon).std()
        df[f'logret{self.horizon}'] = np.log(df['close'] / df['close'].shift(self.horizon))


        # --- Volume features ---
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
        # df['turbulence'] = compute_turbulence_single_symbol(df, window=252)
        return df

    def hyperparameter_fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        n_feats = X_train.shape[2]
        def objective(trial):
            # 1) Suggest hyperparameters
            hidden_size = trial.suggest_int("hidden_size", n_feats-n_feats//2, n_feats + n_feats//2, log=False)
            attn_dim    = trial.suggest_int("attn_dim",    n_feats-n_feats//2, n_feats + n_feats//2, log=False)
            dropout     = trial.suggest_float("dropout",    0.3, 0.5)
            lr          = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
            batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
            max_epochs  = trial.suggest_int("max_epochs",  5, 20)
            
            # 2) Build a skorch net with LSTM Regressor
            net = NeuralNetRegressor(
                module              = AttentionLSTMRegressor,
                module__n_features  = n_feats, 
                module__hidden_size = hidden_size,
                module__attn_dim    = attn_dim,
                module__dropout     = dropout,
                module__with_feature_attn = self.with_feature_attn,
                
                criterion           = nn.MSELoss,
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
        
                with redirect_stdout(f), redirect_stderr(f):
                    pipe.fit(X_train_CV, y_train_CV)
                
                y_pred_CV = pipe.predict(X_val_CV)

                aucs.append(r2_score(y_val_CV, y_pred_CV))
                
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

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()       

        # 1) Compute features + target (next-bar return)
        df['target'] = np.log(df['close'].shift(-self.horizon)/df['close'])
        feat = self._compute_features(df)
        feat = feat.dropna()  #This order (1st event, then features, then dropping na) prevents any misalignment

        # 3) Split train / test
        X = feat.drop(columns=['target']) #Remove data leakage
        # X = feat.drop(columns=['target','open','high','low','close','volume'])
        y = feat['target']

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
        # n_feats = 45

        if self.with_hyperparam_fit:
            best_params = self.hyperparameter_fit(X_train, y_train)
            print("Best hyperparameters:", best_params)
        else:
            best_params = {"hidden_size":n_feats,
                           "attn_dim":64,
                           "dropout":0.3,
                           "lr":1e-3,
                           "batch_size":32,
                           "max_epochs":10,
                           }
            print("No hyperparameter fitting")
        
        self.net = NeuralNetRegressor(
            module              = AttentionLSTMRegressor,
            module__n_features  = n_feats,
            module__hidden_size = best_params["hidden_size"],
            module__attn_dim    = best_params["attn_dim"],
            module__dropout     = best_params["dropout"],
            module__with_feature_attn = self.with_feature_attn,

            criterion           = nn.MSELoss,
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

        self.pipeline = Pipeline([
            ('scale', TimeSeriesScaler(StandardScaler())),
            # ('select', SequenceSelectKBest(score_func=f_classif, k=n_feats)),
            ('clf', self.net)
        ])

        self.ensemble = SequenceBaggingRegressor(
            base_estimator=self.pipeline,
            n_estimators=self.n_models,
            max_samples=self.bootstrap,      # each bag sees max_samples% bootstrap
            random_state=self.random_state
        )
        search = self.ensemble

        search.fit(X_train, y_train)
        
        y_pred = search.predict(X_test)

        # 5) Final predictions on test history
        r2_train = r2_score(y_train, search.predict(X_train))
        r2_test  = r2_score(y_test,  y_pred)
        print(f"R² (Train): {r2_train:.3f}, R² (Test): {r2_test:.3f}")        
        
        # Directional accuracy: how often sign(pred) == sign(true)
        dir_acc_train = (np.sign(y_train) == np.sign(search.predict(X_train))).mean()
        dir_acc_test = (np.sign(y_test) == np.sign(y_pred)).mean()

        print(f"Average directional accuracy (Train): {dir_acc_train:.3f}, (Test): {dir_acc_test:.3f}")

        # 6) Build a signal series
        positions = pd.Series(np.nan, index=df.index)
        positions.at[times_train[0]] = 0.0
        y_pred_series = pd.Series(0.0, index=df.index)
        y_test_series = pd.Series(0.0, index=df.index)
        test_mask = pd.Series(0.0, index=df.index)
        position = 0
        days_left = 0
        for t, pred, test in zip(times_test, y_pred, y_test):
            if days_left > 0:
                days_left -= 1
                if pred  >= self.threshold:
                    days_left += 1
            else:
                if position == 0 and pred  >= self.threshold:
                    position = 1
                    days_left = self.horizon - 1
                elif position == 1 and pred  < self.threshold:
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