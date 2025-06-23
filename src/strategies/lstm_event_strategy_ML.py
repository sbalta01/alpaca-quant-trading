# src/strategies/lstm_event_strategy.py

from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import randint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, recall_score, roc_auc_score

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from torch.optim import Adam

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from pykalman import KalmanFilter

from src.strategies.base_strategy import Strategy
from src.utils.tools import sma, ema, rsi

# ──────────────────────────────────────────────────────────────────────────────
def adf_test(series, signif=0.05):
    """
    Perform ADF test and return True if we reject the unit-root null
    (i.e. we conclude stationarity at the given significance level).
    """
    res = adfuller(series.dropna(), autolag='AIC')
    _, pvalue, _, _, _, _  = res
    return pvalue < signif

# ─── ARIMA/GARCH/Kalman ────────────────────────────────────────────────────────────────────
class ARIMAGARCHKalmanTransformer(BaseEstimator, TransformerMixin):
    """Add ARIMA residuals, GARCH vol forecast & Kalman trend to your DataFrame."""

    def __init__(self, window,
                 n_candidates,
                 arima_p_max,
                 arima_q_max,
                 garch_p_max,
                 garch_q_max,
                 kalman_q_max):
        self.window = window
        self.n_candidates = n_candidates
        self.arima_p = randint(1, arima_p_max+1)
        self.arima_q = randint(1, arima_q_max+1)
        self.garch_p = randint(1, garch_p_max+1)
        self.garch_q = randint(1, garch_q_max+1)
        self.kalman_q = randint(1, kalman_q_max+1)

        # will be set in .fit()
        self.best_arima_order = None
        self.best_garch_pq = None
        self.best_kalman_q = None      

        # storage for warm-start state
        self._arima_params = None
        self._garch_params = None
        self._kf_state   = None

    def fit(self, X_train: pd.DataFrame, y=None):
        """
        1) Compute log-returns on the full training slice.
        2) For each (p,d,q) * (p',q') * q_obs:
             • Fit ONE ARIMA on the entire train returns → get AIC
             • Fit ONE GARCH on train returns → get AIC
             • Fit ONE Kalman EM on train returns → reconstruct log-likelihood or MSE
        3) Pick the triple that minimizes (AIC_arima + AIC_garch + Kalman_MSE).
        """
        df = X_train.copy()
        ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)

        d = 0
        adf_test_series = ret.copy()
        while adf_test(adf_test_series) == False:
            adf_test_series = adf_test_series.diff().fillna(0)
            if d == 4:
                raise Exception(f"This dataset is nowhere near stationarity, even for d = {d}")
            d += 1

        best_score = np.inf
        best_score_kalman = np.inf
        for _ in range(self.n_candidates):
            score = 0.0
            score_kalman = 0
            p = self.arima_p.rvs()
            q = self.arima_q.rvs()
            gp = self.garch_p.rvs()
            gq = self.garch_q.rvs()
            kq = self.kalman_q.rvs()

            # 1) ARIMA on full train
            armod = ARIMA(
                ret.values, order=(p,d,q),
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            arres = armod.fit(
                method='innovations_mle',
                # method_kwargs={'disp': False},
                )
            score += arres.aic

            # 2) GARCH on full train
            gmod = arch_model(ret*100, p=gp, q=gq)
            gres = gmod.fit(disp='off')
            score += gres.aic

            # 3) Kalman EM on full train
            kf = KalmanFilter(
                transition_matrices=np.eye(1),
                observation_matrices=np.eye(1),
                transition_covariance=np.eye(1)*0.001,
                observation_covariance=np.eye(1)*kq
            )
            kf_em = kf.em(ret.values, n_iter=5)
            # use total squared error of the smoothed trend as a proxy
            trend, _ = kf_em.filter(ret.values)
            mse = np.mean((ret.values - trend.flatten())**2)
            score_kalman += mse

            if score < best_score:
                best_score = score
                self.best_arima_order = (p,d,q)
                self.best_garch_pq    = (gp,gq)

            if score_kalman < best_score_kalman:
                best_score_kalman = score_kalman
                self.best_kalman_q = kq            

        print(f"[Best Hyperparams] ARIMA={self.best_arima_order}, "
              f"GARCH={self.best_garch_pq}, "
              f"KF_Q={self.best_kalman_q}")        
        return self
    
    # def fit(self, X, y=None):
    #     return self
    
    # def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    #     df = X.copy().reset_index(drop=True)                        
    #     ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)  

    #     # ARIMA residuals
    #     ar_model = ARIMA(ret, order=self.arima_order)\
    #                     .fit(method_kwargs={'disp': False})          
    #     df['arima_resid'] = ar_model.resid                         

    #     # GARCH volatility forecast
    #     garch = arch_model(ret * 100, p=self.garch_p, q=self.garch_q)\
    #                     .fit(disp='off')                              
    #     fcast = garch.forecast(horizon=1).variance.iloc[-1, 0]     
    #     df['garch_vol'] = np.sqrt(fcast) / 100.0                   

    #     # Kalman trend
    #     kf = KalmanFilter(transition_matrices=np.array([[1]]), observation_matrices=np.array([[1]]))  
    #     kf_em = kf.em(ret.values)                                              
    #     trend = kf_em.filter(ret.values)[0].flatten()                           
    #     df['kf_trend'] = trend                                                  

    #     return df.dropna()     

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # df = X.copy().reset_index(drop=True)
        df = X.copy()
        arima_resid   = [np.nan]*self.window
        garch_vol   = [np.nan]*self.window
        kf_trend = [np.nan]*self.window
        # kf = KalmanFilter(transition_matrices=np.array([[1]]), observation_matrices=np.array([[1]]))
        kq = self.best_kalman_q
        kf = KalmanFilter(
            transition_matrices=np.eye(1),
            observation_matrices=np.eye(1),
            transition_covariance=np.eye(1)*0.001,
            observation_covariance=np.eye(1)*kq
        )
        gp, gq = self.best_garch_pq
        arima_order = self.best_arima_order
        for i in range(self.window, len(df)):
                hist = df.iloc[i-self.window : i]
                ret = np.log(hist['close'] / hist['close'].shift(1)).fillna(0)
                # ARIMA residuals
                arima_model = ARIMA(ret.values, order=arima_order,
                enforce_stationarity=True,
                enforce_invertibility=True)
                if self._arima_params is None:
                    arima_fit = arima_model.fit(method_kwargs={'disp': False})
                else:
                    # warm-start from last fit
                    arima_fit = arima_model.fit(start_params=self._arima_params, method_kwargs={'disp': False})
                
                arima_resid.append(arima_fit.resid[-1])
                self._arima_params = arima_fit.params                    

                # GARCH volatility forecast
                garch_model = arch_model(ret*100, p=gp, q=gq)
                if self._garch_params is None:
                    garch_fit = garch_model.fit(disp='off')
                else:
                    garch_fit = garch_model.fit(disp='off',
                                    starting_values=self._garch_params,)
                self._garch_params = garch_fit.params
                fcast = garch_fit.forecast(horizon=1).variance.iloc[-1, 0]     
                garch_vol.append(np.sqrt(fcast)/100.0)                

                # Kalman trend
                if self._kf_state is None:
                    # run a full filter on the first window
                    # kf = kf.em(ret.values)
                    state_means, state_covs = kf.filter(ret.values)
                    mean, cov = state_means[-1], state_covs[-1]
                else:
                    # one-step update with newest observation
                    obs = ret.values[-1]
                    mean, cov = kf.filter_update(
                        filtered_state_mean=self._kf_state[0],
                        filtered_state_covariance=self._kf_state[1],
                        observation=obs
                    )
                self._kf_state = (mean, cov)
                kf_trend.append(mean[0])
        out = pd.DataFrame({
            'arima_resid': arima_resid,
            'garch_vol' : garch_vol,
            'kf_trend'  : kf_trend
        }, index=df.index)
        return pd.concat([df,out], axis=1).dropna()

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
        arima_garch_kalman_window = 252,
        n_candidates = 30,
        arima_p_max = 5,
        arima_q_max = 5,
        garch_p_max = 3,
        garch_q_max = 3,
        kalman_q_max = 4,
        cv_splits: int = 5,
        lstm_hidden: int = 32,
        lstm_dropout: float = 0.2,
        random_state: int = 42
    ):
        self.horizon   = horizon
        self.threshold = np.log(1 + threshold)
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.train_frac = train_frac

        # # Reproducibility
        # np.random.seed(self.random_state)
        # torch.manual_seed(self.random_state)

        self.recall_scorer = make_scorer(recall_score)

        if torch.cuda.is_available() and False: #Disabled
            device = 'cuda'
            print("Current device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            device = 'cpu'
            print("Current device name: ", device)

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
                window = arima_garch_kalman_window,
                n_candidates = n_candidates,
                arima_p_max = arima_p_max,
                arima_q_max = arima_q_max,
                garch_p_max = garch_p_max,
                garch_q_max = garch_q_max,
                kalman_q_max = kalman_q_max,
            )),
            ('tech', TechnicalTransformer()),
        ])

        self.pipeline = Pipeline([
            ('scale', TimeSeriesScaler(StandardScaler())),
            ('clf', self.net)
        ])

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        target = np.log(df['close'].shift(-self.horizon) / df['close'])
        df['event'] = (target > self.threshold).astype(int) #This first so that X and y are aligned when dropping columns


        split_index = int(len(df) * self.train_frac)
        df_train = df.iloc[:split_index]
        df_test = df.iloc[split_index:]
        self.feature_transform.fit(df_train)  # Fit hyperparameters only on training data
        feat_train = self.feature_transform.transform(df_train)
        feat_test = self.feature_transform.transform(df_test)
        X_train_full = feat_train.drop(columns=['event'])
        X_test_full = feat_test.drop(columns=['event'])
        y_train_full = feat_train['event']
        y_test_full = feat_test['event'] #This way the first test columns are not dropped


        # self.feature_transform.fit(df)
        # feat = self.feature_transform.transform(df)
        # X_full = feat
        # y_full = feat['event'].values  
        # X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        #     X_full, y_full, train_size=self.train_frac, shuffle=False
        # )
        times_train_full = list(X_train_full.index)
        times_test_full  = list(X_test_full.index)

        print('First y_test', X_test_full.iloc[0])

        # feat = df.copy().dropna()
        # split_i = int(len(feat) * self.train_frac)                       
        # feat_train = feat.iloc[:split_i].copy()                          
        # feat_test  = feat.iloc[split_i:].copy()                          

        # self.feature_transform.fit(feat_train.drop(columns=['event']))                           
        # X_train_full = self.feature_transform.transform(feat_train.drop(columns=['event']))
        # X_test_full  = self.feature_transform.transform(feat_test.drop(columns=['event']))
        # y_train_full = feat_train['event'].values                        
        # y_test_full  = feat_test['event'].values                         
        # times_train_full = list(feat_train.index)                        
        # times_test_full  = list(feat_test.index)                         

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
            # verbose=2,
            random_state=self.random_state,
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
        print(f"Train recall: {train_acc:.3f}, Test recall: {test_acc:.3f}")

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