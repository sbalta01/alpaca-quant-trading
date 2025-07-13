# src/strategies/xgboost_regression.py

from scipy.stats import uniform, randint
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.strategies.lstm_regression_ML import SequenceBaggingRegressor
from src.utils.metrics import sharpe_scorer
import optuna

from src.strategies.base_strategy import Strategy
from src.utils.tools import adx, ema, rsi, sma, threshold_adjust

class XGBoostRegressionStrategy(Strategy):
    """
    XGBoost regression on an extended feature set:
      - Technicals: price lags, SMAs, EMAs, RSI, OBV, ROC, MACD, Stoch, CCI, Ichimoku
      - Factor variables: E/P, B/M, EBITDA, EPS, PE, earnings growth, Fama-French, EUR/USD, rates, VIX
      - RFECV for feature selection, GridSearchCV for hyperparams
    Emits +1 long / -1 short signals based on next-bar return prediction.
    """
    name = "XGBoostRegression"
    multi_symbol = False

    def __init__(
        self,
        horizon: int = 5,
        train_frac: float = 0.7,
        cv_splits: int = 5,
        n_models: int = 1,
        bootstrap: float = 0.8,
        rfecv_step: float = 0.1,
        signal_thresh: float = 0.0, #Minimum daily return to trade
        random_state: int = 42,
        n_iter_search: int = 50,
        min_features: int = 1,
        objective: str = 'reg:squarederror',
        quantile: float = 0.2, #Quantile to fit for when objective is 'reg:quantileerror'
        with_hyperparam_fit: bool = True,
        with_feature_selection: bool = True,
        adjust_threshold: bool = True,
    ):
        self.horizon = horizon
        self.train_frac = train_frac
        self.cv_splits = cv_splits
        self.n_models = n_models
        self.bootstrap = bootstrap
        self.rfecv_step = rfecv_step
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.min_features = min_features
        self.with_hyperparam_fit = with_hyperparam_fit
        self.with_feature_selection = with_feature_selection
        self.adjust_threshold = adjust_threshold
        self.signal_thresh = np.log(signal_thresh + 1) #Consistent with log-target
        self.objective = objective
        self.quantile = quantile

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical & Ichimoku features from price DataFrame.
        Expects df has columns: ['open','high','low','close','volume'].
        """
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
        df['volume'] = df['volume']
        # df['volume_1'] = df['volume'].shift(1)
        df['volume_inc'] = df['volume'] - df['volume'].shift(1)

        # --- Moving Averages & their diffs ---
        # for w in (5,10,20,50):
        for w in [self.horizon]:
            df[f'sma{w}']    = sma(df['close'], w)
            # df[f'sma{w}_1'] = df[f'sma{w}'].shift(1)
            df[f'sma{w}_inc'] = df[f'sma{w}'] - df[f'sma{w}'].shift(1)
            df[f'ema{w}']    = ema(df['close'], w)
            # df[f'ema{w}_1'] = df[f'ema{w}'].shift(1)
            df[f'ema{w}_inc'] = df[f'ema{w}'] - df[f'ema{w}'].shift(1)

        # --- RSI(14) ---
        df['RSI14'] = rsi(df['close'], 14)

        # --- OBV ---
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # --- ROC ---
        # for w in (5,10,20):
        for w in [self.horizon]:
            df[f'ROC{w}'] = df['close'].pct_change(w)

        # --- MACD + Signal(9) + hist ---
        # df['ema12'] = ema(df['close'], 12)
        # df['ema26'] = ema(df['close'], 26)
        df['macd']  = ema(df['close'], 12) - ema(df['close'], 26)
        # df['macd_sig']  = ema(df['macd'], 9)
        df['macd_hist'] = df['macd'] - ema(df['macd'], 9)

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
        # df['ichimoku_span_a'] = ((df['ichimoku_conv'] + df['ichimoku_base'])/2).shift(26)
        # high52 = df['high'].rolling(52).max()
        # low52  = df['low'].rolling(52).min()
        # df['ichimoku_span_b'] = ((high52 + low52)/2).shift(26)

        # Higher moments of log‑returns
        df['skew5']    = df['logret'].rolling(self.horizon).skew()
        df['kurt5']    = df['logret'].rolling(self.horizon).kurt()
        return df
    
    def _optimize_hyperparams_features(self, X_train_CV, y_train_CV):
        """Use Optuna for hyperparameter fine tuning."""
        def objective(trial):
            n_estimators   = trial.suggest_int("n_estimators", 50, 500)
            max_depth      = trial.suggest_int("max_depth",    1, 10)
            learning_rate  = trial.suggest_float("learning_rate", 1e-3, 2.0, log=True)
            subsample      = trial.suggest_float("subsample",     0.5, 1.0)
            # build a temporary model + pipeline
            model = XGBRegressor(
                objective=self.objective,
                quantile_alpha = self.quantile,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,

                random_state=self.random_state,
                verbosity=0,
            )
            pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model),
                            ])
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            # cross-validate with your sharpe_scorer
            returns = []
            for train_idx, val_idx in tscv.split(X_train_CV):
                pipe.fit(X_train_CV.iloc[train_idx], y_train_CV.iloc[train_idx])
                preds = pipe.predict(X_train_CV.iloc[val_idx])
                # returns.append(r2_score(y_train_CV.iloc[val_idx], preds))
                returns.append(sharpe_scorer(pipe, X_train_CV.iloc[val_idx], y_train_CV.iloc[val_idx]))
            # maximize average Sharpe
            return np.mean(returns)

        if self.with_hyperparam_fit:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=self.n_iter_search, timeout=3600,show_progress_bar=True,)
            best_params = study.best_params
            print("Best hyperparameters:", best_params)
        else:
            best_params = {"n_estimators":100,
                           "max_depth":5,
                           "learning_rate":1e-1,
                           "subsample":0.7,
                           }
            print("No hyperparameter fitting")

        if self.with_feature_selection:
            print("Feature selection")
            rfecv_model = XGBRegressor(
                    objective=self.objective,
                    quantile_alpha = self.quantile,
                    n_estimators   = best_params["n_estimators"],
                    max_depth      = best_params["max_depth"],
                    learning_rate  = best_params["learning_rate"],
                    subsample      = best_params["subsample"],
                
                    random_state   = self.random_state,
                    verbosity      = 0,
                )
            inner_cv = TimeSeriesSplit(n_splits=self.cv_splits)
            rfecv_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', RFECV(
                estimator=rfecv_model,
                step=self.rfecv_step,
                min_features_to_select=self.min_features,
                cv=inner_cv,
                scoring='r2',
                # scoring='neg_mean_absolute_error',
                n_jobs=1
            )),
            ('model', rfecv_model),
                            ])
            rfecv_pipe.fit(X_train_CV, y_train_CV)
            support_mask = rfecv_pipe.named_steps['rfecv'].support_
            feature_names = X_train_CV.columns
            selected = feature_names[support_mask]
            ranking = rfecv_pipe.named_steps['rfecv'].ranking_
            print(f"[{self.name}] RFECV selected {len(selected)}/{len(feature_names)} features. Ranking:")
            for feature_name, rank in zip(feature_names, ranking):
                print(f"{feature_name:15s} → rank {rank}")
        else:
            selected = X_train_CV.columns
            print("No feature selection. Number of features:", len(selected))

        return best_params, selected

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data: single-symbol DataFrame with columns
          ['open','high','low','close','volume',
           factor_EPR, factor_BMR, factor_EBITDA, ..., factor_VIX]
        """
        df = data.copy()

        # 1) Compute features + target (next-bar return)
        df['target'] = np.log(df['close'].shift(-self.horizon)/df['close'])
        feat = self._compute_features(df)
        feat = feat.ffill().dropna()  #Ffill so that we dont lose last rows to dropping Nas.

        # 3) Split train / test
        # X = feat.drop(columns=['target','close','high','low','open'])
        X = feat.drop(columns=['target'])
        y = feat['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_frac, shuffle=False
        )

        best_params, selected = self._optimize_hyperparams_features(X_train, y_train)
        X_train = X_train[selected]
        X_test = X_test[selected]
            
        self.model = XGBRegressor(
                objective=self.objective,
                quantile_alpha = self.quantile,
                n_estimators   = best_params["n_estimators"],
                max_depth      = best_params["max_depth"],
                learning_rate  = best_params["learning_rate"],
                subsample      = best_params["subsample"],

                # base_score      = float(y_train.mean()),
                random_state   = self.random_state,
                verbosity      = 0,
            )
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model),
        ])

        self.ensemble = SequenceBaggingRegressor(
            base_estimator= self.pipeline,
            n_estimators  = self.n_models,
            max_samples   = self.bootstrap, # fraction bootstrap
            random_state  = self.random_state
        )

        search = self.ensemble
        # search = self.pipeline
        search.fit(X_train, y_train)

        # 5) Evaluate R²
        y_pred = pd.Series(search.predict(X_test), index=X_test.index)
        
        r2_train = r2_score(y_train, search.predict(X_train))
        r2_test = r2_score(y_test, y_pred)
        print(f"[{self.name}] R² score. Train: {r2_train:.3f}. Test: {r2_test:.3f}")
        
        # Directional accuracy: how often sign(pred) == sign(true)
        dir_acc_train = (np.sign(y_train) == np.sign(search.predict(X_train))).mean()
        dir_acc_test = (np.sign(y_test) == np.sign(y_pred)).mean()

        print(f"Average directional accuracy (Train): {dir_acc_train:.3f}, (Test): {dir_acc_test:.3f}")

        # 6) Generate stateful signals from continuous preds
        positions = pd.Series(np.nan, index=feat.index)
        positions.at[list(X_train.index)[0]] = 0.0
        y_pred_series = pd.Series(0.0, index=feat.index)
        y_test_series = pd.Series(0.0, index=feat.index)
        test_mask = pd.Series(0.0, index=feat.index)
        idxs = list(X_test.index)
        position = 0
        days_left = 0
        adjusted_signal_thresh = threshold_adjust(feat['logret'], horizon = self.horizon, base_threshold = self.signal_thresh, max_shift=0.4*self.signal_thresh) if self.adjust_threshold else pd.Series(self.signal_thresh, index = idxs)
        y_pred = pd.Series(y_pred).rolling(3).mean() #Smooth the probabilities to avoid single-day flops
        for t, pred, test in zip(idxs, y_pred, y_test):
            if days_left > 0:
                days_left -= 1
                if pred >= adjusted_signal_thresh.at[t]:
                    days_left += 1
            else:
                if position == 0 and pred > adjusted_signal_thresh.at[t]:
                    position = 1
                    days_left = self.horizon - 1
                elif position == 1 and pred <= 0.0:
                    position = 0 #exit to flat
                    # if pred > self.signal_thresh:
                    #     position = 1
                    #     days_left = self.horizon - 1
                #     elif pred < -self.signal_thresh:
                #         position = -1
                #         days_left = self.horizon - 1
                # elif (position==1 and pred <= -self.signal_thresh) or (position==-1 and pred>=self.signal_thresh):
                #     position = 0 # exit to flat
            positions.at[t] = position
            y_pred_series.at[t] = pred
            y_test_series.at[t] = test
            test_mask.at[t] = 1.0
        # print('Max prediction', y_pred_series.max(),y_pred_series.idxmax())
        # print('Max label', y_test.max(),y_test.idxmax())
        # print('Min prediction', y_pred.min(),y_pred.idxmin())
        # print('Min label', y_test.min(),y_test.idxmin())
        # 7) Merge signals into df
        out = df.copy()
        out["position"] = positions
        out["position"] = out["position"].ffill().bfill().clip(0,1) #clip(0,1) for no short. ffill for missing timestamps in test (because of holidays or outliers removal eg). bfill (afterwards) for position = 0 before test. 
        out['signal'] = out['position'].diff().fillna(0.0).clip(-1,1) #fillna for the first signal
        
        out["y_test"] = y_test_series.reindex(df.index).fillna(0.0)
        out["y_pred"] = y_pred_series.reindex(df.index).fillna(0.0)
        out["test_mask"] = test_mask.reindex(df.index).fillna(0.0)
        return out
