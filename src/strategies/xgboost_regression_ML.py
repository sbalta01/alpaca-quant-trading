# src/strategies/xgboost_regression.py

from scipy.stats import uniform, randint
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    RandomizedSearchCV, train_test_split, TimeSeriesSplit, GridSearchCV
)
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.strategies.base_strategy import Strategy
from src.utils.indicators import ema, rsi, sma

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
        rfecv_step: float = 0.1,
        param_grid: Dict[str, Any] = None,
        signal_thresh: float = 0.0, #Minimum daily return to trade
        random_state: int = 42,
        n_iter_search: int = 50
    ):
        self.horizon = horizon
        # train/test split fraction
        self.train_frac = train_frac
        # time-series cross-validation folds
        self.cv_splits = cv_splits
        # RFECV step size
        self.rfecv_step = rfecv_step
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.signal_thresh = np.log(signal_thresh + 1) #Consistent with log-target

        # default hyperparameter grid for XGB
        self.param_grid = param_grid or {
            'model__n_estimators': randint(50, 500),
            'model__max_depth': randint(1, 10),
            'model__learning_rate': uniform(0.01, 2.0),
            'model__subsample': uniform(0.5, 0.5), #loc, scale --> [loc,loc+scale]
        }

        # Build pipeline:
        # 1) Standardize features
        # 2) RFECV wrapped around a base XGBRegressor
        # 3) Final XGBRegressor
        inner_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        base = XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            verbosity=0
        )
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('rfecv', RFECV(
            #     estimator=base,
            #     step=self.rfecv_step,
            #     cv=inner_cv,
            #     scoring='r2',
            #     n_jobs=-1
            # )),
            ('model', base)
        ])

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical & Ichimoku features from price DataFrame.
        Expects df has columns: ['open','high','low','close','volume'].
        """
        df = df.copy()

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

        # Drop NaNs from all rolling
        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data: single-symbol DataFrame with columns
          ['open','high','low','close','volume',
           factor_EPR, factor_BMR, factor_EBITDA, ..., factor_VIX]
        """
        df = data.copy()

        # 1) Compute features + target (next-bar return)
        feat = self._compute_features(df)
        feat['target'] = np.log(feat['close'].shift(-self.horizon)/feat['close'])
        feat = feat.dropna()

        # 3) Split train / test
        X = feat.drop(columns=['target','close','high','low','open'])
        y = feat['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_frac, shuffle=False
        )

        # 4) Nested CV Grid Search/Randomized Search CV
        outer_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        if isinstance(self.param_grid, dict) and all(
            hasattr(v, "rvs") for v in self.param_grid.values()
        ):
            gs = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=self.param_grid,
                n_iter=self.n_iter_search,
                cv=outer_cv,
                # scoring='neg_mean_squared_error',
                scoring='r2',
                random_state=self.random_state,
                n_jobs=-1
            )
            print('Random Search CV')
        else:
            gs = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=outer_cv,
            # scoring='neg_mean_squared_error',
            scoring='r2',
            n_jobs=-1
            )
            print('Grid Search CV')

        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        try:
            support_mask = best.named_steps['rfecv'].support_
            feature_names = X_train.columns
            selected = feature_names[support_mask]
            ranking = best.named_steps['rfecv'].ranking_
            print(f"[{self.name}] RFECV selected {len(selected)}/{len(feature_names)} features. Ranking:")
            for feature_name, rank in zip(feature_names, ranking):
                print(f"{feature_name:15s} → rank {rank}")
        except:
            print('No RFECV')

        # 5) Evaluate R²
        y_pred = pd.Series(best.predict(X_test), index=X_test.index)
        
        r2_train = r2_score(y_train, best.predict(X_train))
        r2_test = r2_score(y_test, y_pred)
        print(f"[{self.name}] R² score. Train: {r2_train:.3f}. Test: {r2_test:.3f}")
        
        # Directional accuracy: how often sign(pred) == sign(true)
        sign_test = np.sign(y_test)
        sign_pred = np.sign(y_pred)
        dir_acc = (sign_test == sign_pred).mean()

        print(f"[{self.name}] average directional accuracy (Test set): {dir_acc:.3f}")

        # 6) Generate stateful signals from continuous preds
        positions = pd.Series(np.nan, index=feat.index)
        positions.at[list(X_train.index)[0]] = 0.0
        y_pred_series = pd.Series(0.0, index=feat.index)
        y_test_series = pd.Series(0.0, index=feat.index)
        test_mask = pd.Series(0.0, index=feat.index)
        idxs = list(X_test.index)
        position = 0
        days_left = 0
        for t, pred, test in zip(idxs, y_pred, y_test):
            if days_left > 0:
                days_left -= 1
            else:
                if position == 0:
                    if pred > self.signal_thresh:
                        position = 1
                        days_left = self.horizon - 1
                    elif pred < -self.signal_thresh:
                        position = -1
                        days_left = self.horizon - 1
                elif (position==1 and pred <= -self.signal_thresh) or (position==-1 and pred>=self.signal_thresh):
                    position = 0 # exit to flat
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
