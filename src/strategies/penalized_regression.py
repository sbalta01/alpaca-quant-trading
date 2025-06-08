# src/strategies/penalized_regression.py

from scipy.stats import uniform
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.strategies.base_strategy import Strategy
from src.utils.indicators import remove_outliers


class PenalizedRegressionStrategy(Strategy):
    """
    Predict next-bar return with ElasticNet + RFECV + GridSearchCV,
    then go long if pred>0 & flat→long, short if pred<0 & long→flat, etc.
    """
    name = "PenalizedRegression"
    multi_symbol = False

    def __init__(
        self,
        train_frac: float = 0.7,
        cv_splits: int = 5,
        rfecv_step: float = 0.1,
        param_grid: Dict[str, Any] = None,
        random_state: int = 42,
        ratio_outliers: float = np.inf,
        n_iter_search: int = 50
    ):
        """
        train_frac : fraction for train/test split
        cv_splits  : time-series folds
        rfecv_step : fraction of features to remove each RFECV iteration
        param_grid : grid for ElasticNet hyperparams; defaults below
        """
        self.train_frac = train_frac
        self.cv_splits = cv_splits
        self.rfecv_step = rfecv_step
        self.random_state = random_state
        self.ratio_outliers = ratio_outliers
        self.n_iter_search = n_iter_search
        self.param_grid = param_grid or {
            'reg__alpha': uniform(1e-4, 20),
            'reg__l1_ratio': uniform(0.01, 0.99)
        }

        # pipeline: scale → RFECV(ElasticNet) → final ElasticNet
        inner_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', RFECV(
                estimator=ElasticNet(random_state=self.random_state, max_iter=5000),
                step=self.rfecv_step,
                cv=inner_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )),
            ('reg', ElasticNet(random_state=self.random_state, max_iter=5000))
        ])

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic feature set: lag returns, rolling momentum & volatility.
        Returns a DataFrame indexed by timestamp.
        """
        df = df.copy()
        # 1-bar log return
        df['ret1'] = np.log(df['close']).diff()
        # momentum: 5- & 10-bar log returns
        df['mom5']  = np.log(df['close']).diff(5)
        df['mom10'] = np.log(df['close']).diff(10)
        # rolling volatilities
        df['vol5']  = df['ret1'].rolling(5).std()
        df['vol10'] = df['ret1'].rolling(10).std()
        # simple moving averages
        df['ma5']   = df['close'].rolling(5).mean()
        df['ma10']  = df['close'].rolling(10).mean()
        df['ma5_10'] = df['ma5'] - df['ma10']
        return df.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data : single-symbol DataFrame with DatetimeIndex & columns ['open','high','low','close','volume']
        returns : same DataFrame + ['signal'] column
        """
        df = data.copy()
        # 1) Build features & target (next-bar log return)
        feat = self._compute_features(df)
        feat['target'] = np.log(df['close']).shift(-1) - np.log(df['close'])
        feat = feat.dropna()
        feat = remove_outliers(feat, ratio_outliers=self.ratio_outliers)

        # 2) Split train/test (no shuffle)
        X = feat.drop(columns=['target'])
        y = feat['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_frac, shuffle=False
        )

        # 3) Nested CV Grid Search/Randomized Search CV
        outer_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        if isinstance(self.param_grid, dict) and all(
            hasattr(v, "rvs") for v in self.param_grid.values()
        ):
            gs = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=self.param_grid,
                n_iter=self.n_iter_search,
                cv=outer_cv,
                scoring='accuracy',
                random_state=self.random_state,
                n_jobs=-1
            )
            print('Random Search CV')
        else:
            gs = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                cv=outer_cv,
                scoring='accuracy',
                n_jobs=-1
            )
            print('Grid Search CV')

        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        # train_acc = gs.score(X_train, y_train)
        # test_acc  = gs.score(X_test,  y_test)
        # print(f"Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

        # 4) Predict on test set
        y_pred = best.predict(X_test)

        # 5) Build a stateful signal series
        #   +1 = enter long, -1 = enter short, 0 = hold
        sig = pd.Series(0.0, index=feat.index)
        position = 0
        for t in X_test.index:
            pred = y_pred.loc[t]
            if position == 0:
                # long if pred > 0, short if pred < 0
                position = 1 if pred > 0 else -1
                sig.at[t] = position
            else:
                # exit to flat when prediction changes sign or is near zero
                if (position == 1 and pred <= 0) or (position == -1 and pred >= 0):
                    sig.at[t] = -position  # flip back to zero state
                    position = 0

        # 6) Merge signals back into full df
        df['signal'] = sig.reindex(df.index).fillna(0.0).astype(int)
        return df