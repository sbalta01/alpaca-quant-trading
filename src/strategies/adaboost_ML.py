# src/strategies/adaboost_ML.py

from scipy.stats import randint, uniform
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.strategies.base_strategy import Strategy
from src.utils.indicators import remove_outliers, sma, ema, rsi

class AdaBoostStrategy(Strategy):
    """
    Predict the sign of ΔMA(d) = MA(d)_{t+1} - MA(d)_t using AdaBoost + GridSearchCV.
    Features: 32 from your table (price, volume, MA, EMA, RSI, OBV, ROC, MACD, Stoch, CCI).
    Target: sign of ΔMA(d) for d in {5,10,20}.
    """
    name = "AdaBoost"
    multi_symbol = False

    def __init__(
        self,
        d: int = 5,
        train_frac: float = 0.7,
        cv_splits: int = 5,
        param_grid: Dict[str, Any] = None,
        random_state: int = 42,
        ratio_outliers: float = np.inf,
        n_iter_search: int = 50
    ):
        """
        Parameters
        ----------
        d          : int
            MA window to predict (5, 10, or 20).
        train_frac : float
            Fraction of data to train on.
        cv_splits  : int
            Number of folds for time-series CV.
        param_grid : dict
            Grid for GridSearchCV. Defaults to
            {'clf__n_estimators':[50,100], 'clf__learning_rate':[0.5,1.0]}.
        """
        if d not in (5, 10, 20):
            raise ValueError("d must be one of 5, 10, 20")
        self.d = d
        self.train_frac = train_frac
        self.cv_splits = cv_splits
        self.param_grid = param_grid or {
            'clf__n_estimators': randint(50, 500),
            'clf__learning_rate': uniform(0.01, 2.0)
        }
        self.random_state = random_state
        self.ratio_outliers = ratio_outliers
        self.n_iter_search = n_iter_search

        #Pipeline: scale → RFECV(AdaBoost) (with inner CV) → final estimator
        inner_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', RFECV(
                estimator=AdaBoostClassifier(random_state=self.random_state),
                step=0.1,            # remove 10% features each iteration
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )),
            ('clf', AdaBoostClassifier(random_state=self.random_state))
        ])

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 32 features from your table."""
        df = df.copy()
        # Basic price & lag
        df['open'] = df['open']
        df['high'] = df['high']
        df['low']  = df['low']
        df['close'] = df['close']
        df['close_1'] = df['close'].shift(1)
        df['close_inc'] = df['close'] - df['close_1']

        # Volume features
        df['volume'] = df['volume']
        df['volume_1'] = df['volume'].shift(1)
        df['volume_inc'] = df['volume'] - df['volume_1']

        # Moving Averages
        for w in (5,10,20):
            df[f'MA{w}'] = sma(df['close'], w)
            df[f'MA{w}_1'] = df[f'MA{w}'].shift(1)
            df[f'MA{w}_inc'] = df[f'MA{w}'] - df[f'MA{w}_1']

        # EMAs
        for w in (5,10,20):
            df[f'EMA{w}'] = ema(df['close'], w)

        # RSI
        df['RSI'] = rsi(df['close'], 12)

        # OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # ROC
        for w in (5,10,20):
            df[f'ROC{w}'] = df['close'].pct_change(w) * 100

        # MACD (12,26) + Signal(9)
        df['EMA12'] = ema(df['close'], 12)
        df['EMA26'] = ema(df['close'], 26)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACDsignal'] = ema(df['MACD'], 9)
        df['MACDhist']   = df['MACD'] - df['MACDsignal']

        # Stochastic Oscillator %K(3), %D(3)
        low_min  = df['low'].rolling(3).min()
        high_max = df['high'].rolling(3).max()
        df['slowk'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['slowd'] = df['slowk'].rolling(3).mean()

        # CCI(10)
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(10).mean()
        mean_dev = tp.rolling(10).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI'] = (tp - ma_tp) / (0.015 * mean_dev)

        return df.dropna() #Because of this there is a slight mismatch between start control and the start of the ML strat

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # 1) Build features & target
        feat = self._compute_features(df)
        # target = sign of ΔMA(d)
        feat['target'] = (feat[f'MA{self.d}'].shift(-1) > feat[f'MA{self.d}']).astype(int)
        feat = feat.dropna() #Because of this there is a slight mismatch between start control and the start of the ML strat
        feat['target'] = feat['target'].astype(int)

        # 2) Remove outliers
        feat = remove_outliers(feat, ratio_outliers=self.ratio_outliers)

        # 3) Split train / val / test (no shuffle)
        X = feat.drop(columns=['open','high','low','close','volume','target'])
        y = feat['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_frac, shuffle=False
        )

        # 4) Nested CV Grid Search/Randomized Search CV
        outer_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        if isinstance(self.param_grid, dict) and all(
            hasattr(v, "rvs") for v in self.param_grid.values()
        ):
            # The user did NOT supply a fixed grid then use RandomizedSearchCV
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
            # User supplied a fixed param_grid then exhaustive GridSearch
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

        train_acc = gs.score(X_train, y_train)
        test_acc  = gs.score(X_test,  y_test)
        print(f"Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

        # 5) Evaluate on test
        y_pred = best.predict(X_test)

        # 6) Generate signals: only in test period
        positions = pd.Series(0.0, index=feat.index)
        y_test_series = pd.Series(0.0, index=feat.index)
        test_mask = pd.Series(0.0, index=feat.index)
        idxs = list(X_test.index)
        for idx, pred, y in zip(idxs, y_pred, y_test):
            positions.at[idx] = pred
            y_test_series.at[idx] = y
            test_mask.at[idx] = 1.0
        
        out = df.copy()
        out["position"] = positions.reindex(df.index).fillna(0.0)
        out['signal'] = out['position'].diff().fillna(0.0)
        
        out["y_test"] = y_test_series.reindex(df.index).fillna(0.0)
        out["y_pred"] = out["position"].copy()
        out["test_mask"] = test_mask.reindex(df.index).fillna(0.0)
        return out