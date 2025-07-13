# src/strategies/adaboost_ML.py

from scipy.stats import randint, uniform
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.strategies.base_strategy import Strategy
from src.utils.tools import remove_outliers, sma, ema, rsi, threshold_adjust

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

        self.model = AdaBoostClassifier(
            random_state=self.random_state
            )

        #Pipeline: scale → RFECV(AdaBoost) (with inner CV) → final estimator
        inner_cv = TimeSeriesSplit(n_splits=self.cv_splits)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', RFECV(
                estimator=self.model,
                step=0.1,            # remove 10% features each iteration
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )),
            ('clf', self.model)
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

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # 1) Build features & target
        feat = self._compute_features(df)
        # target = sign of ΔMA(d)
        feat['target'] = (feat[f'MA{self.d}'].shift(-1) > feat[f'MA{self.d}'])
        feat['target'] = feat['target'].astype(int)
        feat = feat.ffill().dropna()  #Ffill so that we dont lose last rows to dropping Nas.

        # 2) Remove outliers
        feat = remove_outliers(feat, ratio_outliers=self.ratio_outliers)

        # 3) Split train / val / test (no shuffle)
        X = feat.drop(columns=['target'])
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
        y_prob = best.predict_proba(X_test)[:, 1]
        y_prob_train = best.predict_proba(X_train)[:, 1]


        # 5) Final predictions on test history
        auc_train  = roc_auc_score(y_train, y_prob_train)
        auc_test  = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC (Train): {auc_train:.3f}, ROC AUC (Test): {auc_test:.3f}")

        # 6) Generate signals: only in test period
        positions = pd.Series(0.0, index=feat.index)
        y_test_series = pd.Series(0.0, index=feat.index)
        test_mask = pd.Series(0.0, index=feat.index)
        idxs = list(X_test.index)

        self.prob_positive_threshold = 0.7
        # y_prob = pd.Series(y_prob).rolling(3).mean() #Smooth the probabilities to avoid single-day flops
        self.adjust_threshold = False
        adjusted_prob_threshold = threshold_adjust(feat['close_inc'], horizon = self.d, base_threshold = 0.5, max_shift=0.2) if self.adjust_threshold else pd.Series(self.prob_positive_threshold, index = idxs)
        position = 0

        for idx, pred, prob, y in zip(idxs, y_pred, y_prob, y_test):
            if position == 0 and prob >= adjusted_prob_threshold.at[idx]:
                position = 1
            elif position == 1 and prob < adjusted_prob_threshold.at[idx]:
                position = 0 #exit to flat
            positions.at[idx] = position

            # positions.at[idx] = pred
            y_test_series.at[idx] = y
            test_mask.at[idx] = 1.0
        
        out = df.copy()
        out["position"] = positions.reindex(df.index).ffill().bfill().clip(0,1)
        out['signal'] = out['position'].diff().fillna(0.0).clip(-1,1)
        
        out["y_test"] = y_test_series.reindex(df.index).fillna(0.0)
        out["y_pred"] = out["position"].copy()
        out["test_mask"] = test_mask.reindex(df.index).fillna(0.0)
        return out