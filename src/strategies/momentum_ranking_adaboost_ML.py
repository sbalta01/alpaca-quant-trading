# src/strategies/momentum_ranking_adaboost_ML.py

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML  import AdaBoostStrategy

class MomentumRankingAdaBoostStrategy(Strategy):
    """
    Train each symbol's AdaBoostMAPredictor once, then at each test timestamp
    rank symbols by predicted up‐move probability and go long the top_k.
    """
    name = "MomentumRankingAdaBoost"
    multi_symbol = True

    def __init__(
        self,
        predictor: AdaBoostStrategy,
        top_k: int = 10,
        n_jobs: int = -1
    ):
        self.predictor = predictor
        self.top_k = top_k
        self.n_jobs = n_jobs
        self.train_frac = self.predictor.train_frac

    def _fit_symbol(self, symbol: str, df_sym: pd.DataFrame) -> Tuple[str, List[pd.Timestamp], np.ndarray]:
        """
        Fit on the train slice of df_sym, then return:
          - symbol
          - test timestamps
          - array of predicted probabilities for each test timestamp
        """
        # 1) Build features & target
        feat = self.predictor._compute_features(df_sym)
        feat["target"] = np.sign(
            feat[f"MA{self.predictor.d}"].shift(-1) - feat[f"MA{self.predictor.d}"]
        )
        feat = feat.dropna(subset=["target"])
        # 2) Split train / test
        split = int(len(feat) * self.predictor.train_frac)
        train = feat.iloc[:split]
        test  = feat.iloc[split:]
        # 3) Prepare arrays
        X_train = train.drop(columns=["open","high","low","close","volume","target"])
        y_train = train["target"].astype(int)
        X_test  = test.drop(columns=["open","high","low","close","volume","target"])
        timestamps_test = list(test.index)
        # 4) Grid-search once per symbol
        tscv = TimeSeriesSplit(n_splits=self.predictor.cv_splits)
        gs = GridSearchCV(
            self.predictor.pipeline,
            self.predictor.param_grid,
            cv=tscv,
            scoring="accuracy",
            n_jobs=1
        )
        gs.fit(X_train, y_train)
        # 5) Predict probabilities for all test rows
        probs = gs.best_estimator_.predict_proba(X_test)[:,1]
        return symbol, timestamps_test, probs

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data: MultiIndex ['symbol','timestamp'] → must include 'close','volume',etc.
        Returns full MultiIndex signals over all timestamps and symbols.
        """
        if not isinstance(data.index, pd.MultiIndex) or \
           data.index.names != ["symbol","timestamp"]:
            raise ValueError("Index must be MultiIndex ['symbol','timestamp'].")

        symbols    = data.index.get_level_values("symbol").unique()
        timestamps = sorted(data.index.get_level_values("timestamp").unique())
        # 0) Fit & predict per symbol once, gathering test timestamps & probs
        groups = [(sym, data.xs(sym, level="symbol")) for sym in symbols]
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_symbol)(sym, df_sym) for sym, df_sym in groups
        )
        # 1) Build prob_panels and collect all test timestamps
        prob_panels = {}
        test_ts_all = set()
        for sym, ts_list, probs in fitted:
            prob_panels[sym] = pd.Series(probs, index=ts_list)
            test_ts_all.update(ts_list)

        # 2) At each timestamp, rank symbols by prob_up and mark top_k
        idx_tuples = []
        all_positions = []
        for t in timestamps:
            # only generate non-zero signals if t is in the test period
            if t not in test_ts_all:
                # pre‐test: everyone flat
                for sym in symbols:
                    idx_tuples.append((sym, t))
                    all_positions.append(0.0)
            else:
                # build cross‐section for t
                row = {sym: prob_panels[sym].get(t, 0.0) for sym in symbols}
                # rank
                ranked = sorted(row.items(), key=lambda x: x[1], reverse=True)
                top_syms = {sym for sym, _ in ranked[: self.top_k]}

                # record signals
                for sym in symbols:
                    idx_tuples.append((sym, t))
                    all_positions.append(1.0 if sym in top_syms else 0.0)

        idx = pd.MultiIndex.from_tuples(idx_tuples, names=["symbol","timestamp"])
        out = pd.DataFrame(index=idx)
        out["position"] = all_positions
        data['position'] = out["position"]
        data['signal'] = data.groupby(level='symbol')['position'].diff().fillna(0)
        return data