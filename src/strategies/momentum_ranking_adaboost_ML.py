# src/strategies/momentum_ranking_adaboost_ML.py

import pandas as pd
import numpy as np
from typing import Tuple
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy


class MomentumRankingAdaBoostStrategy(Strategy):
    """
    Cross‐sectional ranking of a universe by next‐bar MA(d) up‐move probability,
    as predicted by AdaBoostMAPredictor, generating a full signal series:
      - signal = +1 for the top_k symbols on each date
      - signal =  0 otherwise
    """
    name = "MomentumRankingAdaBoost"
    multi_symbol = True

    def __init__(
        self,
        predictor: AdaBoostStrategy,
        top_k: int = 10,
        n_jobs: int = -1,
    ):
        """
        predictor : a configured AdaBoostMAPredictor
        top_k     : how many symbols to go long each day
        n_jobs    : parallel jobs for probability estimation
        """
        self.predictor = predictor
        self.top_k = top_k
        self.n_jobs = n_jobs

    def _predict_prob(
        self,
        symbol: str,
        df_sym: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Train & tune on history up to the last row of df_sym,
        then return (symbol, prob_up_next_bar).
        """
        try:
            # Build features & target
            feat = self.predictor._compute_features(df_sym)
            feat["target"] = np.sign(
                feat[f"MA{self.predictor.d}"].shift(-1) -
                feat[f"MA{self.predictor.d}"]
            )
            feat = feat.dropna(subset=["target"])
            split = int(len(feat) * self.predictor.train_frac)
            train = feat.iloc[:split]
            test  = feat.iloc[split:]

            X_train = train.drop(columns=["open","high","low","close","volume","target"])
            y_train = train["target"].astype(int)
            # predict for the very last available row:
            X_pred = test.drop(columns=["open","high","low","close","volume","target"]).iloc[[-1]]

            # time-series CV grid search
            tscv = TimeSeriesSplit(n_splits=self.predictor.cv_splits)
            gs = GridSearchCV(
                self.predictor.pipeline,
                self.predictor.param_grid,
                cv=tscv,
                scoring="accuracy",
                n_jobs=1
            )
            gs.fit(X_train, y_train)
            prob_up = gs.best_estimator_.predict_proba(X_pred)[0,1]
            return symbol, float(prob_up)
        except Exception:
            return symbol, 0.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data : MultiIndex ['symbol','timestamp'] with price & volume
        returns: same MultiIndex with 'signal' column for every timestamp
        """
        df = data.copy()
        if not isinstance(df.index, pd.MultiIndex) \
           or df.index.names != ["symbol","timestamp"]:
            raise ValueError("Index must be MultiIndex ['symbol','timestamp'].")

        symbols    = df.index.get_level_values("symbol").unique()
        timestamps = sorted(df.index.get_level_values("timestamp").unique())

        all_positions = []

        for t in timestamps:
            # slice history up to t
            hist = df.loc[pd.IndexSlice[:, :t], :]
            # build tasks for symbols that have data at t
            tasks = []
            for sym in symbols:
                df_sym = hist.xs(sym, level="symbol")
                if len(df_sym) >= self.predictor.d + 1:
                    tasks.append((sym, df_sym))

            # parallel probability estimates
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._predict_prob)(sym, df_sym)
                for sym, df_sym in tasks
            )
            probs = dict(results)

            # rank and pick top_k
            ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_syms = {sym for sym, _ in ranked[: self.top_k]}

            # build signals at date t
            idx = pd.MultiIndex.from_product(
                [symbols, [t]],
                names=["symbol","timestamp"]
            )
            df_t = pd.DataFrame(index=idx)
            df_t['position'] = [1.0 if sym in top_syms else 0.0 for sym, _ in idx]
            all_positions.append(df_t)
        df['position'] = pd.concat(all_positions).sort_index()
        df['signal'] = df.groupby(level='symbol')['position'].diff().fillna(0)
        return df
