# src/strategies/ranking_topk.py

import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from joblib import Parallel, delayed
from datetime import datetime

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy


class RankingTopKStrategy(Strategy):
    """
    Generic ranking strategy:
      - Expects `data` to be a pd.DataFrame with a MultiIndex ['symbol','timestamp'].
      - Uses an AdaBoostMAPredictor to compute next‐bar up‐move probability per symbol.
      - Ranks symbols and sets signal=1 for the top_k at the latest timestamp, else 0.
    """
    multi_symbol: bool = True
    name = "RankingTopK"

    def __init__(
        self,
        predictor: AdaBoostStrategy,
        top_k: int = 10,
        n_jobs: int = -1
    ):
        """
        Parameters
        ----------
        predictor : AdaBoostMAPredictor
          Already‐configured instance (with d, train_frac, cv_splits, param_grid).
        top_k : int
          Number of symbols to go long (signal=1) each run.
        n_jobs : int
          Parallel jobs for computing probabilities via joblib.
        """
        self.predictor = predictor
        self.top_k = top_k
        self.n_jobs = n_jobs

    def _compute_symbol_prob(
        self,
        symbol: str,
        df_sym: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Given full history up to a timestamp for one symbol, 
        return (symbol, prob_up) using the predictor.
        """
        try:
            # features & target setup
            feat = self.predictor._compute_features(df_sym)
            feat["target"] = np.sign(
                feat[f"MA{self.predictor.d}"].shift(-1) -
                feat[f"MA{self.predictor.d}"]
            )
            feat = feat.dropna(subset=["target"])
            split = int(len(feat) * self.predictor.train_frac)
            train = feat.iloc[:split]
            test  = feat.iloc[split:]

            X_train = train.drop(
                columns=["open","high","low","close","volume","target"]
            )
            y_train = train["target"].astype(int)
            X_pred = test.drop(
                columns=["open","high","low","close","volume","target"]
            ).iloc[[-1]]

            # time-series GridSearchCV
            from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
        if not isinstance(data.index, pd.MultiIndex) or \
           list(data.index.names) != ["symbol","timestamp"]:
            raise ValueError("Data must have a MultiIndex ['symbol','timestamp'].")

        # Unique symbols and timestamps
        symbols   = data.index.get_level_values("symbol").unique()
        timestamps = sorted(data.index.get_level_values("timestamp").unique())

        all_frames = []
        # Loop over each timestamp
        for t in timestamps:
            # Slice history up to and including t
            hist = data.loc[pd.IndexSlice[:, :t], :]

            # Prepare per-symbol tasks
            tasks = []
            for sym in symbols:
                df_sym = hist.xs(sym, level="symbol")
                # only if we have enough history
                if len(df_sym) > self.predictor.d:
                    tasks.append((sym, df_sym))

            # Parallel probability estimates
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_symbol_prob)(sym, df_sym)
                for sym, df_sym in tasks
            )
            probs = dict(results)

            # Rank and pick top_k
            ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_syms = {sym for sym, _ in ranked[: self.top_k]}

            # Build signals for this timestamp
            idx = pd.MultiIndex.from_product(
                [symbols, [t]],
                names=["symbol","timestamp"]
            )
            df_t = pd.DataFrame(index=idx)
            df_t["signal"] = [1.0 if sym in top_syms else 0.0 for sym, _ in idx]

            all_frames.append(df_t)

        # Concatenate full signal series
        signals = pd.concat(all_frames).sort_index()
        return signals