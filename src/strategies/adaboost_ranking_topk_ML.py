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
        df: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Train the predictor on symbol's history and return (symbol, prob_up).
        """
        # Predictor expects a single‐symbol DataFrame indexed by timestamp
        try:
            # generate_signals normally returns a DataFrame with a 'signal' column,
            # but we need its probability. So we replicate its predict logic here:
            feat = self.predictor._compute_features(df)
            feat["target"] = np.sign(
                feat[f"MA{self.predictor.d}"].shift(-1) - feat[f"MA{self.predictor.d}"]
            )
            feat = feat.dropna(subset=["target"])
            split = int(len(feat) * self.predictor.train_frac)
            train = feat.iloc[:split]
            test  = feat.iloc[split:]
            X_train = train.drop(columns=["open","high","low","close","volume","target"])
            y_train = train["target"].astype(int)
            X_pred = test.drop(columns=["open","high","low","close","volume","target"]).iloc[[-1]]

            # grid‐search + fit
            from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
            tscv = TimeSeriesSplit(n_splits=self.predictor.cv_splits)
            gs = GridSearchCV(
                self.predictor.pipeline,
                self.predictor.param_grid,
                cv=tscv,
                scoring="accuracy",
                n_jobs=1  # nested parallelism can be problematic
            )
            gs.fit(X_train, y_train)

            prob = gs.best_estimator_.predict_proba(X_pred)[0,1]
            return symbol, float(prob)
        except Exception:
            # On any error, treat as zero confidence
            return symbol, 0.0

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        data : MultiIndex DataFrame ['symbol','timestamp'] → price & other cols.
        Returns a DataFrame with the same MultiIndex, but only at the latest timestamp:
          - 'signal' = 1.0 for top_k symbols by predicted prob_up
          - 'signal' = 0.0 otherwise
        """
        if not isinstance(data.index, pd.MultiIndex) or \
           list(data.index.names) != ["symbol","timestamp"]:
            raise ValueError("Data must have a MultiIndex ['symbol','timestamp'].")

        # Determine latest timestamp across all symbols
        latest_ts = data.index.get_level_values("timestamp").max()

        # Extract per‐symbol history
        grouped = data.groupby(level="symbol").apply(lambda df: df.droplevel("symbol"))

        # Compute probabilities in parallel
        tasks = [
            (symbol, df_sym)
            for symbol, df_sym in grouped.items()
            if df_sym.index.max() >= latest_ts  # ensure data covers latest date
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_symbol_prob)(symbol, df_sym)
            for symbol, df_sym in tasks
        )

        # Rank and select top_k
        probs = dict(results)
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_syms = {sym for sym, _ in ranked[: self.top_k]}

        # Build output index: all symbols at latest_ts
        symbols = list(probs.keys())
        idx = pd.MultiIndex.from_product(
            [symbols, [latest_ts]],
            names=["symbol","timestamp"]
        )
        out = pd.DataFrame(index=idx)
        out["signal"] = [1.0 if sym in top_syms else 0.0 for sym, _ in idx]

        return out