# src/strategies/momentum_ranking_adaboost_ML.py

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy

class MomentumRankingAdaBoostStrategy(Strategy):
    """
    Train each symbol's AdaBoostMAPredictor once, then at each test timestamp
    rank symbols by predicted up‐move probability and go long the top_k.
    """
    name = "MomentumRankingAdaBoost"
    multi_symbol = True

    def __init__(self, predictor: AdaBoostStrategy, top_k: int = 10):
        self.predictor = predictor
        self.top_k = top_k
        self.train_frac = self.predictor.train_frac
        self.param_grid = self.predictor.param_grid
        self.pipeline = self.predictor.pipeline

    def _fit_and_predict(self, df_sym: pd.DataFrame) -> pd.Series:
        feat = self.predictor._compute_features(df_sym)
        feat["target"] = np.sign(
            feat[f"MA{self.predictor.d}"].shift(-1) - feat[f"MA{self.predictor.d}"]
        )
        feat = feat.dropna(subset=["target"])

        split = int(len(feat) * self.predictor.train_frac)
        train = feat.iloc[:split]
        test = feat.iloc[split:]

        X_train = train.drop(columns=["open", "high", "low", "close", "volume", "target"])
        y_train = train["target"].astype(int)
        X_test = test.drop(columns=["open", "high", "low", "close", "volume", "target"])

        outer_cv = TimeSeriesSplit(n_splits=self.predictor.cv_splits)
        
        if isinstance(self.param_grid, dict) and all(
            hasattr(v, "rvs") for v in self.param_grid.values()
        ):
            gs = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=self.param_grid,
                n_iter=self.predictor.n_iter_search,
                cv=outer_cv,
                scoring='accuracy',
                random_state=self.predictor.random_state,
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
        probs = pd.Series(
            gs.best_estimator_.predict_proba(X_test)[:, 1],
            index=test.index
        )
        return probs

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ["symbol", "timestamp"]:
            raise ValueError("Index must be MultiIndex ['symbol','timestamp'].")

        df = data.copy()
        symbols = df.index.get_level_values("symbol").unique()
        probs_df = []

        # Compute probs for each symbol (serially)
        for sym in symbols:
            df_sym = df.xs(sym, level="symbol")
            probs = self._fit_and_predict(df_sym)
            probs_df.append(pd.DataFrame({"symbol": sym, "timestamp": probs.index, "prob": probs.values}))

        # Combine all probabilities into a single DataFrame
        probs_all = pd.concat(probs_df).set_index(["symbol", "timestamp"])
        probs_all = probs_all.sort_index()

        # Rank by timestamp
        prob_pivot = probs_all.reset_index().pivot(index="timestamp", columns="symbol", values="prob")
        top_k_mask = prob_pivot.apply(lambda row: row.nlargest(self.top_k).index, axis=1)

        # Build MultiIndex for position signals
        idx = pd.MultiIndex.from_product([symbols, prob_pivot.index], names=["symbol", "timestamp"])
        position = pd.Series(0.0, index=idx)

        for t, top_syms in top_k_mask.items():
            position.loc[(list(top_syms), t)] = 1.0

        # Build final DataFrame
        df["position"] = position.sort_index().reindex(df.index).fillna(0.0)
        df["signal"] = df.groupby(level="symbol")["position"].diff().fillna(0.0)
        return df
