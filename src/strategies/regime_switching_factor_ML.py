# src/strategies/regime_switching_factor.py

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy
from src.strategies.hybrid_adaboost_filter_ML import HybridAdaBoostFilterStrategy


class RegimeSwitchingFactorStrategy(Strategy):
    """
    Regime-Switching Factor Investing via Hidden Markov Model:
      - Fit a 3-state GaussianHMM on [return, volatility] every day
      - Label states {0,1,2} --> {bear, sideways, bull}
      - For each new day, detect regime by PDF thresholding
      - Dispatch to the corresponding factor model strategy to get signals
    """
    name = "RegimeSwitch"
    multi_symbol = True

    def __init__(
        self,
        regime_symbol: str,
        hmm_states: int = 3,
        vol_window: int = 10,
        vol_thresh: float = 0.3,
        ret_thresh: float = 0.5,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        regime_symbol : str
            Ticker in your data used to fit the HMM (e.g. "SPY").
        hmm_states    : int
            Number of hidden regimes (3 = bear/sideways/bull).
        hmm_window    : int
            Lookback (in days) to refit HMM each day.
        vol_window    : int
            Window for volatility proxy (MSE on a rolling MA).
        vol_threshold : float
            Minimum PDF(confidence) for volatility to assign a regime.
        ret_threshold : float
            Minimum PDF(confidence) for return to assign a regime.
        """
        self.regime_symbol = regime_symbol
        self.n_states = hmm_states
        self.vol_window = vol_window
        self.vol_thresh = vol_thresh
        self.ret_thresh = ret_thresh
        self.random_state = random_state

        predictor = AdaBoostStrategy(
            d=10,
            train_frac=0.7,
            cv_splits=5,
            param_grid={
                'clf__n_estimators': [50,100,200],
                'clf__learning_rate': [0.1,0.5,1.0]
            },
            # ratio_outliers = 3.00,
            n_iter_search = 50,
            random_state=random_state
        )
        self.value_model = HybridAdaBoostFilterStrategy(
            predictor=predictor,
            short_ma=9,
            long_ma=20,
            angle_threshold_deg=10,
            atr_window=14,
            vol_threshold=0.01
        )

        self._scaler = StandardScaler()
        self.train_frac = predictor.train_frac

    def _compute_observables(self, df):
        """
        On df (DatetimeIndex), compute:
         - 'ret' = daily return
         - 'vol' = MSE of (close - MA(close)) over vol_window
        """
        ret = df["close"].pct_change().fillna(0.0)
        ma  = df["close"].rolling(self.vol_window).mean()
        mse = ((df["close"] - ma) ** 2).rolling(self.vol_window).mean().fillna(0.0)
        return pd.DataFrame({"ret": ret, "vol": mse})

    def _fit_hmm(self, X):
        """Fit a Gaussian HMM."""
        Xs = self._scaler.fit_transform(X)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000, #Maybe 200 for speed
            random_state=self.random_state
        )
        model.fit(Xs)
        return model

    def _label_states(self, model):
        """
        Given a fitted HMM, sort its states by their
        expected return mean to assign labels.
        Returns mapping {state_index --> "bear"/"sideways"/"bull"}.
        """
        means = model.means_[:, 0]
        order = np.argsort(means)
        return {order[0]: "bear", order[1]: "sideways", order[2]: "bull"}

    def _detect_regime(
        self,
        model: GaussianHMM,
        labels: Dict[int,str],
        obs_today: np.ndarray
    ) -> str:
        """
        Compute each state's PDF for today's obs and pick the
        regime whose probability exceeds both thresholds.
        """
        # scale today's obs the same way
        x = self._scaler.transform(obs_today.reshape(1, -1))
        # compute per‐state weighted PDF
        probs = np.array([
            np.exp(model._compute_log_likelihood(x)[0, s])
            for s in range(self.n_states)
        ])
        # normalize
        probs = probs / probs.sum()
        # find the most likely state
        s = np.argmax(probs)
        if probs[s] >= self.ret_thresh and probs[s] >= self.vol_thresh:
            return labels[s]
        # fallback: choose the highest anyway
        return labels[s]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data : MultiIndex ['symbol','timestamp'] --> must include price series
        Returns the consolidated signals from the selected factor model.
        """
        df = data.copy()
        symbols = df.index.get_level_values("symbol").unique()

        # 1) Precompute value‐model signals for every symbol
        def _compute_value_signals(df_sym):
            return self.value_model.generate_signals(df_sym)

        value_signals = df.groupby(level="symbol") \
                            .apply(lambda grp: _compute_value_signals(grp.droplevel("symbol")))

        # 2) Prepare regime‐symbol observables
        df_reg = value_signals.xs(self.regime_symbol, level="symbol")
        obs = self._compute_observables(df_reg)

        out = pd.DataFrame(index=df.index)
        out["position"] = 0.0

        # 3) Loop over t to detect regime & pick signals
        for t in df_reg.loc[df_reg['test_mask']==1].index:
            window = obs.loc[:t]
            hmm    = self._fit_hmm(window.values)
            labels = self._label_states(hmm)
            regime = self._detect_regime(hmm, labels, obs.loc[t].values)

            # 3) select factor model and get that day's signal slice
            if regime == "bull":
                pos_slice = value_signals.xs(t, level="timestamp")["position"]

            elif regime == "bear":
                pos_slice = value_signals.xs(t, level="timestamp")["position"]
                # continue
            else:  # sideways
                # pos_slice = value_signals.xs(t, level="timestamp")["position"]
                continue #Doing nothing leaves the position at 0

            # Vectorized assignment
            idx = pd.MultiIndex.from_arrays([pos_slice.index, [t] * len(pos_slice)],
                                names=["symbol", "timestamp"])
            out.loc[idx, "position"] = pos_slice.values


        # 4) Return
        df["position"] = out["position"].fillna(0.0)
        df["signal"] = df["position"].diff().fillna(0.0)
        return df
