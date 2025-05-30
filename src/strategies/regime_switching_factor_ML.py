# src/strategies/regime_switching_factor.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.strategies.base_strategy import Strategy
from src.strategies.adaboost_ML import AdaBoostStrategy
# from src.strategies.fama_french import FamaFrenchStrategy  # assume exists

class RegimeSwitchingFactorStrategy(Strategy):
    """
    Regime‐Switching Factor Investing via Hidden Markov Model:
      - Fit a 3‐state GaussianHMM on [return, volatility] every day
      - Label states {0,1,2} → {bear, sideways, bull}
      - For each new day, detect regime by PDF thresholding
      - Dispatch to the corresponding factor model strategy to get signals
    """
    name = "RegimeSwitch"
    multi_symbol = True

    def __init__(
        self,
        regime_symbol: str,
        hmm_states: int = 3,
        hmm_window: int = 2707,
        vol_window: int = 10,
        return_col: str = "close",
        vol_threshold: float = 0.3,
        ret_threshold: float = 0.5,
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
        self.hmm_window = hmm_window
        self.vol_window = vol_window
        self.vol_thresh = vol_threshold
        self.ret_thresh = ret_threshold
        self.random_state = random_state

        # instantiate your factor‐model strategies
        self.value_model = AdaBoostStrategy(d=10, random_state=random_state)
        # self.fama_french = FamaFrenchStrategy(...)  # placeholder

        # we’ll reuse one scaler per fit
        self._scaler = StandardScaler()

    def _compute_observables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        On df (DatetimeIndex), compute:
         - 'ret' = daily return
         - 'vol' = MSE of (close – MA(close)) over vol_window
        """
        ret = df["close"].pct_change().fillna(0.0)
        ma = df["close"].rolling(self.vol_window).mean()
        mse = ((df["close"] - ma)**2).rolling(self.vol_window).mean().fillna(0.0)
        return pd.DataFrame({"ret": ret, "vol": mse})

    def _fit_hmm(self, obs: np.ndarray) -> GaussianHMM:
        """Fit a Gaussian HMM to the last `hmm_window` observations."""
        # scale for numerical stability
        X = self._scaler.fit_transform(obs)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=self.random_state
        )
        model.fit(X)
        return model

    def _label_states(
        self,
        model: GaussianHMM,
        obs: np.ndarray
    ) -> Dict[int, str]:
        """
        Given a fitted HMM, sort its states by their
        expected return mean to assign labels.
        Returns mapping {state_index → "bear"/"sideways"/"bull"}.
        """
        # retrieve means on the _scaled_ space:
        means = model.means_[:, 0]  # the 'ret' dimension
        order = np.argsort(means)
        # lowest → bear, middle → sideways, highest → bull
        labels = {order[0]: "bear", order[1]: "sideways", order[2]: "bull"}
        return labels

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
        data : MultiIndex ['symbol','timestamp'] → must include price series
        Returns the consolidated signals from the selected factor model.
        """
        # 1) isolate the regime_symbol history
        df_reg = data.xs(self.regime_symbol, level="symbol")
        obs = self._compute_observables(df_reg)

        full_signals = []

        # 2) loop through each timestamp, refit HMM on lookback window
        for t in obs.index[self.hmm_window:]:
            window = obs.loc[:t].iloc[-self.hmm_window :]
            hmm = self._fit_hmm(window.values)
            mapping = self._label_states(hmm, window.values)
            regime = self._detect_regime(hmm, mapping, obs.loc[[t]].values.flatten())

            # 3) select factor model and get that day's signal slice
            if regime == "bull":
                sig_df = self.value_model.generate_signals(
                    data.xs(self.regime_symbol, level="symbol").loc[:t]
                )
            elif regime == "bear":
                # placeholder — replace with real Fama‐French when available
                data.xs(self.regime_symbol, level="symbol").loc[:t]["position","signal"] = 0
                sig_df = data.xs.loc[:t] 
            else:  # sideways
                sig_df = self.value_model.generate_signals(
                    data.xs(self.regime_symbol, level="symbol").loc[:t]
                )

            # extract only the final row at t
            signal_t = sig_df.loc[t, ["signal"]]
            # broadcast that signal to every symbol at t
            symbols = data.index.get_level_values("symbol").unique()
            idx = pd.MultiIndex.from_product(
                [symbols, [t]], names=["symbol","timestamp"]
            )
            row = pd.DataFrame(index=idx)
            row["signal"] = signal_t.values[0]
            full_signals.append(row)

        # 4) concatenate all dates into a full history
        print(full_signals)
        out = pd.concat(full_signals).sort_index()
        return out
