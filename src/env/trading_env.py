import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Gymnasium environment for N-stock trading.
    State = [cash, shares_i, prices_i, techs..., macros...]
    Action = vector in [-1,1]^n for each symbol (fractional allocation).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,    # MultiIndex [symbol,timestamp], with 'close' + tech/macro cols
        tech_cols: list,
        macro_cols: list,
        initial_cash: float = 1e6,
        transaction_cost: float = 0.001,
        turbulence_col: str = None,
        turbulence_percentile: float = 0.9,
        render_mode: str = None
    ):
        super().__init__()
        self.render_mode = render_mode

        # symbols & arrays
        self.symbols = price_df.index.get_level_values("symbol").unique().tolist()
        self.n       = len(self.symbols)
        # pivot close price
        self.prices = price_df["close"].unstack(0).values   # shape [T, n]
        # pivot technicals
        self.techs  = price_df[tech_cols].unstack(0).values  # [T, n*len(tech_cols)]
        # pivot macros (once per timestamp)
        self.macros = price_df[macro_cols].groupby(level="timestamp").first().values  # [T, len(macro_cols)]

        # turbulence threshold
        if turbulence_col:
            turb = price_df[turbulence_col].unstack(0).max(axis=1)
            self.turb_thr = np.nanpercentile(turb, turbulence_percentile*100)
            self.turb_series = turb.values
        else:
            self.turb_thr = None
            self.turb_series = None

        self.initial_cash     = initial_cash
        self.transaction_cost = transaction_cost

        # define gymnasium spaces
        obs_dim = 1 + self.n + self.n + self.techs.shape[1] + self.macros.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.n,), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self):
        self.t       = 0
        self.cash    = self.initial_cash
        self.shares  = np.zeros(self.n, dtype=np.float32)
        self.done    = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # [cash] + [shares] + [prices] + [techs] + [macros]
        return np.concatenate([
            [self.cash],
            self.shares,
            self.prices[self.t],
            self.techs[self.t].flatten(),
            self.macros[self.t].flatten()
        ]).astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after done=True")

        prev_obs = self._get_obs()
        # turbulence check
        if self.turb_series is not None and self.turb_series[self.t] > self.turb_thr:
            action = np.zeros_like(action)

        prev_val = self.cash + (self.shares * self.prices[self.t]).sum()

        # target allocations [0,1]
        alloc = (action + 1) / 2
        port_val      = prev_val
        target_values = alloc * port_val
        target_shares = target_values / self.prices[self.t]
        delta_shares  = target_shares - self.shares

        # transaction costs
        cost = np.abs(delta_shares) * self.prices[self.t] * self.transaction_cost
        self.cash   -= (delta_shares * self.prices[self.t]).sum() + cost.sum()
        self.shares  = target_shares

        # advance time
        self.t += 1
        if self.t >= len(self.prices):
            self.done = True

        cur_val = self.cash + (self.shares * self.prices[self.t-1]).sum()
        reward  = cur_val - prev_val

        obs = self._get_obs() if not self.done else np.zeros_like(prev_obs)
        info = {"portfolio_value": cur_val}
        return obs, reward, self.done, False, info

    def render(self):
        pv = self.cash + (self.shares * self.prices[self.t]).sum()
        print(f"Step {self.t}: Cash={self.cash:.2f}, PV={pv:.2f}")