# src/utils/monte_carlo.py

import numpy as np
import pandas as pd

def estimate_gbm_params(prices: pd.Series, dt: float = 1/252):
    """
    Estimate drift (mu) and vol (sigma) from historical price series.
    prices: daily price series
    dt: timestep in years (default 1 trading day = 1/252)
    Returns (mu, sigma)
    """
    # log‐returns
    rets = np.log(prices / prices.shift(1)).dropna()
    mu    = rets.mean() / dt + 0.5 * rets.var() / dt   # GBM drift
    sigma = rets.std() / np.sqrt(dt)
    return mu, sigma

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_sims: int,
    random_state: int = None
) -> np.ndarray:
    """
    Simulate n_sims GBM paths for time horizon T (in years), step dt.
    Returns an array of shape (n_steps+1, n_sims) including t=0.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_steps = int(T / dt)
    # pre‐allocate
    paths = np.empty((n_steps + 1, n_sims))
    paths[0] = S0
    # Brownian increments
    drift = (mu - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)
    for t in range(1, n_steps + 1):
        z = np.random.randn(n_sims)
        paths[t] = paths[t-1] * np.exp(drift + vol * z)
    return paths

def portfolio_value_paths(
    price_paths: np.ndarray,
    positions: np.ndarray
) -> np.ndarray:
    """
    Given simulated price_paths (n_steps+1, n_sims) and
    positions (n_assets,) gives portfolio value series
    of shape (n_steps+1, n_sims).
    """
    # price_paths: dict of asset→paths or 3‑D array
    # Here assume price_paths is (n_steps+1, n_sims, n_assets)
    # and positions is (n_assets,)
    # Broadcast multiplication & sum over assets
    return np.einsum("tsn,n->ts", price_paths, positions)

def compute_var_cvar(
    ending_values: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Given array of simulated portfolio end‐values (n_sims,),
    compute VaR and CVaR at level alpha.
    Returns dict { 'VaR': ..., 'CVaR': ... }
    """
    losses = np.sort(-ending_values)   # negative values → losses
    idx = int(alpha * len(losses))
    var  = losses[idx]
    cvar = losses[:idx].mean() if idx > 0 else var
    return {"VaR": var, "CVaR": cvar}

def monte_carlo_portfolio_risk(
    price_hist: pd.DataFrame,
    positions: np.ndarray,
    T: float,
    dt: float = 1/252,
    n_sims: int = 10_000,
    alpha: float = 0.05,
    random_state: int = None
) -> dict:
    """
    End‐to‐end Monte Carlo GBM risk estimator.
    price_hist: DataFrame of historical prices, columns=assets
    positions: array of number of shares per asset
    T: horizon in years (e.g. 5 trading days = 5/252)
    dt: time step (default daily)
    n_sims: number of Monte Carlo paths
    alpha: VaR confidence level
    random_state: for reproducibility
    Returns dict with keys:
      - "paths": simulated portfolio paths (n_steps+1, n_sims)
      - "VaR", "CVaR"
    """
    assets = price_hist.columns
    S0 = price_hist.iloc[-1].values           # last prices
    # estimate params per asset
    mus, sigmas = [], []
    for col in assets:
        mu, sigma = estimate_gbm_params(price_hist[col], dt)
        mus.append(mu)
        sigmas.append(sigma)
    mus = np.array(mus)
    sigmas = np.array(sigmas)

    # simulate each asset
    n_steps = int(T / dt)
    all_paths = np.empty((n_steps+1, n_sims, len(assets)))
    for i, (s0, mu, sigma) in enumerate(zip(S0, mus, sigmas)):
        all_paths[..., i] = simulate_gbm_paths(
            S0=s0,
            mu=mu,
            sigma=sigma,
            T=T,
            dt=dt,
            n_sims=n_sims,
            random_state=(None if random_state is None else random_state + i)
        )

    # portfolio value
    port_paths = portfolio_value_paths(all_paths, positions)  # (steps+1,n_sims)

    # risk at horizon
    terminal = port_paths[-1]
    risk = compute_var_cvar(terminal, alpha=alpha)

    return {
        "paths": port_paths,
        "VaR":   risk["VaR"],
        "CVaR":  risk["CVaR"],
        "time_index": np.linspace(0, T, n_steps+1)
    }