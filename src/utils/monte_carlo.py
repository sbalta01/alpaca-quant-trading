# src/utils/monte_carlo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto

def compute_var_cvar(terminal: np.ndarray, #Shape (n_sims,)
                      alpha: float = 0.05,
                      port_0 = 1.0) -> dict:
    """
    Given an array of simulated portfolio end-values (shape (n_sims,) or (n_sims, n_assets)),
    compute VaR and CVaR at level alpha on the returns:

    Parameters
    ----------
    ending_values : array_like
        Simulated end values.  If shape is (n_sims, n_assets), this flattens to 1D.
    alpha : float
        Tail probability.  e.g. alpha=0.05 means 95% VaR/CVaR.
    """
    # compute losses relative to initial value S0 = vals[0] or user‑supplied
    # if you have a known S0, you can pass it in.  For simplicity we assume S0=1.0:
    returns = terminal/port_0 - 1

    # 1) VaR is the α‑quantile of the loss distribution
    var = np.quantile(returns, alpha)  # = smallest loss L such that P(Loss ≤ L) ≥ α

    # 2) CVaR is the *conditional* average of the worst-α tail
    tail_returns = returns[returns <= var]  # those losses <= VaR (losses are negative)
    cvar = tail_returns.mean() if len(tail_returns) > 0 else var
    return {"VaR": float(var),
            "CVaR": float(cvar),
            "alpha": alpha}

def compute_evt_var_cvar(terminal: np.ndarray,
                          alpha: float = 0.05,
                          threshold_q: float = 0.90,
                          port_0 = 1.0,) -> dict:
    
    """
    Fit a GPD to the worst (1 - threshold_q)% of losses, then
    extrapolate VaR/CVaR at tail probability alpha. EVT: extrem value theory

    returns: array of simulated returns (r = S_T/S0 - 1)
    alpha: tail-probability for VaR/CVaR (e.g. 0.05)
    threshold_q: quantile to define the fitting threshold (e.g. 0.90). I.e., above which percentile to pick the fitting data.
    """
    # 1) build losses so that larger = worse
    returns = terminal/port_0 - 1

    n = len(returns)
    # 1) Threshold u: the threshold_q‐quantile of returns
    u = np.quantile(returns, 1 - threshold_q)  # e.g. for threshold_q=0.90, u is 10th percentile

    # 2) Excesses: how far below u each return lies
    excess = u - returns[returns < u]          # positive values
    k = len(excess)
    p_exc = k / n                              # empirical prob of exceeding u

    if k < 5:
        print("Not enough tail points to fit GPD - lower threshold_q. Returning NaN")
        return {"VaR_evt": np.nan,
            "CVaR_evt": np.nan,
            "alpha": alpha}

    # 3) Fit GPD to those excesses
    xi, loc, beta = genpareto.fit(excess, floc=0)

    # 4) EVT‐VaR in **return** terms solves
    #    P(r <= VaR_evt) = α = p_exc * GPD.cdf(u - VaR_evt)
    #  ⇒ VaR_evt = u - (β/ξ)*((p_exc/α)**ξ - 1)
    var_evt = u - (beta/xi) * ((p_exc/alpha)**xi - 1)

    # 5) EVT‐CVaR: conditional mean of r ≤ VaR_evt
    if xi < 1:
        cvar_evt = u - (beta + (beta/xi) * ((p_exc/alpha)**xi - 1)) / (1 - xi)
    else:
        cvar_evt = np.nan  # infinite tail mean

    return {"VaR_evt": float(var_evt),
            "CVaR_evt": float(cvar_evt),
            "alpha": alpha}


def plot_monte_carlo_results(paths: np.ndarray,
                            empirical: dict,
                            evt: dict,
                            symbol: str, 
                            times: list, 
                            final_price: float = None, 
                            ):
    """
    Plots sample Monte Carlo price paths and histogram of terminal returns with VaR/CVaR lines.
    
    Parameters
    ----------
    paths : np.ndarray, shape (n_steps, n_sims)
        Simulated price paths (rows=time, columns=simulations).
    var : float
        Value at Risk at the given confidence level (as a return, e.g. -0.10 for -10%).
    cvar : float
        Conditional VaR (expected return in the worst (1 - confidence_level)%).
    confidence_level : float
        The confidence level used for VaR/CVaR (e.g., 0.95).
    """
    n_steps, n_sims = paths.shape

    var = empirical["VaR"]
    cvar = empirical["CVaR"]
    alpha_empirical = empirical["alpha"]
    var_evt = evt["VaR_evt"]
    cvar_evt = evt["CVaR_evt"]
    alpha_evt = evt["alpha"]

    # 1) Sample paths
    plt.figure()
    mean_path = np.mean(paths[:, :]/paths[0,:] - 1, axis = 1)
    sigma2_pos_path = np.quantile(paths[:, :]/paths[0,:] - 1, axis = 1, q = 0.95)
    sigma2_neg_path = np.quantile(paths[:, :]/paths[0,:] - 1, axis = 1, q = 0.05)
    for i in np.random.randint(0, n_sims, 100):
        plt.plot(times, (paths[:, i]/paths[0,i] - 1)*100)
    plt.plot(times, mean_path *100, '--k', ms = 10,label = "Average path")
    plt.plot(times, sigma2_pos_path *100, '--k', ms = 10,label = "95% percentile path")
    plt.plot(times, sigma2_neg_path*100, '--k', ms = 10, label = "5% percentile path")
    if final_price is not None:
        plt.axhline((final_price/paths[0,0] - 1)*100, color = 'navy',label = "Historical return")
    plt.title(f"Sample Monte Carlo Price Paths in {n_steps-1} steps of {symbol}")
    plt.xlabel("Time step")
    plt.ylabel("Simulated Total Return (%)")
    plt.legend()
    # plt.show()
    
    # 2) Histogram of terminal returns with risk metrics
    terminal_returns = paths[-1,:] / paths[0,:] - 1
    plt.figure()
    _, bins, patches = plt.hist(terminal_returns*100, bins=25, color = 'steelblue')
    
    plt.axvline(var*100, linestyle='--', color = 'k',label=f"Day-scaled VaR ({int((1 - alpha_empirical)*100)}%) = {var/np.sqrt(n_steps-1):.2%}")
    plt.axvline(cvar*100, linestyle='-', color = 'k', label=f"Day-scaled CVaR ({alpha_empirical:0.1%} tail) = {cvar/np.sqrt(n_steps-1):.2%}")

    plt.axvline(var_evt*100, linestyle='--', color = 'grey',label=f"Day-scaled VaR_EVT ({int((1 - alpha_evt)*100)}%) = {var_evt/np.sqrt(n_steps-1):.2%}")
    plt.axvline(cvar_evt*100, linestyle='-', color = 'grey', label=f"Day-scaled CVaR_EVT ({alpha_evt:0.1%} tail) = {cvar_evt/np.sqrt(n_steps-1):.2%}")
    if final_price is not None:
        plt.axvline((final_price/paths[0,0] - 1)*100, color = 'navy',label = "Historical return")

    # find bins in the worst α tail
    for rect, edge in zip(patches, bins):
        if edge <= var*100:
            rect.set_facecolor('C3')  # tail shaded
    
    plt.title(f"Distribution of Terminal Returns in {n_steps-1} steps of {symbol}")
    plt.xlabel("Returns (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def monte_carlo_portfolio_risk(
    price_hist: pd.DataFrame,
    T: float,
    dt: float = 1/252,
    n_sims: int = 10_000,
    alpha_empirical: float = 0.05,
    alpha_evt: float = 0.01,
    trials: int = 1,
    random_state: int = 42,
    backtest : bool = False,
    normalize: bool = True,
    max_weight: float = None,  # e.g. 0.25 to cap any asset at 25%
):
    """
    End-to-end Monte Carlo GBM risk estimator.
    price_hist: DataFrame of historical prices, columns=assets
    dt: time step (default daily)
    n_sims: number of Monte Carlo paths
    alpha: VaR confidence level
    random_state: for reproducibility
    Returns dict with keys:
      - "paths": simulated portfolio paths (n_steps, n_sims)
      - "VaR", "CVaR"
    """
    if isinstance(price_hist.index, pd.MultiIndex):
        # expect level 0='symbol', level 1='timestamp'
        # and a column named 'close'
        price_hist = price_hist["close"].unstack(level="symbol")
    price_hist = price_hist.dropna() #Avoid missing prices
    
    n_steps = int(T / dt)

    if backtest:
        if len(price_hist) < n_steps:
            raise Exception("Need to increase the number of datapoints or quit backtesting")
        logrets = np.log(price_hist / price_hist.shift(1)).iloc[:-n_steps].dropna() #Use only up until -n_steps timesteps to compute GBM parameters
        start_prices = price_hist.iloc[-n_steps].values
        times = price_hist.index[-n_steps:]
    else:
        logrets = np.log(price_hist / price_hist.shift(1)).dropna() #Use all data to compute GBM parameters
        start_prices = price_hist.iloc[-1].values
        times = range(n_steps)

    assets  = price_hist.columns.tolist()
    n_assets= len(assets)

    def run(r, position):
        S0_vec = start_prices * position

        mus     = logrets.mean().values / dt + 0.5 * logrets.var().values / dt
        sigmas  = logrets.std().values / np.sqrt(dt)

        corr    = np.corrcoef(logrets.values, rowvar=False)
        corr    = np.atleast_2d(corr)       # now shape = (1,1) when single asset
        eps  = 1e-6
        corr[np.diag_indices_from(corr)] += eps #Avoid non-positive definite matrices
        L       = np.linalg.cholesky(corr)             # for correlating normals

        assets  = price_hist.columns.tolist()
        n_assets= len(assets)
        port_0  = S0_vec.sum()

        ### 
        drift = (mus - 0.5 * sigmas**2) * dt            # shape=(n_assets,)
        vol   = sigmas * np.sqrt(dt)

        # container: (time, sims, assets)
        all_paths = np.zeros((n_steps, n_sims, n_assets))
        all_paths[0,:,:] = start_prices #Simulate prices (not positions)

        for t in range(1, n_steps):
            # uncorrelated draws
            Z = r.randn(n_sims, n_assets)
            # impose correlation
            Zc = Z @ L.T
            # Zc = Z
            # exponent increment per asset
            inc = drift + vol * Zc
            # update
            all_paths[t,:,:] = all_paths[t-1,:,:] * np.exp(inc)

        # 3) Portfolio value paths
        port_paths = (all_paths * position[None,None,:]).sum(axis=2) #Combine scaled prices (based on positions)

        # Compute terminal returns & risk metrics
        port_terminal = port_paths[-1,:]

        empirical = compute_var_cvar(port_terminal, alpha=alpha_empirical, port_0 = port_0)
        evt = compute_evt_var_cvar(port_terminal, alpha= alpha_evt, threshold_q=0.90, port_0 = port_0)
        
        return port_paths, empirical, evt
    
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**16 - 1, size=trials)
    iterative_vars = []
    positions = []
    for i, seed in enumerate(seeds):
        r = np.random.RandomState(seed)
        position = r.uniform(low = 0, high = 1, size=n_assets)
        position = position/np.sum(position) #Normalize as percentage of portfolio
        
        _, empirical, _ = run(r, position)
        
        iterative_vars.append(empirical["VaR"])
        positions.append(position)

    iterative_vars = np.array(iterative_vars)
    best_arg = np.argmax(iterative_vars)
    best_position = positions[best_arg]

    best_port_paths, best_empirical, best_evt = run(np.random.RandomState(seeds[best_arg]), best_position)

    final_price = (price_hist.iloc[-1].values * best_position).sum() if backtest else None #Save last price for benchmarking (only if backtesting)
    
    # Plot portfolio results
    plot_monte_carlo_results(
        paths=best_port_paths,
        empirical=best_empirical,
        evt=best_evt,
        symbol=assets,
        times = times,
        final_price=final_price,
    )

    print('Best position', best_position)
    print(f'VaR {best_empirical["VaR"]:0.2%}')

    return {
        "paths":   best_port_paths,
        "empirical":best_empirical,
        "evt":      best_evt,
        "position": best_position,
        "times":    times
    }