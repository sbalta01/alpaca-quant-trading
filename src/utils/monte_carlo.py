# src/utils/monte_carlo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from scipy.optimize import minimize

class monte_carlo_portfolio_risk():
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
    def __init__(
        self,
        price_hist: pd.DataFrame,
        T: float,
        dt: float = 1/252,
        n_sims: int = 10_000,
        alpha_empirical: float = 0.05,
        alpha_evt: float = 0.01,
        returns_penalty = 1.0,
        volat_penalty: float = 1.0,
        cvar_penalty: float = 1.0,
        random_state: int = 42,
        predict : bool = True,
        optimize_position_sizing: bool = False,
    ):
        self.price_hist = price_hist
        self.T = T
        self.dt = dt
        self.n_sims = n_sims
        self.alpha_empirical = alpha_empirical
        self.alpha_evt = alpha_evt
        self.returns_penalty = returns_penalty
        self.volat_penalty = volat_penalty
        self.cvar_penalty = cvar_penalty
        self.rng = np.random.RandomState(random_state)
        self.predict = predict
        self.optimize_position_sizing = optimize_position_sizing

        if isinstance(self.price_hist.index, pd.MultiIndex):
            self.price_hist = self.price_hist["close"].unstack(level="symbol")

        self.price_hist = self.price_hist.dropna() #Avoid missing prices
    
        self.n_steps = int(self.T / self.dt)

        if self.predict:
            self.logrets = np.log(self.price_hist / self.price_hist.shift(1)).dropna() #Use all data to compute GBM parameters
            self.start_prices = self.price_hist.iloc[-1].values
            self.times = range(self.n_steps)
            self.final_price = None
        else:
            if len(self.price_hist) < self.n_steps:
                raise Exception("The number of data points intended for backtesting is larger than the dataset")
            self.logrets = np.log(self.price_hist / self.price_hist.shift(1)).dropna().iloc[:-self.n_steps] #Use only up until -n_steps timesteps to compute GBM parameters
            self.start_prices = self.price_hist.iloc[-self.n_steps].values
            self.times = self.price_hist.index[-self.n_steps:]
            self.final_price = self.price_hist.iloc[-1].values #Save last price for benchmarking (only if backtesting,ie, not predicting)

        self.assets  = self.price_hist.columns.tolist()
        self.n_assets= len(self.assets)

        if self.n_assets == 1:
            self.optimize_position_sizing = False
            print("Only 1 asset --> No position sizing")

    def simulate_gbm_paths(self,):
        mus     = self.logrets.mean().values / self.dt + 0.5 * self.logrets.var().values / self.dt
        sigmas  = self.logrets.std().values / np.sqrt(self.dt)

        corr    = np.corrcoef(self.logrets.values, rowvar=False)
        corr    = np.atleast_2d(corr)       # now shape = (1,1) when single asset
        corr[np.diag_indices_from(corr)] += 1e-6 #Avoid non-positive definite matrices
        L       = np.linalg.cholesky(corr)             # for correlating normals

        drift = (mus - 0.5 * sigmas**2) * self.dt            # shape=(n_assets,)
        vol   = sigmas * np.sqrt(self.dt)  #These are agnostic of dt

        # container: (time, sims, assets)
        all_paths = np.zeros((self.n_steps, self.n_sims, self.n_assets))
        all_paths[0,:,:] = np.tile(self.start_prices, (self.n_sims, 1)) #Simulate prices (not positions)

        for t in range(1, self.n_steps):
            # uncorrelated draws
            Z = self.rng.randn(self.n_sims, self.n_assets)
            # impose correlation
            Zc = Z @ L.T
            # exponent increment per asset
            inc = drift + vol * Zc
            # update
            all_paths[t,:,:] = all_paths[t-1,:,:] * np.exp(inc)

        return all_paths

    def compute_var_cvar(self, final_returns: np.ndarray,): #Shape (n_sims,)

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
        # 1) VaR is the α‑quantile of the loss distribution
        var = np.quantile(final_returns, self.alpha_empirical)  # = smallest loss L such that P(Loss ≤ L) ≥ α

        # 2) CVaR is the *conditional* average of the worst-α tail
        tail_returns = final_returns[final_returns <= var]  # those losses <= VaR (losses are negative)
        cvar = tail_returns.mean() if len(tail_returns) > 0 else var

        return float(var), float(cvar)
    
    def compute_evt_var_cvar(self,final_returns: np.ndarray,
                            threshold_q: float = 0.90,):
        
        """
        Fit a GPD to the worst (1 - threshold_q)% of losses, then
        extrapolate VaR/CVaR at tail probability alpha. EVT: extrem value theory

        returns: array of simulated returns (r = S_T/S0 - 1)
        alpha: tail-probability for VaR/CVaR (e.g. 0.05)
        threshold_q: quantile to define the fitting threshold (e.g. 0.90). I.e., above which percentile to pick the fitting data.
        """
        # 1) Threshold u: the threshold_q‐quantile of returns
        u = np.quantile(final_returns, 1 - threshold_q)  # e.g. for threshold_q=0.90, u is 10th percentile

        # 2) Excesses: how far below u each return lies
        excess = u - final_returns[final_returns < u]          # positive values
        k = len(excess)
        p_exc = k / self.n_sims                              # empirical prob of exceeding u

        if k < 5:
            print("Not enough tail points to fit GPD - lower threshold_q. Returning NaN")
            return {"VaR_evt": np.nan,
                "CVaR_evt": np.nan,}

        # 3) Fit GPD to those excesses
        xi, loc, beta = genpareto.fit(excess, floc=0)

        # 4) EVT‐VaR in **return** terms solves
        #    P(r <= VaR_evt) = α = p_exc * GPD.cdf(u - VaR_evt)
        #  ⇒ VaR_evt = u - (β/ξ)*((p_exc/α)**ξ - 1)
        var_evt = u - (beta/xi) * ((p_exc/self.alpha_evt)**xi - 1)

        # 5) EVT‐CVaR: conditional mean of r ≤ VaR_evt
        if xi < 1:
            cvar_evt = u - (beta + (beta/xi) * ((p_exc/self.alpha_evt)**xi - 1)) / (1 - xi)
        else:
            cvar_evt = np.nan  # infinite tail mean

        return float(var_evt), float(cvar_evt)

    def plot_monte_carlo_results(self,
                                port_paths: np.ndarray,
                                empirical: tuple,
                                evt: tuple,
                                position: np.ndarray,
                                ):
        """
        Plots sample Monte Carlo price paths and histogram of terminal returns with VaR/CVaR lines.
        
        Parameters
        ----------
        port_paths : np.ndarray, shape (n_steps, n_sims)
            Simulated price paths (rows=time, columns=simulations).
        var : float
            Value at Risk at the given confidence level (as a return, e.g. -0.10 for -10%).
        cvar : float
            Conditional VaR (expected return in the worst (1 - confidence_level)%).
        confidence_level : float
            The confidence level used for VaR/CVaR (e.g., 0.95).
        """

        var = empirical[0]
        cvar = empirical[1]
        var_evt = evt[0]
        cvar_evt = evt[1]


        # 1) Sample paths
        plt.figure()
        mean_path = np.mean(port_paths[:, :]/port_paths[0,:] - 1, axis = 1)
        sigma2_pos_path = np.quantile(port_paths[:, :]/port_paths[0,:] - 1, axis = 1, q = 0.95)
        sigma2_neg_path = np.quantile(port_paths[:, :]/port_paths[0,:] - 1, axis = 1, q = 0.05)
        for i in np.random.randint(0, self.n_sims, 100):
            plt.plot(self.times, (port_paths[:, i]/port_paths[0,i] - 1)*100)
        plt.plot(self.times, mean_path *100, '--k', ms = 10,label = "Average path")
        plt.plot(self.times, sigma2_pos_path *100, '--k', ms = 10,label = "95% percentile path")
        plt.plot(self.times, sigma2_neg_path*100, '--k', ms = 10, label = "5% percentile path")
        if self.final_price is not None:
            historical_return = ((self.final_price/port_paths[0,0] - 1) * position).sum()
            plt.axhline(historical_return*100, color = 'navy',label = f"Historical return = {historical_return:.2%}")
        plt.title(f"Sample Monte Carlo Price Paths in {self.n_steps-1} steps of {self.assets}")
        plt.xlabel("Time step")
        plt.ylabel("Simulated Total Return (%)")
        plt.legend()
        
        # 2) Histogram of terminal returns with risk metrics
        terminal_returns = port_paths[-1,:] / port_paths[0,:] - 1
        plt.figure()
        _, bins, patches = plt.hist(terminal_returns*100, bins=25, color = 'steelblue')
        
        plt.axvline(var*100, linestyle='--', color = 'k',label=f"VaR ({int((1 - self.alpha_empirical)*100)}%) = {var:.2%}")
        plt.axvline(cvar*100, linestyle='-', color = 'k', label=f"CVaR ({self.alpha_empirical:0.1%} tail) = {cvar:.2%}")

        plt.axvline(var_evt*100, linestyle='--', color = 'grey',label=f"VaR_EVT ({int((1 - self.alpha_evt)*100)}%) = {var_evt:.2%}")
        plt.axvline(cvar_evt*100, linestyle='-', color = 'grey', label=f"CVaR_EVT ({self.alpha_evt:0.1%} tail) = {cvar_evt:.2%}")
        if self.final_price is not None:
            plt.axvline((historical_return)*100, color = 'navy',label = f"Historical return = {historical_return:.2%}")

        # find bins in the worst α tail
        for rect, edge in zip(patches, bins):
            if edge <= var*100:
                rect.set_facecolor('C3')  # tail shaded
        
        plt.title(f"Distribution of Terminal Returns in {self.n_steps-1} steps of {self.assets}")
        plt.xlabel("Returns (%)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def simulate_portfolio(self,):
        all_paths = self.simulate_gbm_paths()
        S_start = all_paths[0,:,:]
        S_end   = all_paths[-1,:,:]
        assets_returns = S_end / S_start - 1 # shape = (n_sims, n_assets)

        if self.optimize_position_sizing:
            def portfolio_obj(w):
                w = np.array(w)
                exp_ret = (assets_returns @ w).mean() #Final expected returns averaged across all sims
                volat = w @ np.cov(assets_returns, rowvar=False) @ w #Variance of all assets (diagonal of covariance matrix)

                port_returns = assets_returns @ w #Combine scaled prices (based on positions)
                var_emp, cvar_emp = self.compute_var_cvar(port_returns) # CVaR calculated on returns (not losses)
                # we want to maximize: exp_ret - volat_wt*volat + cvar_wt*cvar_emp
                return -(self.returns_penalty * exp_ret - self.volat_penalty * volat + self.cvar_penalty * cvar_emp)

            # constraints & bounds
            cons   = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0,1)] * self.n_assets
            x0     = np.ones(self.n_assets) / self.n_assets
            # solve
            res = minimize(portfolio_obj, x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=cons,
                        options={'disp': True})
            w_opt = res.x
        else:
            w_opt = np.ones(self.n_assets) / self.n_assets
            print("Position size NOT optimized")

        optimized_portfolio = pd.DataFrame({
            "position": w_opt,
            "Avg Simulated Return (%)": (assets_returns * w_opt).mean(axis = 0) * 100,
            "Var Simulated Returns (%)": np.diag(np.cov(assets_returns, rowvar=False)) * w_opt * 100 if self.n_assets > 1 else np.var(assets_returns) * 100
        }, index=self.logrets.columns)

        empirical = self.compute_var_cvar(assets_returns @ w_opt)
        evt = self.compute_evt_var_cvar(assets_returns @ w_opt)
        best_port_path = all_paths @ w_opt

        return best_port_path, optimized_portfolio, empirical, evt