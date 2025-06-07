# src/strategies/pair_trading_cointegration.py

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple

from statsmodels.tsa.stattools import coint

from src.strategies.base_strategy import Strategy

class CointegrationPairTradingStrategy(Strategy):
    """
    Pair-Trading on cointegrated pairs:
      - Pre-screen by correlation
      - Test cointegration (Engle-Granger)
      - Trade z-score of log-spread
      - Aggregate exposures across all good pairs
    """
    name = "CointegratedPairTrading"
    multi_symbol = True

    def __init__(
        self,
        corr_threshold: float = 0.8,
        pvalue_threshold: float = 0.05,
        lookback: int = 20,
        z_entry: float = 2.0,
        z_exit: float = 0.5
    ):
        """
        corr_threshold   : min abs(corr) to consider a pair
        pvalue_threshold : max p-value in Engle-Granger test to accept cointegration
        lookback         : window for rolling mean/std of log-spread
        z_entry, z_exit  : entry/exit thresholds on z-score
        """
        self.corr_threshold = corr_threshold
        self.pvalue_threshold = pvalue_threshold
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 1) Pivot close prices: index=timestamp, columns=symbol
        price = data["close"].unstack("symbol").sort_index()

        # 2) Pre-screen by correlation over entire period
        corr = price.corr().abs()
        symbols = price.columns.tolist()
        candidate_pairs = [
            (s1, s2) for s1, s2 in combinations(symbols, 2)
            if corr.at[s1, s2] >= self.corr_threshold
        ]

        # 3) Cointegration test on candidates
        good_pairs: List[Tuple[str,str]] = []
        for s1, s2 in candidate_pairs:
            # dropna to align
            series1 = price[s1].dropna()
            series2 = price[s2].dropna()
            # align on intersection
            common_idx = series1.index.intersection(series2.index)
            if len(common_idx) < self.lookback:
                continue
            pvalue = coint(series1.loc[common_idx], series2.loc[common_idx])[1]
            if pvalue <= self.pvalue_threshold:
                good_pairs.append((s1, s2))

        if not good_pairs:
            # no pairs → flat for everyone
            out = data.copy()
            out["position"] = 0.0
            out["signal"] = 0.0
            print("There are NO good pairs")
            return out

        # 4) For each good pair, compute its z-score time series and pos1/pos2
        # We'll build a dict: pair → DataFrame with columns {s1:pos1, s2:pos2}
        pair_positions = {}
        for s1, s2 in good_pairs:
            spread = np.log(price[s1]) - np.log(price[s2])
            mean_sp = spread.rolling(self.lookback).mean()
            std_sp  = spread.rolling(self.lookback).std()
            zscore  = (spread - mean_sp) / std_sp

            pos1 = pd.Series(0.0, index=price.index)
            pos2 = pd.Series(0.0, index=price.index)
            position = 0  # 0=flat, +1=long spread, -1=short spread

            for t in zscore.index:
                z = zscore.loc[t]
                # entry/exit logic
                if position == 0:
                    if z > self.z_entry:
                        position = -1
                    elif z < -self.z_entry:
                        position = 1
                elif position == 1 and z > -self.z_exit:
                    position = 0
                elif position == -1 and z < self.z_exit:
                    position = 0
                pos1.at[t] = position
                pos2.at[t] = -position

            pair_positions[(s1,s2)] = pd.DataFrame({s1: pos1, s2: pos2})

        # 5) Aggregate across all pairs: for each symbol, average its positions
        all_pos = pd.DataFrame(0.0, index=price.index, columns=symbols)
        for (s1,s2), df_pair in pair_positions.items():
            all_pos[s1] += df_pair[s1].fillna(0.0)
            all_pos[s2] += df_pair[s2].fillna(0.0)

        all_pos = all_pos.clip(-1,1) #Summarize signals over all pairs that a particular symbol belongs to

        # 6) Build MultiIndex output
        out = data.copy()
        # position per symbol/timestamp
        pos_flat = all_pos.stack()                # Series indexed (timestamp, symbol)
        pos_flat.index.names = ["timestamp","symbol"]
        pos_flat = pos_flat.swaplevel()           # now (symbol, timestamp)
        out["position"] = pos_flat.reindex(out.index).fillna(0.0).clip(0,1) #No short selling by clipping
        out["signal"] = out.groupby(level="symbol")["position"].diff().fillna(0.0)
        return out