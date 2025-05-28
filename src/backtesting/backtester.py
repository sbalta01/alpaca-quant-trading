# src/backtesting/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, Union

from sklearn.metrics import confusion_matrix
from src.strategies.base_strategy import Strategy

class BacktestEngine:
    """
    A simple backtest engine that:
      - Takes a Strategy instance and historical price data (DataFrame).
      - Calls strategy.generate_signals(data) to get 'signal' & (new) 'position' columns.
      - Simulates P&L over time assuming:
          * Full allocation on each trade (all‐in / all‐out)
          * Zero transaction costs
          * No leverage
      - Computes basic performance metrics.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_cash_per_stock: float = 10_000.0,
    ):
        self.strategy = strategy
        self.data = data
        self.initial_cash_per_stock = initial_cash_per_stock

        # Exposed for tests/reporting
        self.position: float = 0.0  # number of shares held

    def _run_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest for ONE symbol (DatetimeIndex).
        """
        # Initialize
        df["position"] = float(0.0)   # share count
        df["cash"]     = float(self.initial_cash_per_stock)
        df["returns"]   = float(0.0)
        cash = float(self.initial_cash_per_stock)
        position = float(0.0)

        for t in range(1, len(df)):
            price  = df["close"].iat[t]
            signal = df["signal"].iat[t]  # +1 buy, -1 sell, 0 hold

            if signal ==  1.0 and position == 0.0:
                # Buy all‐in
                position = cash / price
                cash = 0.0
            elif signal == -1.0 and position > 0.0:
                # Sell all‐out
                cash = position * price
                position = 0.0

            df.iat[t, df.columns.get_loc("position")] = position
            df.iat[t, df.columns.get_loc("cash")]     = cash
            df.iat[t, df.columns.get_loc("returns")]   = (cash + position * price - self.initial_cash_per_stock)/self.initial_cash_per_stock

        return df

    def run(self) -> pd.DataFrame:
        """
        If self.data is multi‐symbol (MultiIndex), run each symbol separately
        and concatenate the results with a MultiIndex. Otherwise, run single.
        """
        df = self.data

        if self.strategy.multi_symbol:
            df = self.strategy.generate_signals(df).copy() ##Generate signals all at once
            results = []
            for symbol, subdf in df.groupby(level="symbol"):
                single = subdf.droplevel("symbol")
                out = self._run_single(single)
                # Reattach symbol level
                out["symbol"] = symbol
                results.append(out)
        else:
            results = []
            for symbol, subdf in df.groupby(level="symbol"):
                single = subdf.droplevel("symbol")
                single = self.strategy.generate_signals(single).copy() ##Generate signals one by one
                out = self._run_single(single)
                # Reattach symbol level
                out["symbol"] = symbol
                results.append(out)

        # concat and re‐set MultiIndex(symbol, timestamp)
        final = pd.concat(results)
        final = final.set_index("symbol", append=True)
        final = final.reorder_levels(["symbol", final.index.names[0]])
        return final.sort_index()

    def performance(self, results: pd.DataFrame, num_years) -> Dict[str, float]:
        """Return expanded metrics: Sharpe, Sortino, Calmar, Turnover, Fitness."""
        results["equity"] = results["position"]*results["close"] + results["cash"]
        initial_cash = results["cash"].groupby(level="timestamp").sum().iloc[0]
        total_equity = results["equity"].groupby(level="timestamp").sum()
        final_equity = total_equity.iloc[-1]
        total_returns = results['returns'].groupby(level="timestamp").mean() #Mean assumes same cash_per_trade for each asset
        final_returns = total_returns.iloc[-1] 
        profit = final_equity - initial_cash
        total_returns_cum = total_equity.pct_change().dropna()

        def metrics(total_returns_cum, total_equity): #All metrics are calculated in a Day TimeFrame
            # Annualization factor
            ann = np.sqrt(252)
            mean = np.mean(total_returns_cum); std = np.std(total_returns_cum, ddof=1)
            sharpe = (mean/std)*ann if std>0 else np.nan

            # Sortino (only downside)
            neg = total_returns_cum[total_returns_cum<0]
            downside = np.std(neg, ddof=1)
            sortino = (mean/downside)*ann if downside>0 else np.nan

            # Max drawdown
            cum = np.cumprod(1+total_returns_cum)
            peak = np.maximum.accumulate(cum)
            max_drawdown = np.min(cum/peak -1)
            # max_drawdown = (((total_returns+1) / (total_returns+1).cummax()) - 1).min() ##Equivalent

            # Calmar = CAGR / |maxDD|
            cagr = cum.iloc[-1]**(1/num_years) -1
            # cagr = ((final_returns + 1) ** (1 / num_years) - 1) ##Equivalent
            calmar = cagr/abs(max_drawdown) if max_drawdown<0 else np.nan

            # Turnover = sum |Δposition| / len
            pos = total_equity / results['close'].groupby(level="timestamp").mean()  # Avg count share (across all symbols)
            # approximate turnover as fraction of portfolio traded
            to = np.mean(np.abs(np.diff(pos)))  

            # Fitness = sharpe / turnover
            fitness = sharpe / to if to>0 else np.nan

            try:
                total_cm = np.array([[0, 0], [0, 0]])
                for symbol, subdf in results.groupby(level="symbol"):
                    y_test = subdf.loc[subdf['test_mask']==1, 'y_test']
                    y_pred = subdf.loc[subdf['test_mask']==1, 'y_pred']
                    cm = confusion_matrix(y_test, y_pred)
                    total_cm += cm

                TN, FP, FN, TP = total_cm[0, 0], total_cm[0, 1], total_cm[1, 0], total_cm[1, 1]
                # Accuracy
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                # Precision and Recall
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
                # F1 Score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                ML_metrics = dict(zip(
                    ['F1 Score', 'Accuracy', 'Precision', 'Recall'],
                    [[f1_score,'>0.6 good, >0.8 strong. Balance precision-recall'], [accuracy,'>0.7 good. Proportion of correct predictions (Generally biased in trading)'],
                        [precision,'>0.6 good. Ratio of correct buy predictions of all predictions'], [recall,'>0.6 good. Ratio of buy predicted of all buys']
                        ]))
            except:
                ML_metrics = {'No ML algorithm': ['','']}

            return initial_cash, final_equity, profit, max_drawdown, cagr, final_returns, sharpe, sortino, calmar, to, fitness, ML_metrics

        agg = metrics(total_returns_cum,total_equity)
        out = dict(zip(
            ['Initial Cash', 'Final Equity','Profit','Max Drawdown','CAGR','Final Return','Sharpe','Sortino','Calmar','Turnover','Fitness', 'ML metrics'],
            agg
        ))
        return out
