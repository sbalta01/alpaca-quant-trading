# src/utils/metrics.py

import numpy as np
from sklearn.metrics import make_scorer

def sharpe_from_signals(y_true: np.ndarray, y_pred: np.ndarray, freq: float = 252) -> float:
    """
    Compute a “signal-Sharpe” where:
      - At each t, position_t = sign(y_pred[t])
      - Daily P&L_t  = position_{t-1} * y_true[t]
    We drop the first day (no prior position).
    """
    # 1) build positions: -1, 0 or +1
    pos = np.sign(y_pred)
    # 2) shift positions by 1 day (can't trade same day you predict)
    pos_lag = np.concatenate([[0.0], pos[:-1]])
    # 3) P&L series
    pnl = pos_lag * y_true
    # 4) Sharpe = mean/std * sqrt(annualization)
    mean = np.nanmean(pnl)
    std  = np.nanstd(pnl, ddof=1)
    return (mean / std) * np.sqrt(freq) if std > 0 else 0.0

# Wrap in a sklearn scorer
sharpe_scorer = make_scorer(
    sharpe_from_signals,
    greater_is_better=True,
    needs_proba=False,   # we pass y_pred directly
    needs_threshold=False
)
