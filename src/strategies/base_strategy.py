# src/strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    Each subclass must define:
      - A `name` attribute
      - A `generate_signals(data: pd.DataFrame) -> pd.DataFrame` method
        which returns the input DataFrame augmented with at least:
          * 'signal'    column (1.0 = long, 0.0 = flat)
          * 'positions' column (diff of signal: +1 buy, -1 sell)
    """

    name: str  # e.g. "MovingAverage"

    def __init__(self, **kwargs):
        """
        Optional: accept strategy-specific parameters via kwargs.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trading signals given price data.
        Must return a DataFrame with 'signal' and 'positions' columns.
        """
        ...
