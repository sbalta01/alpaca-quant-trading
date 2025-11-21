# src/strategies/macd_strategy.py

import numpy as np
import pandas as pd

from src.strategies.base_strategy import Strategy

class MACDStrategy(Strategy):
    """
    A pure-technical, rule-based MACD strategy.
    - Buy when MACD crosses above Signal line and histogram is expanding.
    - Sell (go flat) when MACD crosses below Signal and histogram is contracting.
    - Optional zero-line filter to weight signals differently above/below zero.
    """
    name = "MACD"
    multi_symbol = False

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        hist_mom: int = 3,
        zero_filter: bool = True,
    ):
        """
        Parameters
        ----------
        fast, slow : int
            Lookback windows for the two EMA lines of MACD.
        signal : int
            Lookback for the Signal-line EMA of the MACD line.
        hist_mom : int
            Number of bars over which we require histogram to be expanding/contracting.
        zero_filter : bool
            If True, only generate bullish signals when both lines are above zero,
            and bearish when both below.
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.hist_mom = hist_mom
        self.zero_filter = zero_filter

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # 1) Compute MACD components
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        hist = macd_line - signal_line

        df['macd'] = macd_line
        df['signal'] = signal_line
        df['hist'] = hist

        # 2) Lagged values for crossover detection
        df['macd_prev']   = df['macd'].shift(1)
        df['signal_prev']= df['signal'].shift(1)
        df['hist_prev']  = df['hist'].shift(1)

        # 3) Detect crossovers
        bullish_xov = (df['macd_prev'] < df['signal_prev']) & (df['macd'] > df['signal'])
        bearish_xov = (df['macd_prev'] > df['signal_prev']) & (df['macd'] < df['signal'])

        # 4) Histogram momentum: require hist to be rising for bullish; falling for bearish
        #    over the last `hist_mom` bars.
        df['hist_mom'] = df['hist'] - df['hist'].shift(self.hist_mom)
        hist_up = df['hist_mom'] > 0
        hist_down = df['hist_mom'] < 0

        # 5) Zero-line context
        if self.zero_filter:
            above_zero = (df['macd'] > 0) & (df['signal'] > 0)
            below_zero = (df['macd'] < 0) & (df['signal'] < 0)
        else:
            above_zero = below_zero = pd.Series(True, index=df.index)

        # 6) Build positions: 1 = long, 0 = flat
        position = 0
        positions = []
        for t, (bx, sx, hu, hd, az, bz) in enumerate(
            zip(bullish_xov, bearish_xov, hist_up, hist_down, above_zero, below_zero)
        ):
            # if we're flat and get a bullish signal in a bullish regime:
            if position == 0 and (bx and hu and az):
                position = 1
            # if we're long and get a bearish signal in a bearish regime:
            elif position == 1 and (sx and hd and bz):
                position = 0
            positions.append(position)

        out = df.copy()
        out['position'] = positions

        # 7) Signal = diff of position
        out['signal'] = out['position'].diff().fillna(0).clip(-1,1)

        return out