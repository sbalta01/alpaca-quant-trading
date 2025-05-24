from alpaca.data.timeframe import TimeFrame

from src.strategies.moving_average import MovingAverageStrategy
from src.execution.live_executor import run_live_strategy

if __name__ == "__main__":
    # Example: live-run moving-average on AAPL or multiple symbols. Opening times NYSE/NASDAC: 15:30-22:00 CEST
    symbols = ["AAPL", "MSFT"]
    ma_strategy = MovingAverageStrategy(short_window=5, long_window=20, ma= 'ema')
    lookback_time = 30
    run_live_strategy(
        ma_strategy,
        symbols=symbols,
        timeframe = TimeFrame.Minute,
        lookback_minutes=lookback_time, #In minutes regardless of timeframe
        interval_seconds=30,
        cash_per_trade=5000,
        feed=None
        )