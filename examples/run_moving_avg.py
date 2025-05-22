from src.strategies.moving_average import MovingAverageStrategy
from src.execution.live_executor import run_live_strategy

if __name__ == "__main__":
    # Example: live-run moving-average on AAPL or multiple symbols
    symbols = ["AAPL", "MSFT"]
    ma_strategy = MovingAverageStrategy(short_window=5, long_window=20)
    run_live_strategy(ma_strategy, symbols=symbols, lookback_minutes=30, interval_seconds=60)