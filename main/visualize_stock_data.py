from datetime import datetime
from src.backtesting.visualizer import plot_candles_with_macd
from src.data.data_loader import fetch_yahoo_data as fetch_data


start   = datetime(2024, 8, 30)
end     = datetime.now()
symbols = ["AAPL"]
df = fetch_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe="1d",
        )

plot_candles_with_macd(df, animate = False)