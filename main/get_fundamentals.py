from datetime import datetime
from finnhub import Client
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import pandas as pd
load_dotenv()

finnhub_client = Client(api_key=os.getenv("FINNHUB_API_KEY"))

# Fetch current + historical fundamentals:
symbols = ["AAPL"]
financials = finnhub_client.company_basic_financials(symbols, 'all')

# Current metrics:
current = financials['metric']
# print(current.keys())
print("P/E today:", current["peTTM"])

# Historical snapshots (quarterly):
series = financials['series']['quarterly']
# print(series.keys())
for pe_point in series['peTTM'][:]:
    print(pe_point['period'], "PE =", pe_point['v'])

from src.data.data_loader import fetch_yahoo_data as fetch_data
start   = datetime(2000, 8, 30)
# start   = datetime(2008, 1, 1)
# end   = datetime(2008, 8, 30)
end     = datetime.now()
df = fetch_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe="1d",
        )

peTTM = series['peTTM'][:-10]
f_dates = pd.to_datetime([f['period'] for f in peTTM])
f_values = pd.Series([f['v'] for f in peTTM])

fig, ax1 = plt.subplots(figsize=(10,5))
for symbol, subdf in df.groupby(level="symbol"):
    subdf = subdf.droplevel("symbol")
    ax1.plot(subdf.index, subdf["close"], color='navy', label=f'{symbol} Close')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.set_xlim(subdf.index[0])

    ax2 = ax1.twinx()
    ax2.scatter(f_dates, f_values, color='steelblue', label='Fundamental', s=5)
    ax2.vlines(f_dates, ymin=f_values.min(),ymax=f_values.max(), color='steelblue')
    ax2.set_ylabel('Fundamental Value', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    fig.suptitle(f"{symbol} Price vs Fundamentals")
    fig.tight_layout()
    plt.show()