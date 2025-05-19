import pandas as pd

def fetch_sp500_symbols():
    url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    df = pd.read_csv(url)
    df[['Symbol']].to_csv('sp500.csv', index=False)
    print("sp500.csv created with", len(df), "symbols.")

fetch_sp500_symbols()