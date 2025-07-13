from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import numpy as np
from src.data.data_loader import fetch_nasdaq_100_symbols
from src.utils.monte_carlo import monte_carlo_portfolio_risk, plot_monte_carlo_results

# symbols = ["SPY"]
# symbols = ["AAPL","AMZN","MSFT","GOOG","ROP", "VRTX"]
# symbols = ["MSFT"]
# symbols = ["ROP"]
# symbols = ["AAPL","MSFT"]
# symbols = ["AAPL"]
# symbols = ["PFE"]
# symbols = ["HAG.DE"]
# symbols = ["RHM.DE"]
# symbols = ["MRK"]
# symbols = ["LMT"]
# symbols = ["WOLF"]

symbols = fetch_nasdaq_100_symbols()

start   = datetime(2024, 1, 1)
# start   = datetime(2025, 1, 5)
end     = datetime.now()
# end     = datetime(2025, 1, 1)
timeframe = TimeFrame.Day  # or pd.Timedelta(days=1)
feed = 'iex'

# Fetch historical data
if timeframe.unit_value == TimeFrameUnit.Month:
    print('Timeframe set to Month')
    timeframe_yahoo = '1mo'
elif timeframe.unit_value == TimeFrameUnit.Week:
    print('Timeframe set to Week')
    timeframe_yahoo = '1wk'
elif timeframe.unit_value == TimeFrameUnit.Day:
    print('Timeframe set to Day')
    timeframe_yahoo = '1d'
elif timeframe.unit_value == TimeFrameUnit.Hour:
    print('Timeframe set to Hour')
    timeframe_yahoo = '1h'
elif timeframe.unit_value == TimeFrameUnit.Minute:
    print('Timeframe set to Minute')
    timeframe_yahoo = '1m'
try:
    from src.data.data_loader import fetch_alpaca_data as fetch_data
    raise Exception("Chose not to import from alpaca")
    df = fetch_data(
    symbol=symbols,
    start=start,
    end=end,
    timeframe=timeframe,
    feed = feed
    )
    print('USING ALPACA DATA')
except:
    from src.data.data_loader import fetch_yahoo_data as fetch_data
    print('USING YAHOO DATA')
    timeframe = timeframe_yahoo
    feed = None
    df = fetch_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        feed = feed
        )
    

results = monte_carlo_portfolio_risk( #10-day horizon over >250 days of data is good
    price_hist   = df,                    # multiindex or wide
    T            = 11/252, #In years. n_steps = T - 1
    dt           = 1/252, #1/252 = 1 day
    n_sims       = 10000,
    alpha_empirical= 0.05,
    alpha_evt    = 0.01,
    trials       = 100, #Number of iterations to run
    random_state = 43,
    normalize = False,
    max_weight = None,  # e.g. 0.25 to cap any asset at 25%
    backtest = True, #Backtesting uses all data up until -T to compute GBM parameters and starts predicting from -T. It is equivalent to predicting from -T.
)

# VaR: With (95%) confidence, my returns are above VaR.
# CVaR: If the loss does exceed VaR (returns below VaR), the average loss is CVaR.
