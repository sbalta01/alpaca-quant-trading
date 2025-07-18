from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import numpy as np
from src.data.data_loader import fetch_nasdaq_100_symbols
from src.utils.monte_carlo import monte_carlo_portfolio_risk

# symbols = ["SPY"]
symbols = ["AAPL","AMZN","MSFT","GOOG"]
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

# symbols = fetch_nasdaq_100_symbols()

end     = datetime.now()
# end     = datetime(2023, 1, 1)
start = end - timedelta(days=2*365)
# start   = datetime(2024, 1, 1)
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
    

# VaR: With (95%) confidence, my returns are above VaR.
# CVaR: If the loss does exceed VaR (returns below VaR), the average loss is CVaR.


monte_carlo_simulator = monte_carlo_portfolio_risk( #10-day horizon over >250 days of data is good
        price_hist   = df,                    # multiindex or wide
        T            = 26/252, #In years. n_steps = T - 1
        # T            = 1, #In years. n_steps = T - 1
        dt           = 1/252, #1/252 = 1 day
        n_sims       = 10_000,
        alpha_empirical= 0.05,
        alpha_evt    = 0.01,
        returns_penalty = 1.0,
        volat_penalty = 1.0,
        cvar_penalty = 1.0,
        random_state = 43,
        predict = True, #Backtesting uses all data up until -T to compute GBM parameters and starts predicting from -T. It is equivalent to predicting from -T.
        optimize_position_sizing = True,
    )


best_port_paths, optimized_portfolio, empirical, evt = monte_carlo_simulator.simulate_portfolio()


print("Portfolio:")
print(optimized_portfolio.sort_values(by="position", axis = 0, ascending=False).loc[optimized_portfolio["position"]>1e-3])
print(f"\nPortfolio VaR (95%): {empirical[0]:.2%}")
print(f"Portfolio CVaR (5% tail): {empirical[1]:.2%}")


monte_carlo_simulator.plot_monte_carlo_results(best_port_paths,
                        empirical,
                        evt,
                        optimized_portfolio["position"],
                        )