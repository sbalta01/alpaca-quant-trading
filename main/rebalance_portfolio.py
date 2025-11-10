import os
import time
import json
import sys
import holidays
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from src.data.data_loader import fetch_nasdaq_100_symbols, fetch_yahoo_data as fetch_data

from dotenv import load_dotenv

from src.utils.monte_carlo import monte_carlo_portfolio_risk
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PAPER = os.getenv("PAPER")

from src.data.data_loader import fetch_alpaca_data
from src.strategies.base_strategy import Strategy
from src.execution.live_tracker import LivePerformanceTracker

# Alpaca trading client
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
account = trading_client.get_account()

def get_target_positions(symbols: list, start: datetime, end: datetime, timeframe: str, predict_horizon: int) -> dict:
    """
    Calculate the current best portfolio size for each symbol
    maximizing returns and miniziming risk and VaR
    """
    print('USING YAHOO DATA')
    df = fetch_data(
        symbol=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        )

    n_symbols = len(symbols)
    monte_carlo_simulator = monte_carlo_portfolio_risk( #10-day horizon over >250 days of data is good
        price_hist   = df,                    # multiindex or wide
        T            = (predict_horizon + 1)/252, #In years. n_steps = T - 1
        dt           = 1/252, #1/252 = 1 day
        n_sims       = 10_000,
        alpha_empirical= 0.05,
        alpha_evt    = 0.01,
        returns_penalty = 1.0,
        volat_penalty = 1.0,
        cvar_penalty = 1.0,
        random_state = 42,
        predict = True, #Backtesting uses all data up until -T to compute GBM parameters and starts predicting from -T. It is equivalent to predicting from -T.
        optimize_position_sizing = True,
    )
    _, optimized_portfolio, empirical, evt = monte_carlo_simulator.simulate_portfolio()
    optimized_portfolio.loc[optimized_portfolio["position"]<1/(5*n_symbols)] = 0.0 #Positions below the threshold are just removed from the portfolio
    optimized_portfolio = optimized_portfolio.sort_values(by="position", axis = 0, ascending=False)
    days_report = f"Metrics predicted for the next {predict_horizon} days:\n"
    var_report = f"Portfolio VaR ({1-monte_carlo_simulator.alpha_empirical:0.0%}): {empirical[0]:.2%}\n"
    cvar_report = f"Portfolio CVaR ({monte_carlo_simulator.alpha_empirical:0.0%} tail): {empirical[1]:.2%}\n"
    var_evt_report = f"Portfolio VaR ({1-monte_carlo_simulator.alpha_evt:0.1%}): {evt[0]:.2%}\n"
    cvar_evt_report = f"Portfolio CVaR ({monte_carlo_simulator.alpha_evt:0.1%} tail): {evt[1]:.2%}\n"

    stats_report = days_report + var_report + cvar_report + var_evt_report + cvar_evt_report
    print(stats_report)

    return optimized_portfolio["position"].to_dict(), stats_report


def get_current_positions() -> dict:
    """
    Fetch current open positions from Alpaca account.
    Returns dict of ticker -> quantity (int).
    """
    positions = trading_client.get_all_positions()
    return {pos.symbol: float(pos.qty) * float(pos.current_price) for pos in positions} #Returns current equity for each symbol in dollars


def rebalance_portfolio(targets: dict, current: dict, initial_investment: float, md_report_file_path: str):
    """
    Rebalance portfolio to match target positions.
    Buys or sells the difference between target and current.
    """
    total_portfolio_notional = 0
    for symbol, _ in targets.items():
        total_portfolio_notional += current.get(symbol, 0)
    for symbol, target_proportion in targets.items():
        current_notional = current.get(symbol, 0) #Return the value for key if key is in the dictionary, else default.
        target_notional = total_portfolio_notional*target_proportion #Every time I rebalance, I always keep whatever the total equity is, and not realize any differences
        delta = target_notional - current_notional
        if abs(delta) < 1: #No trades below 1$ are allowed in Alpaca
            report = f"Order below 1$ -> No order submitted for {symbol}.\n"
            # print(report)
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty_to_order = round(abs(delta),2)
        try:
            if abs(target_notional) < 1: #If the new target is almost 0 then sell everything
                qty_to_order = trading_client.get_open_position(symbol).qty_available
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty_to_order,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                report = f"Submitted {side} order for ALL shares of {symbol}. New proportion of portfolio: {target_proportion:0.2%}\n"
            else:
                order = MarketOrderRequest(
                    symbol=symbol,
                    notional=qty_to_order,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                report = f"Submitted {side} order for {qty_to_order}$ of {symbol}. New proportion of portfolio: {target_proportion:0.2%}\n"
            trading_client.submit_order(order)
            print(report)
        except Exception as e:
            report = f"Error submitting order for {symbol}: {e}\n"
            print(report)

        with open(md_report_file_path, "a", encoding="utf-8") as md_file:
            md_file.write(report)

    realized_returns = total_portfolio_notional/initial_investment - 1
    returns_report = f"Total returns from 2025-07-22: {realized_returns:0.2%}\n"
    print(returns_report)
    with open(md_report_file_path, "a", encoding="utf-8") as md_file:
        md_file.write(returns_report)

def main():
    md_report_file_path = "live_portfolio_rebalance.md"
    open(md_report_file_path, "w", encoding="utf-8").close() #Clear report

    # Available for tuning
    now_utc = datetime.now(tz=timezone.utc)
    start = now_utc - timedelta(days= 2* 365) #Dataset for fitting the GBM parameters
    timeframe = "1d"
    update_portfolio_horizon = 30
    initial_investment = 50_000 #50_000 for nas100 tickers at 2025-11-10

    market_hols = holidays.financial_holidays("NYSE")
    today = now_utc.date()
    if today.weekday() >= 5:
        print("Today is weekend; exiting.")
        with open(md_report_file_path, "a", encoding="utf-8") as md_file:
            md_file.write(f"Today is weekend. Rerun MANUALLY next business day.\n")
        sys.exit(0)
    if today in market_hols:
        print(f"Today is a holiday; exiting.")
        with open(md_report_file_path, "a", encoding="utf-8") as md_file:
            md_file.write(f"Today is a market holiday. Rerun MANUALLY next business day.\n")
        sys.exit(0)

    print("Business day; continuing with the rest of the workflow.")

    current = get_current_positions()

    rebalance_entire_portfolio = False
    if rebalance_entire_portfolio:
        current_symbols = list(current.keys())
        
        with open(md_report_file_path, "a", encoding="utf-8") as md_file:
            md_file.write(f"{now_utc}: Start rebalancing the ENTIRE portfolio:\n\n")
        targets, stats_report = get_target_positions(current_symbols, start = start, end = now_utc, timeframe = timeframe, predict_horizon = update_portfolio_horizon)
    else:
        symbols = fetch_nasdaq_100_symbols()
        # symbols = ['AAPL']

        with open(md_report_file_path, "a", encoding="utf-8") as md_file:
            md_file.write(f"{now_utc}: Start rebalancing the following symbols:\n{symbols}\n\n")
        targets, stats_report = get_target_positions(symbols, start = start, end = now_utc, timeframe = timeframe, predict_horizon = update_portfolio_horizon)
    
    with open(md_report_file_path, "a", encoding="utf-8") as md_file:
            md_file.write(stats_report)
            md_file.write("\n")

    rebalance_portfolio(targets, current, initial_investment, md_report_file_path=md_report_file_path)

    with open(md_report_file_path, "a", encoding="utf-8") as md_file: # Update report
        md_file.write("\n")
        md_file.write(f"{now_utc}: Rebalance completed.\nNext run in {update_portfolio_horizon} days (Double check GitHub scheduler's time).")


if __name__ == "__main__":
    main()