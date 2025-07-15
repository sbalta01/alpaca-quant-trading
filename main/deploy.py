from alpaca.data.timeframe import TimeFrame

from src.strategies.lstm_event_technical_ML import LSTMEventTechnicalStrategy
from src.execution.live_executor import run_live_strategy

if __name__ == "__main__":

    md_report_file_path = "live_trading_report.md"
    open(md_report_file_path, "w", encoding="utf-8").close() #Clear file
    #US stocks
    # symbols = ["AAPL","AMZN","MSFT","GOOG","ROP", "VRTX"]
    symbols = ["MSFT"]
    # symbols = ["SPY"]

    strat = LSTMEventTechnicalStrategy(
        horizon=10,        # predict horizon-day return
        threshold=0.05,   # event = next-horizon-day log-return > threshold
        train_frac = 0.7,
        cv_splits = 2, #For optuna hyperparameter fitting
        n_models = 5,
        bootstrap = 0.8,
        random_state=42,
        sequences_length = 25,
        prob_positive_threshold = 0.7,
        with_hyperparam_fit = False, #Seems useful
        with_feature_attn = False,  #Seems useless
        with_pos_weight = True, #Crucial
        adjust_threshold = True, #More appropriate but obv it is safer a higher flat threshold
    )

    lookback_time = 5*365*24*60 #In minutes
    run_live_strategy(
        strat,
        symbols=symbols,
        timeframe = TimeFrame.Day,
        lookback_minutes=lookback_time, #In minutes regardless of timeframe
        interval_seconds=None, #None if intended to run only once
        cash_per_trade=10000,
        feed='iex'
        )
    
    #EU stocks
    symbols = ["HAG.DE","RHM.DE","IDR.MC","ECR.MC"]
    run_live_strategy(
        strat,
        symbols=symbols,
        timeframe = TimeFrame.Day,
        lookback_minutes=lookback_time, #In minutes regardless of timeframe
        interval_seconds=None, #None if intended to run only once
        cash_per_trade=10000,
        feed='iex'
        )