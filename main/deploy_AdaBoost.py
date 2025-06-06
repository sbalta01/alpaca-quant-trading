from alpaca.data.timeframe import TimeFrame

from src.strategies.adaboost_ML import AdaBoostStrategy
from src.execution.live_executor import run_live_strategy

if __name__ == "__main__":
    symbols = ["SPY"]

    strat = AdaBoostStrategy(
        d=10, #10 has the best
        train_frac=0.99,
        cv_splits=5,
        # param_grid={
        #     'clf__n_estimators': [50,100,200],
        #     'clf__learning_rate': [0.1,0.5,1.0]
        # },
        n_iter_search = 50
    )

    lookback_time = 3*365*24*60 #Train for three years
    run_live_strategy(
        strat,
        symbols=symbols,
        timeframe = TimeFrame.Day,
        lookback_minutes=lookback_time, #In minutes regardless of timeframe
        interval_seconds=None, #None if intended to run only once
        cash_per_trade=10000,
        feed=None
        )