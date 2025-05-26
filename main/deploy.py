from alpaca.data.timeframe import TimeFrame

from src.strategies.adaboost_ML import AdaBoostStrategy
from src.strategies.momentum_ranking_adaboost_ML import MomentumRankingAdaBoostStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.execution.live_executor import run_live_strategy

if __name__ == "__main__":
    # Example: live-run moving-average on AAPL or multiple symbols. Opening times NYSE/NASDAC: 15:30-22:00 CEST
    symbols = ["AAPL", "MSFT"]
    # strat = MovingAverageStrategy(short_window=5, long_window=20, ma= 'ema')
    
    predictor = AdaBoostStrategy(
        d=10,
        train_frac=0.7,
        cv_splits=5,
        param_grid={
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.5, 1.0]
        }
    )
    strat = MomentumRankingAdaBoostStrategy(
        predictor=predictor,
        top_k=1,
        n_jobs=-1
    )

    lookback_time = 30
    run_live_strategy(
        strat,
        symbols=symbols,
        timeframe = TimeFrame.Minute,
        lookback_minutes=lookback_time, #In minutes regardless of timeframe
        interval_seconds=30,
        cash_per_trade=5000,
        feed=None
        )