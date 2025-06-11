# src/strategies/thresholded_regressor.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdedPredictor(BaseEstimator, ClassifierMixin):
    """
    Wraps a regressor, applies a threshold to its continuous outputs
    to produce discrete signals {-1, 0, +1}.
    """
    def __init__(self, base_regressor, threshold):
        self.base_regressor = base_regressor
        self.threshold = threshold

    def fit(self, X, y):
        # Delegate fitting
        self.base_regressor.fit(X, y)
        return self

    def predict(self, X):
        # Predict continuous returns
        pred = self.base_regressor.predict(X)
        # Apply threshold
        out = np.zeros_like(pred)
        out[pred >  self.threshold] =  1
        out[pred < -self.threshold] = -1
        return out
