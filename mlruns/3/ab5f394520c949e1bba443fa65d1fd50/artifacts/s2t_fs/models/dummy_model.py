import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DummyModel(BaseEstimator, ClassifierMixin):
    """Always predicts a specific model index."""

    def __init__(self, target_index=0):
        self.target_index = target_index
        self.num_classes_ = None

    def fit(self, X, y):
        self.num_classes_ = y.shape[1]
        return self

    def predict(self, X):
        return np.full((X.shape[0],), self.target_index)

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], self.num_classes_))
        proba[:, self.target_index] = 1.0
        return proba

    def score(self, X, y):
        """
        Returns negative Mean Word Error Rate (WER).
        Sklearn meta-estimators expect 'greater is better'.
        """
        preds = self.predict(X)
        actual_wers = y[np.arange(len(y)), preds]
        return -actual_wers.mean()
