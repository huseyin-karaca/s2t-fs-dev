import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomModel(BaseEstimator, ClassifierMixin):
    """Predicts a random base model."""

    def __init__(self, seed=42):
        self.seed = seed
        self.num_classes_ = None

    def fit(self, X, y):
        self.num_classes_ = y.shape[1]
        return self

    def predict(self, X):
        np.random.seed(self.seed)
        return np.random.randint(0, self.num_classes_, size=(X.shape[0],))

    def predict_proba(self, X):
        np.random.seed(self.seed)
        proba = np.random.uniform(size=(X.shape[0], self.num_classes_))
        return proba / proba.sum(axis=1, keepdims=True)

    def score(self, X, y):
        """
        Returns negative Mean Word Error Rate (WER).
        Sklearn meta-estimators expect 'greater is better'.
        """
        preds = self.predict(X)
        actual_wers = y[np.arange(len(y)), preds]
        return -actual_wers.mean()
