import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

class MultiTargetSelectKBest(SelectKBest):
    """
    Wrapper around SelectKBest to handle multi-dimensional targets (e.g. WER matrices).
    Converts (N, C) continuous targets into (N,) categorical targets representing the best class.
    """
    def __init__(self, score_func=f_classif, *, k=10):
        super().__init__(score_func=score_func, k=k)

    def fit(self, X, y):
        # Convert (N, C) into a 1D array of best model indices
        # WER implies lower is better. Thus, we take argmin across classes.
        if y.ndim > 1:
            y_1d = np.argmin(y, axis=1)
        else:
            y_1d = y
        # Suppress warnings for constant features (zero variance → NaN F-score).
        # These features are harmlessly ranked last and never selected.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Features .* are constant")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            return super().fit(X, y_1d)

    def partial_fit(self, X, y):
        if y.ndim > 1:
            y_1d = np.argmin(y, axis=1)
        else:
            y_1d = y
        return super().fit(X, y_1d)  # SelectKBest doesn't natively have partial_fit, but we intercept fit anyway
