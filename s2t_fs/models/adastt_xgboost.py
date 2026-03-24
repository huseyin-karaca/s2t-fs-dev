import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import xgboost as xgb


class AdaSTTObjective:
    def __init__(self, wer_matrix):
        self.wer_matrix = wer_matrix
        self.epsilon = 0.02

    def __call__(self, preds, dtrain):
        if preds.ndim == 1:
            rows = self.wer_matrix.shape[0]
            cols = self.wer_matrix.shape[1]
            z = preds.reshape(rows, cols)
        else:
            z = preds
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        L = np.sum(w * self.wer_matrix, axis=1, keepdims=True)
        grad = w * (self.wer_matrix - L)
        term_1 = w * (1.0 - 2.0 * w)
        term_2 = self.wer_matrix - L
        hess = np.maximum(term_1 * term_2, self.epsilon)
        flat_grad = grad.astype(np.float32).flatten()
        flat_hess = hess.astype(np.float32).flatten()
        return flat_grad, flat_hess


class AdaSTTXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=4,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        nthread=-1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.nthread = nthread
        self.random_state = random_state
        self.model_ = None
        self.num_classes_ = None

    def fit(self, X, y):
        self.num_classes_ = y.shape[1]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Pass dummy labels to satisfy some XGBoost internal checks for n_groups
        dummy_labels_train = np.random.randint(0, self.num_classes_, size=len(X_train))
        dummy_labels_val = np.random.randint(0, self.num_classes_, size=len(X_val))
        dtrain = xgb.DMatrix(X_train, label=dummy_labels_train)
        dval = xgb.DMatrix(X_val, label=dummy_labels_val)

        obj_train = AdaSTTObjective(Y_train)
        wer_map = {id(dtrain): Y_train, id(dval): Y_val}

        def eval_expected_wer(preds, dmat):
            num_class = self.num_classes_
            if preds.ndim == 1:
                z = preds.reshape(-1, num_class)
            else:
                z = preds
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            wers = wer_map.get(id(dmat))
            if wers is None:
                return "ExpectedWER", 0.0
            expected_wer = np.sum(w * wers, axis=1).mean()
            return "ExpectedWER", expected_wer

        n_threads = self.nthread if self.nthread > 0 else os.cpu_count()
        params = {
            "disable_default_eval_metric": 1,
            "num_class": self.num_classes_,
            "objective": "multi:softprob",
            "tree_method": "hist",
            "device": "cpu",
            "nthread": n_threads,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "seed": self.random_state,
        }

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=obj_train,
            evals=[(dtrain, "train"), (dval, "val")],
            custom_metric=eval_expected_wer,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        return self

    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        preds = self.model_.predict(dmat)
        if preds.ndim == 1:
            preds = preds.reshape(-1, self.num_classes_)
        z_shifted = preds - np.max(preds, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """
        Returns negative Mean Word Error Rate (WER).
        Sklearn meta-estimators expect 'greater is better'.
        """
        preds = self.predict(X)
        actual_wers = y[np.arange(len(y)), preds]
        return -actual_wers.mean()
