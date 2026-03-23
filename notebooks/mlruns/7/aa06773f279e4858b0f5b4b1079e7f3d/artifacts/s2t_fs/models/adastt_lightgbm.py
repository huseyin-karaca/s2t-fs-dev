import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import torch
import lightgbm as lgb


class AdaSTTObjectiveLGBM:
    def __init__(self, wer_matrix, num_classes):
        self.wer_matrix = wer_matrix
        self.epsilon = 0.02
        self.num_classes = num_classes

    def __call__(self, preds, train_data):
        # LightGBM >=4.0 passes multi-class preds directly as (N, K)
        z = preds
        
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        L = np.sum(w * self.wer_matrix, axis=1, keepdims=True)
        grad = w * (self.wer_matrix - L)
        
        term_1 = w * (1.0 - 2.0 * w)
        term_2 = self.wer_matrix - L
        hess = np.maximum(term_1 * term_2, self.epsilon)
        
        # Return same shape as preds (N, K) for LightGBM >=4.0
        return grad, hess


class AdaSTTLightGBM(BaseEstimator, ClassifierMixin):
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
        self.random_state = random_state
        self.model_ = None
        self.num_classes_ = None

    def fit(self, X, y):
        self.num_classes_ = y.shape[1]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        dtrain = lgb.Dataset(X_train, label=np.zeros(len(X_train)))  # dummy label
        dval = lgb.Dataset(X_val, label=np.zeros(len(X_val)), reference=dtrain)

        obj_train = AdaSTTObjectiveLGBM(Y_train, self.num_classes_)
        
        def eval_expected_wer(preds, train_data):
            z = preds
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            expected_wer = np.sum(w * Y_val, axis=1).mean()
            return "ExpectedWER", expected_wer, False

        params = {
            "num_class": self.num_classes_,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda_l2": self.reg_lambda,
            "seed": self.random_state,
            "verbose": -1,
            "device_type": "gpu" if torch.cuda.is_available() else "cpu",
        }

        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            fobj=obj_train,
            valid_sets=[dval],
            valid_names=["val"],
            feval=eval_expected_wer,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict_proba(self, X):
        # LightGBM handles probabilities naturally or raw values if custom object is used.
        # Since we used custom obj, predict() returns raw logits of shape (N, K)
        # However, due to the obj_train logic mapping to LightGBM backend, we manually softmax it.
        preds = self.model_.predict(X)
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
