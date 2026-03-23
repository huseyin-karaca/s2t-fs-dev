import lightgbm as lgb
import numpy as np

def obj(preds, train_data):
    print("OBJ PREDS SHAPE:", preds.shape)
    grad = np.zeros_like(preds)
    hess = np.ones_like(preds)
    return grad, hess

X = np.random.rand(100, 5)
y = np.random.randint(0, 3, 100)
dtrain = lgb.Dataset(X, label=y)
lgb.train({"objective": obj, "num_class": 3, "min_data_in_bin": 1, "min_data_in_leaf": 1}, dtrain, num_boost_round=1)
