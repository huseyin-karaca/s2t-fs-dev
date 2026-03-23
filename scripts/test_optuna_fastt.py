import numpy as np
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from s2t_fs.models.fastt.fastt_alternating import FASTTAlternating
from s2t_fs.models.fastt.fastt_boosted import FASTTBoosted

class MockSelector:
    def __init__(self, max_depth=3, learning_rate=0.1):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "learning_rate": self.learning_rate}
        
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        self.classes_ = np.arange(y.shape[1])
        return self
        
    def predict_proba(self, X):
        return np.ones((X.shape[0], len(self.classes_))) / len(self.classes_)
        
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

def main():
    print("Generating dummy data...")
    X = np.random.randn(100, 5)
    y_wer = np.random.rand(100, 3) # (n_samples, n_classes)

    print("\n1. Testing FASTTBoosted with OptunaSearchCV")
    fastt_b = FASTTBoosted(epochs=1, num_trees=1, batch_size=32)
    param_dists_b = {
        'num_rounds': optuna.distributions.IntDistribution(1, 2),
        'depth': optuna.distributions.IntDistribution(2, 3),
        'lr': optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True)
    }
    search_b = OptunaSearchCV(
        fastt_b, 
        param_dists_b, 
        cv=KFold(2), 
        n_trials=2,
        random_state=42
    )
    search_b.fit(X, y_wer)
    print("FASTTBoosted Best Params:", search_b.best_params_)


    print("\n2. Testing FASTTAlternating with OptunaSearchCV (including base model params)")
    mock_base = MockSelector()
    fastt_a = FASTTAlternating(base_selector=mock_base, num_iterations=1, transform_steps=2)
    
    param_dists_a = {
        'num_iterations': optuna.distributions.IntDistribution(1, 2),
        'transform_lr': optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        # Here is the base model parameter!
        'base_selector__max_depth': optuna.distributions.IntDistribution(2, 5),
        'base_selector__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.2)
    }
    
    search_a = OptunaSearchCV(
        fastt_a, 
        param_dists_a, 
        cv=KFold(2), 
        n_trials=2,
        random_state=42
    )
    search_a.fit(X, y_wer)
    print("FASTTAlternating Best Params:", search_a.best_params_)

if __name__ == "__main__":
    main()
