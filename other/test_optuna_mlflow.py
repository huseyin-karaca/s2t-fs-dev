import mlflow
import optuna
from optuna.integration import OptunaSearchCV
from optuna.integration.mlflow import MLflowCallback
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=100)
mlflow.set_experiment("test_optuna")

mlflow_callback = MLflowCallback(
    create_experiment=False,
    mlflow_kwargs={"nested": True}
)

search = OptunaSearchCV(
    estimator=SVC(),
    param_distributions={"C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True)},
    cv=3,
    n_trials=2,
    callbacks=[mlflow_callback]
)

with mlflow.start_run(run_name="Parent_Run"):
    search.fit(X, y)
    print("Done!")
