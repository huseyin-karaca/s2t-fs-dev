import mlflow
import numpy as np
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.utils.logger import custom_logger as logger


def run_nested_evaluation(models, X_train, Y_train, X_test, Y_test, search_params):
    """
    Runs OptunaSearchCV for each model under a nested MLflow run.
    Expects to be called inside an active parent mlflow.start_run() context.
    """
    seed = search_params["seed"]
    n_trials = search_params["num_samples"]

    cv = ShuffleSplit(
        n_splits=search_params["num_folds"],
        test_size=search_params["test_size"],
        random_state=seed,
    )

    mlflow.sklearn.autolog(log_models=False, silent=True)

    wers, searches = {}, {}

    for model_name, (model_instance, model_hpt_space) in models.items():
        with logger.contextualize(model=model_name):
            logger.bind(category="Inner-Run").info(
                f"Hyperparameter tuning of {model_name}: ({n_trials} trials)"
            )

            search = OptunaSearchCV(
                estimator=model_instance,
                param_distributions=model_hpt_space,
                cv=cv,
                n_trials=n_trials,
                random_state=seed,
                scoring=None,
                refit=search_params["refit"],
                return_train_score=True,
            )

            with mlflow.start_run(run_name=f"Model_Tuning_{model_name}", nested=True):
                search.fit(X_train, Y_train)
                searches[model_name] = search

                preds = search.best_estimator_.predict(X_test)
                wers[model_name] = float(Y_test[np.arange(len(Y_test)), preds].mean())

                logger.bind(category="HPT-Detail").success(
                    f"Tuning of {model_name} completed. Best Test WER: {wers[model_name]:.4f}"
                )

                mlflow.log_metric("custom_test_wer", wers[model_name])

                best_idx = search.best_index_
                cv_res = search.cv_results_
                mlflow.log_metric(
                    "cv_mean_test_score", -float(cv_res["mean_test_score"][best_idx])
                )

    competitors = [v for k, v in wers.items() if not k.startswith("BoostedSDTR")]
    baseline_min = min(competitors) if competitors else 1.0
    main_wer = wers.get("BoostedSDTR", 0.0)
    margin = (
        ((main_wer - baseline_min) / baseline_min) * 100
        if baseline_min > 0 and main_wer > 0
        else 0.0
    )

    return margin, wers, searches
