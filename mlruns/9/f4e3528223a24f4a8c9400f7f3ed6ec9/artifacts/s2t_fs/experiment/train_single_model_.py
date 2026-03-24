import argparse
import json

import mlflow
import numpy as np
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata, log_experiment_results


def hpt_single_model(cfg, config_path=None, preloaded_data=None):
    """
    Runs hyperparameter tuning for a single model via OptunaSearchCV.

    When called standalone, sets up MLflow and loads data.
    When called from multi_model_experiment (preloaded_data is not None),
    skips redundant setup — the parent has already handled it.
    """
    mlflow_params = cfg["mlflow_params"]
    model_params = cfg["model_params"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]

    is_standalone = preloaded_data is None

    if is_standalone:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    model_name, model_instance, model_hpt_space = prepare_model_from_config(model_params)

    if is_standalone:
        X_train, Y_train, X_test, Y_test, dataset_stats = load_and_prepare_data(data_params)
    else:
        X_train, Y_train, X_test, Y_test, dataset_stats = preloaded_data

    seed = search_params["seed"]
    n_trials = search_params["num_samples"]

    cv = ShuffleSplit(
        n_splits=search_params["num_folds"],
        test_size=search_params["test_size"],
        random_state=seed,
    )

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

        log_experiment_metadata(cfg, config_path)

        search.fit(X_train, Y_train)

        preds = search.best_estimator_.predict(X_test)
        wer_best_model = float(Y_test[np.arange(len(Y_test)), preds].mean())

        log_experiment_results(search, wer_best_model, dataset_stats)

        logger.bind(category="HPT-Detail").success(
            f"Hyperparameter tuning of {model_name} completed. Best Test WER: {wer_best_model:.4f}"
        )

    return wer_best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Single-Model HPT with MLflow Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/xgboost_hpt_config.json",
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    hpt_single_model(cfg, config_path=args.config)