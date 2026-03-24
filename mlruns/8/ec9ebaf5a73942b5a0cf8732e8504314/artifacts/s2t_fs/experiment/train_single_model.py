import argparse
import json
import os

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata, log_experiment_results


def hpt_single_model(
    cfg,
    config_path=None,
    preloaded_data=None,
    run_id: str | None = None,
):
    """
    Runs hyperparameter tuning for a single model via OptunaSearchCV.

    Modes
    -----
    - **Standalone** (``preloaded_data=None``, ``run_id=None``):
      Sets up MLflow, loads data, and opens its own child run via the fluent API.
    - **Parallel-model orchestration** (``run_id`` provided):
      Uses an already-created MLflow run (passed explicitly). All logging goes
      through ``MlflowClient`` to remain thread/process safe.  OptunaSearchCV
      runs with ``n_jobs=1`` so TPE receives sequential feedback from each trial.

    Parameters
    ----------
    cfg : dict
        Single-model experiment config (mlflow_params, data_params, search_params, model_params).
    config_path : str, optional
        Path to the config file on disk, for artifact logging.
    preloaded_data : tuple, optional
        Pre-loaded (X_train, Y_train, X_test, Y_test, dataset_stats). When provided,
        skips data loading.
    run_id : str, optional
        Explicit MLflow run ID. When set, logging bypasses the fluent API entirely.
    """
    mlflow_params = cfg["mlflow_params"]
    model_params = cfg["model_params"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]

    is_standalone = preloaded_data is None
    parallel_mode = run_id is not None

    # --- MLflow Setup (standalone only) ---
    if is_standalone and not parallel_mode:
        if "MLFLOW_RUN_ID" not in os.environ:
            mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
            mlflow.set_experiment(mlflow_params["experiment_name"])

    # --- Model & Data ---
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

    # --- Search Object ---
    if not model_hpt_space:
        logger.bind(category="Inner-Run").info(
            f"No hyperparameter space provided for {model_name}. "
            "Skipping Optuna HPT - evaluating instantly."
        )
        search = None
        best_estimator = model_instance
    else:
        # n_jobs=1 → sequential trials so TPE sampler builds on previous results
        n_jobs = 1 if parallel_mode else -1
        mode_label = "sequential trials, parallel models" if parallel_mode else "parallel trials, sequential models"
        logger.bind(category="Inner-Run").info(
            f"Hyperparameter tuning of {model_name}: ({n_trials} trials, {mode_label})"
        )

        search = OptunaSearchCV(
            estimator=model_instance,
            param_distributions=model_hpt_space,
            cv=cv,
            n_trials=n_trials,
            n_jobs=n_jobs,
            random_state=seed,
            scoring=None,
            refit=search_params["refit"],
            return_train_score=True,
        )

    # --- Logging & Fitting ---
    def _log_and_fit(active_run_id):
        log_experiment_metadata(cfg, config_path, run_id=active_run_id)

        if search is not None:
            search.fit(X_train, Y_train)
            best_est = search.best_estimator_
        else:
            best_estimator.fit(X_train, Y_train)
            best_est = best_estimator
            if active_run_id:
                client = MlflowClient()
                for k, v in best_est.get_params().items():
                    try:
                        client.log_param(active_run_id, k, str(v))
                    except Exception:
                        pass
            else:
                mlflow.log_params(best_est.get_params())

        preds = best_est.predict(X_test)
        wer = float(Y_test[np.arange(len(Y_test)), preds].mean())

        if search is not None:
            log_experiment_results(search, wer, dataset_stats, run_id=active_run_id)

        # Always log the final test WER explicitly for MLflow UI comparisons
        if active_run_id:
            MlflowClient().log_metric(active_run_id, "custom_test_wer", wer)
        else:
            mlflow.log_metric("custom_test_wer", wer)

        logger.bind(category="HPT-Detail").success(
            f"Execution of {model_name} completed. Final Test WER: {wer:.4f}"
        )
        return wer

    if parallel_mode:
        # run_id was created and will be finalized by the orchestrator
        wer_best_model = _log_and_fit(run_id)
    else:
        with mlflow.start_run(run_name=f"Model_Tuning_{model_name}", nested=not is_standalone):
            wer_best_model = _log_and_fit(active_run_id=None)

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
