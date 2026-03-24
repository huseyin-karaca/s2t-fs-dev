"""
Level 1 — Single-Model Hyperparameter Tuning.

Runs OptunaSearchCV for a single model. Can be called:
  - **Standalone** via CLI or ``make run-single``
  - **From Level 2** (multi-model comparison) with pre-loaded data and an explicit run_id

All MLflow logging uses MlflowClient for process/thread safety.
"""

import argparse
import json
import os

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.callbacks import run_callbacks
from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata, log_experiment_results


def hpt_single_model(
    model_name,
    model_cfg,
    data_params,
    search_params,
    callbacks_config=None,
    preloaded_data=None,
    run_id=None,
    tracking_uri=None,
    config_path=None,
):
    """
    Run hyperparameter tuning for a single model via OptunaSearchCV.

    Parameters
    ----------
    model_name : str
        Human-readable model name (e.g. "FASTT_Alternating").
    model_cfg : dict
        Model spec: ``{class_path, init_args, hyperparameters}``.
    data_params : dict
        Data configuration.
    search_params : dict
        Search/CV configuration.
    callbacks_config : list[dict], optional
        Post-experiment callbacks to invoke after fitting.
    preloaded_data : tuple, optional
        ``(X_train, Y_train, X_test, Y_test, dataset_stats)``. Skips loading when provided.
    run_id : str, optional
        Explicit MLflow run ID (when called from Level 2). When None, creates its own run.
    tracking_uri : str, optional
        MLflow tracking URI (needed in subprocess contexts).
    config_path : str, optional
        Path to config file on disk, for artifact logging.

    Returns
    -------
    float
        Best test WER of the tuned model.
    """
    # --- Reconstruct model_params dict for registry compatibility ---
    model_params = {"model_name": model_name, **model_cfg}
    _, model_instance, model_hpt_space = prepare_model_from_config(model_params)

    # --- Data ---
    if preloaded_data is None:
        X_train, Y_train, X_test, Y_test, dataset_stats = load_and_prepare_data(data_params)
    else:
        X_train, Y_train, X_test, Y_test, dataset_stats = preloaded_data

    # --- Build full config dict for metadata logging ---
    full_cfg = {
        "data_params": data_params,
        "search_params": search_params,
        "models": {model_name: model_cfg},
    }

    seed = search_params["seed"]
    n_trials = search_params["num_samples"]

    cv = ShuffleSplit(
        n_splits=search_params["num_folds"],
        test_size=search_params["test_size"],
        random_state=seed,
    )

    # --- Search Object ---
    parallel_mode = run_id is not None

    if not model_hpt_space:
        logger.bind(category="Inner-Run").info(
            f"No hyperparameter space for {model_name}. Evaluating with default params."
        )
        search = None
        best_estimator = model_instance
    else:
        # Respect the "parallel" flag from search_params:
        #   parallel=True  → n_jobs=-1 (parallel trials, faster wall-clock)
        #   parallel=False → n_jobs=1  (sequential trials, better TPE feedback)
        parallel = search_params.get("parallel", True)
        n_jobs = -1 if parallel else 1
        mode_label = "parallel" if parallel else "sequential"
        logger.bind(category="Inner-Run").info(
            f"HPT of {model_name}: {n_trials} trials ({mode_label} trials)"
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

    # --- Core: log, fit, evaluate ---
    def _log_and_fit(active_run_id):
        log_experiment_metadata(full_cfg, config_path, run_id=active_run_id)

        if search is not None:
            search.fit(X_train, Y_train)
            best_est = search.best_estimator_
        else:
            best_estimator.fit(X_train, Y_train)
            best_est = best_estimator
            # Log default params for models with no HPT space
            client = MlflowClient()
            for k, v in best_est.get_params().items():
                try:
                    client.log_param(active_run_id, k, str(v)[:250])
                except Exception:
                    pass

        preds = best_est.predict(X_test)
        wer = float(Y_test[np.arange(len(Y_test)), preds].mean())

        if search is not None:
            log_experiment_results(search, wer, dataset_stats, run_id=active_run_id)

        MlflowClient().log_metric(active_run_id, "custom_test_wer", wer)

        logger.bind(category="HPT-Detail").success(
            f"{model_name} completed. Test WER: {wer:.4f}"
        )

        # Invoke callbacks
        if callbacks_config:
            run_callbacks(callbacks_config, best_est, active_run_id, dataset_stats)

        return wer

    if parallel_mode:
        # run_id was created by the orchestrator
        wer_best = _log_and_fit(run_id)
    else:
        # Standalone mode — create our own run
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if "MLFLOW_RUN_ID" not in os.environ:
            # Only set tracking when not invoked via `mlflow run`
            pass  # tracking_uri already set above if needed

        with mlflow.start_run(run_name=f"Model_Tuning_{model_name}") as run:
            wer_best = _log_and_fit(run.info.run_id)

    return wer_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Single-Model HPT with MLflow Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    mlflow_params = cfg["mlflow_params"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]
    models = cfg["models"]
    callbacks_config = cfg.get("callbacks")

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    # Single-model config: extract the one model
    model_name = next(iter(models))
    model_cfg = models[model_name]

    hpt_single_model(
        model_name=model_name,
        model_cfg=model_cfg,
        data_params=data_params,
        search_params=search_params,
        callbacks_config=callbacks_config,
        config_path=args.config,
    )
