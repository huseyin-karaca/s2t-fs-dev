"""
Level 2 — Multi-Model Comparison Benchmark.

Runs Level 1 (single-model HPT) for each model in parallel via a process pool.
Can be called:
  - **Standalone** via CLI or ``make run-comparison``
  - **From Level 3** (margin optimization) with pre-loaded data and an explicit run_id

MLflow hierarchy: Comparison (parent) → per-model Tuning (children).
"""

import argparse
import json
import os
from multiprocessing import Pool, cpu_count

import mlflow
from mlflow.tracking import MlflowClient

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.experiment.train_single_model import hpt_single_model
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import compute_and_log_margin, log_experiment_metadata


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process per model)
# ---------------------------------------------------------------------------


def _run_model_worker(args):
    """
    Subprocess entry-point for one model's HPT.

    Creates a child MLflow run via MlflowClient (process-safe),
    calls hpt_single_model, then finalises the run.

    Returns
    -------
    tuple[str, float]
        (model_name, best_wer)
    """
    (
        model_name,
        model_cfg,
        data_params,
        search_params,
        callbacks_config,
        preloaded_data,
        tracking_uri,
        experiment_id,
        parent_run_id,
        config_path,
    ) = args

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=f"Model_Tuning_{model_name}",
        tags={"mlflow.parentRunId": parent_run_id},
    )
    child_run_id = run.info.run_id

    try:
        logger.bind(category="Orchestrator").info(f"Starting HPT for {model_name}...")
        best_wer = hpt_single_model(
            model_name=model_name,
            model_cfg=model_cfg,
            data_params=data_params,
            search_params=search_params,
            callbacks_config=callbacks_config,
            preloaded_data=preloaded_data,
            run_id=child_run_id,
            tracking_uri=tracking_uri,
            config_path=config_path,
        )
        client.set_terminated(child_run_id, status="FINISHED")
        logger.bind(category="Orchestrator").success(
            f"Finished HPT for {model_name}. WER: {best_wer:.4f}"
        )
    except Exception as exc:
        client.set_terminated(child_run_id, status="FAILED")
        logger.bind(category="Orchestrator").error(f"HPT for {model_name} failed: {exc}")
        raise

    return model_name, best_wer


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def multi_model_experiment(
    models,
    data_params,
    search_params,
    evaluation_params,
    mlflow_params,
    callbacks_config=None,
    preloaded_data=None,
    run_id=None,
    tracking_uri=None,
    config_path=None,
):
    """
    Run multi-model comparison benchmark.

    Parameters
    ----------
    models : dict
        ``{model_name: {class_path, init_args, hyperparameters}}``.
    data_params : dict
        Data configuration.
    search_params : dict
        Search/CV configuration.
    evaluation_params : dict
        Must contain ``target_model_name``.
    mlflow_params : dict
        MLflow tracking configuration.
    callbacks_config : list[dict], optional
        Callbacks to invoke per model after fitting.
    preloaded_data : tuple, optional
        Pre-loaded data. Skips loading when provided.
    run_id : str, optional
        Parent run ID (when called from Level 3). When None, creates its own parent.
    tracking_uri : str, optional
        MLflow tracking URI.
    config_path : str, optional
        Path to config file on disk.

    Returns
    -------
    tuple[dict, float]
        ``(results_dict, margin)`` where results_dict maps model_name → best_wer.
    """
    target_model_name = evaluation_params["target_model_name"]

    # --- MLflow setup ---
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    resolved_tracking_uri = mlflow.get_tracking_uri()

    # --- Data ---
    if preloaded_data is None:
        logger.bind(category="Data").info("Pre-loading data for multi-model experiment...")
        preloaded_data = load_and_prepare_data(data_params)
        logger.bind(category="Data").success("Data loaded successfully.")

    # --- Full config for metadata ---
    full_cfg = {
        "mlflow_params": mlflow_params,
        "data_params": data_params,
        "search_params": search_params,
        "evaluation_params": evaluation_params,
        "models": models,
    }

    def _run_comparison(parent_run_id, experiment_id):
        log_experiment_metadata(full_cfg, config_path, run_id=parent_run_id)

        # Build worker args
        worker_args = []
        for m_name, m_cfg in models.items():
            worker_args.append((
                m_name,
                m_cfg,
                data_params,
                search_params,
                callbacks_config,
                preloaded_data,
                resolved_tracking_uri,
                experiment_id,
                parent_run_id,
                config_path,
            ))

        n_workers = min(len(worker_args), cpu_count())
        logger.bind(category="Orchestrator").info(
            f"Launching {len(worker_args)} model(s) across {n_workers} worker(s)."
        )

        with Pool(processes=n_workers) as pool:
            outputs = pool.map(_run_model_worker, worker_args)

        results = dict(outputs)

        # Log per-model WER on parent
        client = MlflowClient()
        for m_name, wer in results.items():
            client.log_metric(parent_run_id, f"wer_{m_name}", wer)

        # Compute and log margin
        margin = compute_and_log_margin(results, target_model_name, parent_run_id)

        return results, margin

    if run_id is not None:
        # Called from Level 3 — run_id is the comparison child run
        experiment_id = MlflowClient().get_run(run_id).info.experiment_id
        results, margin = _run_comparison(run_id, experiment_id)
    else:
        # Standalone — create our own parent run
        with mlflow.start_run(run_name="Multi_Model_Benchmark") as parent_run:
            results, margin = _run_comparison(
                parent_run.info.run_id, parent_run.info.experiment_id
            )

    return results, margin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Multi-Model HPT Benchmark with MLflow Tracking"
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

    multi_model_experiment(
        models=cfg["models"],
        data_params=cfg["data_params"],
        search_params=cfg["search_params"],
        evaluation_params=cfg["evaluation_params"],
        mlflow_params=cfg["mlflow_params"],
        callbacks_config=cfg.get("callbacks"),
        config_path=args.config,
    )
