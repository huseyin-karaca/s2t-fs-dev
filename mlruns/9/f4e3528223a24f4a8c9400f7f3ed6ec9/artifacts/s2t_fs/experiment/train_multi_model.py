import argparse
import json
import os
from multiprocessing import Pool, cpu_count

import mlflow
from mlflow.tracking import MlflowClient

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.experiment.train_single_model import hpt_single_model
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process per model)
# ---------------------------------------------------------------------------

def _run_model_worker(args):
    """
    Isolated subprocess entry-point for one model's HPT.

    Creates its own MLflow child run via MlflowClient (process-safe),
    calls hpt_single_model with the explicit run_id, then finalises the run.

    Returns
    -------
    tuple[str, float]
        (model_name, best_wer)
    """
    (
        model_name,
        single_cfg,
        config_path,
        preloaded_data,
        tracking_uri,
        experiment_id,
        parent_run_id,
    ) = args

    # Each process must configure its own tracking URI
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
            single_cfg,
            config_path=config_path,
            preloaded_data=preloaded_data,
            run_id=child_run_id,
        )
        client.set_terminated(child_run_id, status="FINISHED")
        logger.bind(category="Orchestrator").success(
            f"Finished HPT for {model_name}. Best WER: {best_wer:.4f}"
        )
    except Exception as exc:
        client.set_terminated(child_run_id, status="FAILED")
        logger.bind(category="Orchestrator").error(
            f"HPT for {model_name} failed: {exc}"
        )
        raise

    return model_name, best_wer


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def multi_model_experiment(cfg, config_path=None):
    mlflow_params = cfg["mlflow_params"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]
    evaluation_params = cfg["evaluation_params"]
    models_dict = cfg["models"]

    target_model_name = evaluation_params["target_model_name"]

    # Respect MLFLOW_RUN_ID when invoked via `mlflow run`
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    # Pre-load data once in the parent process; workers receive arrays by copy
    logger.bind(category="Data").info("Pre-loading data for multi-model experiment...")
    preloaded_data = load_and_prepare_data(data_params)
    logger.bind(category="Data").success("Data loaded successfully.")

    tracking_uri = mlflow.get_tracking_uri()

    with mlflow.start_run(run_name="Multi_Model_Benchmark") as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        log_experiment_metadata(cfg, config_path, run_id=parent_run_id)

        # Build worker argument list (one per model)
        worker_args = []
        for model_name, model_cfg in models_dict.items():
            single_cfg = {
                "mlflow_params": mlflow_params,
                "data_params": data_params,
                "search_params": search_params,
                "model_params": {
                    "model_name": model_name,
                    **model_cfg,
                },
            }
            worker_args.append((
                model_name,
                single_cfg,
                config_path,
                preloaded_data,
                tracking_uri,
                experiment_id,
                parent_run_id,
            ))

        n_workers = min(len(worker_args), cpu_count())
        logger.bind(category="Orchestrator").info(
            f"Launching {len(worker_args)} model(s) across {n_workers} worker process(es)."
        )

        with Pool(processes=n_workers) as pool:
            outputs = pool.map(_run_model_worker, worker_args)

        results = dict(outputs)

        # Log per-model WER into parent run
        for model_name, best_wer in results.items():
            MlflowClient().log_metric(parent_run_id, f"wer_{model_name}", best_wer)

        # Compute and log margin
        if target_model_name not in results:
            logger.error(f"Target model '{target_model_name}' not found in results!")
            return results

        our_wer = results[target_model_name]
        competitors = {n: w for n, w in results.items() if n != target_model_name}

        if competitors:
            best_competitor_name = min(competitors, key=competitors.get)
            best_competitor_wer = competitors[best_competitor_name]
            margin = best_competitor_wer - our_wer

            client = MlflowClient()
            client.log_metric(parent_run_id, "our_wer", our_wer)
            client.log_metric(parent_run_id, "best_competitor_wer", best_competitor_wer)
            client.log_metric(parent_run_id, "wer_margin", margin)

            summary = {
                "target_model": target_model_name,
                "our_wer": our_wer,
                "best_competitor": best_competitor_name,
                "best_competitor_wer": best_competitor_wer,
                "wer_margin": margin,
                "all_results": results,
            }
            mlflow.log_dict(summary, "benchmark_summary.json")

            logger.bind(category="Margin").success(
                f"Our WER ({target_model_name}): {our_wer:.4f} | "
                f"Best Competitor ({best_competitor_name}): {best_competitor_wer:.4f} | "
                f"Margin: {margin:+.4f}"
            )
        else:
            logger.warning("No competitors found to calculate margin.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Multi-Model HPT Benchmark with MLflow Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multi_model_hpt_config.json",
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    multi_model_experiment(cfg, config_path=args.config)
