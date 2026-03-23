"""
Synthetic validation experiment for FASTT framework.

Follows the same structure as train_multi_model.py:
  - Config-driven
  - MLflow parent/child runs
  - Loguru logging
  - Per-model WER metrics on parent run
  - Margin computation

Uses direct fit/evaluate (no OptunaSearchCV) since this is a validation
experiment with fixed hyperparameters, not a tuning run.
"""

import argparse
import json

import mlflow
import numpy as np

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.models.registry import instantiate_model_from_config
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata


def _evaluate_single_model(model_name, model_cfg, preloaded_data, config_path=None):
    """Fit and evaluate a single model inside a child MLflow run."""
    X_train, Y_train, X_test, Y_test, dataset_stats = preloaded_data

    model = instantiate_model_from_config(model_cfg)

    with mlflow.start_run(run_name=f"Model_{model_name}", nested=True):
        logger.bind(category="Inner-Run").info(
            f"Fitting {model_name} ({model_cfg['class_path']})..."
        )

        model.fit(X_train, Y_train)

        preds = model.predict(X_test)
        wer = float(Y_test[np.arange(len(Y_test)), preds].mean())

        mlflow.log_metric("test_wer", wer)
        mlflow.log_params({"model_name": model_name, "class_path": model_cfg["class_path"]})
        mlflow.log_dict({"test_wer": wer, "dataset_stats": dataset_stats}, "results.json")

        # Log gating weights for FASTT models with diagonal transform
        _log_gating_weights(model, model_name)

        logger.bind(category="Inner-Run").success(
            f"{model_name} — Test WER: {wer:.4f}"
        )

    return wer


def _log_gating_weights(model, model_name):
    """Log diagonal gating analysis if applicable."""
    gating = None
    if hasattr(model, "get_gating_weights"):
        gating = model.get_gating_weights()

    if gating is None:
        return

    weights = gating[0] if isinstance(gating, list) else gating
    abs_w = np.abs(weights)
    mlflow.log_dict(
        {"gating_weights": abs_w.tolist()},
        f"gating_weights_{model_name}.json",
    )

    logger.bind(category="Gating").info(
        f"{model_name} — |q| mean: {abs_w.mean():.4f}, "
        f"std: {abs_w.std():.4f}, max: {abs_w.max():.4f}, min: {abs_w.min():.4f}"
    )


def synthetic_experiment(cfg, config_path=None):
    """Run FASTT synthetic validation benchmark.

    Same orchestration pattern as multi_model_experiment:
    parent run logs cross-model metrics, each model gets a child run.
    """
    mlflow_params = cfg["mlflow_params"]
    data_params = cfg["data_params"]
    evaluation_params = cfg["evaluation_params"]
    models_dict = cfg["models"]

    target_model_name = evaluation_params["target_model_name"]

    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    logger.bind(category="Data").info("Loading data for synthetic validation...")
    preloaded_data = load_and_prepare_data(data_params)
    logger.bind(category="Data").success("Data ready.")

    results = {}

    with mlflow.start_run(run_name="FASTT_Synthetic_Benchmark") as parent_run:

        log_experiment_metadata(cfg, config_path)

        for model_name, model_cfg in models_dict.items():
            logger.bind(category="Orchestrator").info(
                f"Starting evaluation of {model_name}..."
            )

            wer = _evaluate_single_model(
                model_name, model_cfg, preloaded_data, config_path
            )
            results[model_name] = wer
            mlflow.log_metric(f"wer_{model_name}", wer)

            logger.bind(category="Orchestrator").success(
                f"Finished {model_name}. WER: {wer:.4f}"
            )

        # ── Margin computation (same logic as train_multi_model) ─────────
        if target_model_name not in results:
            logger.error(f"Target model '{target_model_name}' not found in results!")
            return results

        our_wer = results[target_model_name]
        competitors = {
            name: wer for name, wer in results.items() if name != target_model_name
        }

        if competitors:
            best_competitor_name = min(competitors, key=competitors.get)
            best_competitor_wer = competitors[best_competitor_name]
            margin = best_competitor_wer - our_wer

            _, Y_train, _, Y_test, _ = preloaded_data
            oracle_wer = float(Y_test.min(axis=1).mean())

            mlflow.log_metrics({
                "our_wer": our_wer,
                "best_competitor_wer": best_competitor_wer,
                "wer_margin": margin,
                "oracle_wer": oracle_wer,
            })

            summary = {
                "target_model": target_model_name,
                "our_wer": our_wer,
                "best_competitor": best_competitor_name,
                "best_competitor_wer": best_competitor_wer,
                "wer_margin": margin,
                "oracle_wer": oracle_wer,
                "all_results": results,
            }
            mlflow.log_dict(summary, "benchmark_summary.json")

            logger.bind(category="Margin").success(
                f"Our WER ({target_model_name}): {our_wer:.4f} | "
                f"Best Competitor ({best_competitor_name}): {best_competitor_wer:.4f} | "
                f"Margin: {margin:+.4f} | Oracle: {oracle_wer:.4f}"
            )
        else:
            logger.warning("No competitors found to calculate margin.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FASTT Synthetic Validation with MLflow Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fastt_synthetic_config.json",
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    synthetic_experiment(cfg, config_path=args.config)
