import argparse
import json

import mlflow

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.experiment.train_single_model import hpt_single_model
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata


def multi_model_experiment(cfg, config_path=None):
    mlflow_params = cfg["mlflow_params"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]
    evaluation_params = cfg["evaluation_params"]
    models_dict = cfg["models"]

    target_model_name = evaluation_params["target_model_name"]

    import os
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    logger.bind(category="Data").info("Pre-loading data for multi-model experiment...")
    preloaded_data = load_and_prepare_data(data_params)
    logger.bind(category="Data").success("Data loaded successfully.")

    results = {}

    with mlflow.start_run(run_name="Multi_Model_Benchmark") as parent_run:

        log_experiment_metadata(cfg, config_path)

        for model_name, model_cfg in models_dict.items():
            logger.bind(category="Orchestrator").info(f"Starting HPT for {model_name}...")

            single_cfg = {
                "mlflow_params": mlflow_params,
                "data_params": data_params,
                "search_params": search_params,
                "model_params": {
                    "model_name": model_name,
                    **model_cfg,
                },
            }

            best_wer = hpt_single_model(
                single_cfg,
                config_path=config_path,
                preloaded_data=preloaded_data,
            )
            results[model_name] = best_wer

            mlflow.log_metric(f"wer_{model_name}", best_wer)

            logger.bind(category="Orchestrator").success(
                f"Finished HPT for {model_name}. Best WER: {best_wer:.4f}"
            )

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

            mlflow.log_metrics({
                "our_wer": our_wer,
                "best_competitor_wer": best_competitor_wer,
                "wer_margin": margin,
            })

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
