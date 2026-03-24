"""
Unified entry point for all S2T-FS experiments.

Usage:
    python -m s2t_fs.experiment --config configs/some_config.json
    python -m s2t_fs.experiment --config configs/some_config.json --mode single

Mode auto-detection:
    1. If data/search params contain list values → ``margin``
    2. If ``models`` dict has multiple entries → ``multi``
    3. Otherwise → ``single``
"""

import argparse
import json
import os

import mlflow

from s2t_fs.experiment.train_margin_optimization import margin_optimization_experiment
from s2t_fs.experiment.train_multi_model import multi_model_experiment
from s2t_fs.experiment.train_single_model import hpt_single_model
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.torch_utils import log_hardware_info


def _has_list_values(params_dict):
    """Check if any value in the dict is a list (indicating search space)."""
    return any(isinstance(v, list) for v in params_dict.values())


def _detect_mode(cfg):
    """Auto-detect experiment mode from config structure."""
    # Margin optimization: list-valued data/search params + optimization_params
    if "optimization_params" in cfg:
        return "margin"

    data_params = cfg.get("data_params", {})
    search_params = cfg.get("search_params", {})
    if _has_list_values(data_params) or _has_list_values(search_params):
        return "margin"

    # Multi-model: multiple models in the dict
    models = cfg.get("models", {})
    if len(models) > 1:
        return "multi"

    return "single"


def main():
    parser = argparse.ArgumentParser(
        description="S2T-FS Experiment Runner — HPT with MLflow Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi", "margin"],
        default=None,
        help="Experiment mode. Auto-detected from config if not specified.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    mode = args.mode or _detect_mode(cfg)

    log_hardware_info(logger)

    # --- Common MLflow setup ---
    mlflow_params = cfg["mlflow_params"]
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    if mode == "margin":
        margin_optimization_experiment(cfg, config_path=args.config)

    elif mode == "multi":
        multi_model_experiment(
            models=cfg["models"],
            data_params=cfg["data_params"],
            search_params=cfg["search_params"],
            evaluation_params=cfg["evaluation_params"],
            mlflow_params=mlflow_params,
            callbacks_config=cfg.get("callbacks"),
            config_path=args.config,
        )

    else:  # single
        models = cfg["models"]
        model_name = next(iter(models))
        model_cfg = models[model_name]
        hpt_single_model(
            model_name=model_name,
            model_cfg=model_cfg,
            data_params=cfg["data_params"],
            search_params=cfg["search_params"],
            callbacks_config=cfg.get("callbacks"),
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
