"""
Entry point for running experiments via:
    python -m s2t_fs.experiment --config configs/multi_model_hpt_config.json
    python -m s2t_fs.experiment --config configs/xgboost_hpt_config.json --mode single
    python -m s2t_fs.experiment --config configs/fastt_synthetic_config.json --mode synthetic
"""

import argparse
import json

from s2t_fs.experiment.train_multi_model import multi_model_experiment
from s2t_fs.experiment.train_single_model import hpt_single_model
from s2t_fs.experiment.train_synthetic import synthetic_experiment


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
        choices=["single", "multi", "synthetic"],
        default=None,
        help="Experiment mode. Auto-detected from config if not specified.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    mode = args.mode
    if mode is None:
        if cfg.get("data_params", {}).get("source") == "synthetic":
            mode = "synthetic"
        elif "models" in cfg:
            mode = "multi"
        else:
            mode = "single"

    if mode == "synthetic":
        synthetic_experiment(cfg, config_path=args.config)
    elif mode == "multi":
        multi_model_experiment(cfg, config_path=args.config)
    else:
        hpt_single_model(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
