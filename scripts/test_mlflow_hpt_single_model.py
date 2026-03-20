import argparse
import json
import mlflow

from s2t_fs.experiment.train_single_model import hpt_single_model

def main():
    parser = argparse.ArgumentParser(description="Run High Parameter Tuning Single Model with MLFlow")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/xgboost_hpt_config.json", 
        help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Start tuning
    hpt_single_model(cfg)

if __name__ == "__main__":
    main()

