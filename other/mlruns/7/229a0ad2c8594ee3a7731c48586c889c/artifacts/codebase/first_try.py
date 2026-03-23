import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.utils.mlflow_utils import log_experiment_metadata, log_experiment_results
from s2t_fs.data.synthetic import generate_synthetic_data
from s2t_fs.models.fastt.fastt_alternating import FASTTAlternating
from s2t_fs.models.adastt_xgboost import AdaSTTXGBoost
from s2t_fs.utils.logger import custom_logger as logger

import argparse
import json

# 1. Parse configuration from JSON file
parser = argparse.ArgumentParser(description="Run synthetic experiment with a config string/file.")
parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file.")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = json.load(f)

# 2. Generate Data using the config
X_train, Y_train, X_test, Y_test, stats = generate_synthetic_data(cfg["data_cfg"])
logger.info(f"Data generated: {stats['num_total_rows']} rows, {stats['num_features']} total features")

# 3. Parse the config securely using the registry
model_name, model_instance, optuna_space = prepare_model_from_config(cfg["model_cfg"])

# 4. Instantiate the Scikit-Learn OptunaSearchCV Estimator
search = OptunaSearchCV(
    estimator=model_instance,
    param_distributions=optuna_space,
    cv=ShuffleSplit(
        n_splits=cfg["search_cfg"]["num_folds"], 
        test_size=cfg["search_cfg"]["test_size"], 
        random_state=cfg["search_cfg"]["seed"]
    ),
    n_trials=cfg["search_cfg"]["num_samples"],
    random_state=cfg["search_cfg"]["seed"],
    refit=cfg["search_cfg"]["refit"]
)

# 5. MLflow Tracking Setup
if "MLFLOW_RUN_ID" not in os.environ:
    if "tracking_uri" in cfg["mlflow_cfg"]:
        mlflow.set_tracking_uri(cfg["mlflow_cfg"]["tracking_uri"])
    if "experiment_name" in cfg["mlflow_cfg"]:
        mlflow.set_experiment(cfg["mlflow_cfg"]["experiment_name"])

with mlflow.start_run(run_name=cfg["mlflow_cfg"]["run_name"]) as run:

    description = """
    Testing if the `FASTTAlternating` model can learn the true feature importances.
    """
    
    mlflow.set_tag("mlflow.note.content", description)

    logger.info(f"Starting Optuna search for {model_name}...")
    
    # Fit the estimator on the preloaded data
    search.fit(X_train, Y_train)
    
    # Evaluate the best estimator
    preds = search.predict(X_test)
    wer_best = float(Y_test[np.arange(len(Y_test)), preds].mean())
    logger.info(f"Best Test WER: {wer_best:.4f}")
    
    # Log config and results to MLflow using the centralized custom utility
    log_experiment_metadata(cfg)
    log_experiment_results(search, wer_best, stats)
    
    # Extract the gating weights from the best fitted model
    best_model = search.best_estimator_
    gating_weights_tuned = best_model.get_gating_weights()
    
    # 6. Plotting and Artifact Logging
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.abs(gating_weights_tuned), marker='o', alpha=0.9, color='green', label='Tuned Diagonal Weights')
    
    ax.axvline(x=4.5, color='red', linestyle='--', linewidth=2, label='Informative / Noise Boundary')
    ax.set_title('Learned Config-Driven FASTT-Alternating Gating Weights (Tuned)', fontsize=14)
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Weight Magnitude $|q_j|$', fontsize=12)
    ax.set_xticks(range(30))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Log the figure natively to MLflow as an artifact
    mlflow.log_figure(fig, "gating_weights_tuned.png")
    logger.info("Saved gating weights plot to MLflow artifacts.")
    
    # Close the figure to prevent hanging during automated MLflow runs
    plt.close(fig)