import os
import sys
import subprocess
import pandas as pd
import numpy as np
import mlflow

from s2t_fs.utils.dict_utils import flatten_dict
from s2t_fs.utils.logger import custom_logger as logger

def log_experiment_metadata(cfg, config_path=None):
    """
    Logs configuration, artifacts, codebase, and git commit details to MLflow.
    """
    # 1. Log Config as Flattened Parameters
    mlflow.log_params(flatten_dict(cfg))
    
    # 2. Log Config JSON as Artifact
    if config_path and os.path.exists(config_path):
        mlflow.log_artifact(config_path, artifact_path="")

    # 3. Log Script & Codebase Artifacts
    main_script = sys.argv[0]
    if os.path.exists(main_script):
        mlflow.log_artifact(main_script, artifact_path="codebase")

    s2t_fs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.path.exists(s2t_fs_dir):
        mlflow.log_artifacts(s2t_fs_dir, artifact_path="codebase/s2t_fs")

    # 4. Log Git Commit
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        if commit_hash:
            mlflow.set_tag("mlflow.source.git.commit", commit_hash)
    except Exception:
        pass


def log_experiment_results(search, wer_test, dataset_stats):
    """
    Extracts the best trial's CV results, dumps CV output to CSV, 
    and logs aggregated train/test metrics into results.json.
    """
    # # 1. Log all CV Results as CSV artifact
    # try:
    #     cv_results_df = pd.DataFrame(search.cv_results_)
    #     cv_csv_path = "temp_cv_results.csv"
    #     cv_results_df.to_csv(cv_csv_path, index=True)
    #     mlflow.log_artifact(cv_csv_path, artifact_path="results")
    #     os.remove(cv_csv_path)
    # except Exception as e:
    #     logger.warning(f"Could not save cv_results_ artifact: {e}")

    # Extract aggregated metrics for the best HPT trial
    best_idx = search.best_index_
    cv_res = search.cv_results_
    
    # Fallback to 0.0 if not found
    def get_cv_stat(key):
        val = cv_res.get(key)
        return float(val[best_idx]) if val is not None else None

    results_dict = {
        "cv_mean_train_time": get_cv_stat("mean_fit_time"),
        "cv_mean_test_time": get_cv_stat("mean_score_time"),
        "cv_mean_train_wer": -get_cv_stat("mean_train_score"),
        "cv_mean_test_wer": -get_cv_stat("mean_test_score"),
        "wer_test": float(wer_test),
        "dataset_statistics": dataset_stats
    }
    
    # Clean None values in case return_train_score=False was used previously
    results_dict = {k: v for k, v in results_dict.items() if v is not None}
    
    # 3. Log results.json artifact
    mlflow.log_dict(results_dict, "results.json")
    
    # 4. Separately log them directly as flat metrics for MLflow UI comparisons
    for key, val in results_dict.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            mlflow.log_metric(key, val)
