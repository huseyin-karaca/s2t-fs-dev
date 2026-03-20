import mlflow
import numpy as np
import os
import sys
import subprocess
import pandas as pd
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.utils.dict_utils import flatten_dict

from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.data.loader import load_and_prepare_data

import argparse
import json


def hpt_single_model(cfg, config_path=None):


    mlflow_params = cfg["mlflow_params"]
    model_params = cfg["models_config"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]


    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    # mlflow.sklearn.autolog(log_models=False, silent=True)




    model_name, model_instance, model_hpt_space = prepare_model_from_config(model_params)

    X_train, Y_train, X_test, Y_test, dataset_stats = load_and_prepare_data(data_params)

    seed = search_params["seed"]
    n_trials = search_params["num_samples"]

    cv = ShuffleSplit(
        n_splits=search_params["num_folds"],
        test_size=search_params["test_size"],
        random_state=seed,
    )

    


    logger.bind(category="Inner-Run").info(
        f"Hyperparameter tuning of {model_name}: ({n_trials} trials)"
    )

    # def custom_mlflow_callback(study, trial):
    #     trial_run_name = f"{model_name}_Trial_{trial.number}"
        
    #     with mlflow.start_run(run_name=trial_run_name, nested=True):
    #         mlflow.log_params(trial.params)
            
    #         if trial.value is not None:
    #             mlflow.log_metric("cv_score", trial.value)
                
    #         mlflow.set_tag("state", trial.state.name)

    search = OptunaSearchCV(
        estimator=model_instance,
        param_distributions=model_hpt_space,
        cv=cv,
        n_trials=n_trials,
        random_state=seed,
        scoring=None,
        refit=search_params["refit"],
        return_train_score=True,
        # callbacks=[custom_mlflow_callback],
    )

    with mlflow.start_run(run_name=f"Model_Tuning_{model_name}", nested=True):

        # ---- Robust MLflow Logging ----

        # 1. Log Config as Flattened Parameters
        mlflow.log_params(flatten_dict(cfg))
        
        # 2. Log Config JSON as Artifact
        if config_path and os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="config")

        # 3. Log Script & Codebase Artifacts
        main_script = sys.argv[0]
        if os.path.exists(main_script):
            mlflow.log_artifact(main_script, artifact_path="codebase/scripts")

        s2t_fs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 's2t_fs'))
        if os.path.exists(s2t_fs_dir):
            mlflow.log_artifacts(s2t_fs_dir, artifact_path="codebase/s2t_fs")

        # 4. Log Git Commit
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
            if commit_hash:
                mlflow.set_tag("mlflow.source.git.commit", commit_hash)
        except Exception:
            pass

        # -------------------------------
        
        search.fit(X_train, Y_train)

        # 5. Log CV Results as CSV artifact
        try:
            cv_results_df = pd.DataFrame(search.cv_results_)
            cv_csv_path = "temp_cv_results.csv"
            cv_results_df.to_csv(cv_csv_path, index=True)
            mlflow.log_artifact(cv_csv_path, artifact_path="results")
            os.remove(cv_csv_path)
        except Exception as e:
            logger.warning(f"Could not save cv_results_: {e}")

        preds = search.best_estimator_.predict(X_test)


        wer_best_model = float(Y_test[np.arange(len(Y_test)), preds].mean())

        logger.bind(category="HPT-Detail").success(
            f"Hyperparameter tuning of {model_name} has been completed. Best Test WER: {wer_best_model:.4f}"
        )
        mlflow.log_metrics({"test_wer": wer_best_model})

    return wer_best_model








if __name__ == "__main__":
    
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
    hpt_single_model(cfg, config_path=args.config)

