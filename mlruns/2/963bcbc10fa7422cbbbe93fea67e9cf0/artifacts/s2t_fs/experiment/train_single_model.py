import mlflow
import numpy as np
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.utils.mlflow_utils import log_experiment_metadata, log_experiment_results
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.data.loader import load_and_prepare_data

import argparse
import json


def hpt_single_model(cfg, config_path=None, preloaded_data=None):


    mlflow_params = cfg["mlflow_params"]
    model_params = cfg["models_config"]
    data_params = cfg["data_params"]
    search_params = cfg["search_params"]


    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    # mlflow.sklearn.autolog(log_models=False, silent=True)




    model_name, model_instance, model_hpt_space = prepare_model_from_config(model_params)

    if preloaded_data is None:
        X_train, Y_train, X_test, Y_test, dataset_stats = load_and_prepare_data(data_params)
    else:
        X_train, Y_train, X_test, Y_test, dataset_stats = preloaded_data

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

        # Log Metadata (Config, artifacts, scripts, codebase, git, etc.)
        log_experiment_metadata(cfg, config_path)
        
        # -------------------------------
        
        search.fit(X_train, Y_train)

        preds = search.best_estimator_.predict(X_test)
        wer_best_model = float(Y_test[np.arange(len(Y_test)), preds].mean())

        # Log CV Results, JSON metadata, and test metrics
        log_experiment_results(search, wer_best_model, dataset_stats)

        logger.bind(category="HPT-Detail").success(
            f"Hyperparameter tuning of {model_name} has been completed. Best Test WER: {wer_best_model:.4f}"
        )

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

