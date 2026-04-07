"""
Level 3 — Margin Optimization Study.

Uses Optuna's ``study.optimize()`` to search over data/search hyperparameters
(when given as lists in the config). Each trial runs a full Level 2 multi-model
comparison and returns the WER margin as the objective value.

Goal: maximize margin (find configs where our model dominates competitors most).

MLflow hierarchy:
    Margin_Optimization_Study (parent)
      └── Trial_N_Comparison (child)
           ├── Model_Tuning_A (grandchild)
           ├── Model_Tuning_B (grandchild)
           └── ...
"""

import argparse
import json
import os

import mlflow
import optuna
from mlflow.tracking import MlflowClient

from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.experiment.train_multi_model import multi_model_experiment
from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.utils.mlflow_utils import log_experiment_metadata


def _resolve_params(params_dict, trial, prefix):
    """
    For each key in params_dict: if the value is a list, ask Optuna to pick;
    if scalar, pass through unchanged.

    Returns a resolved dict with concrete values for this trial.
    """
    resolved = {}
    for key, value in params_dict.items():
        if isinstance(value, list):
            resolved[key] = trial.suggest_categorical(f"{prefix}.{key}", value)
        else:
            resolved[key] = value
    return resolved


def margin_optimization_experiment(cfg, config_path=None):
    """
    Run a margin optimization study.

    Parameters
    ----------
    cfg : dict
        Full configuration with ``mlflow_params``, ``optimization_params``,
        ``data_params``, ``search_params``, ``evaluation_params``, ``models``.
    config_path : str, optional
        Path to config file on disk.
    """
    mlflow_params = cfg["mlflow_params"]
    optimization_params = cfg["optimization_params"]
    data_params_template = cfg["data_params"]
    search_params_template = cfg["search_params"]
    evaluation_params = cfg["evaluation_params"]
    models = cfg["models"]
    callbacks_config = cfg.get("callbacks")

    n_trials = optimization_params["n_trials"]
    seed = optimization_params.get("seed", 42)

    # --- MLflow setup ---
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
        mlflow.set_experiment(mlflow_params["experiment_name"])

    tracking_uri = mlflow.get_tracking_uri()

    with mlflow.start_run(run_name="Margin_Optimization_Study") as parent_run:
        parent_run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id

        log_experiment_metadata(cfg, config_path, run_id=parent_run_id)

        def objective(trial):
            # Resolve list-valued params via Optuna suggestions
            data_params = _resolve_params(data_params_template, trial, "data")
            search_params = _resolve_params(search_params_template, trial, "search")

            # Pre-load data with this trial's data params
            logger.bind(category="Margin-Opt").info(
                f"Trial {trial.number}: data_params={data_params}, search_params={search_params}"
            )
            preloaded_data = load_and_prepare_data(data_params)

            # Create a child run for this trial's comparison
            client = MlflowClient()
            trial_run = client.create_run(
                experiment_id=experiment_id,
                run_name=f"Trial_{trial.number}_Comparison",
                tags={"mlflow.parentRunId": parent_run_id},
            )
            trial_run_id = trial_run.info.run_id

            try:
                results, margin = multi_model_experiment(
                    models=models,
                    data_params=data_params,
                    search_params=search_params,
                    evaluation_params=evaluation_params,
                    mlflow_params=mlflow_params,
                    callbacks_config=callbacks_config,
                    preloaded_data=preloaded_data,
                    run_id=trial_run_id,
                    tracking_uri=tracking_uri,
                    config_path=config_path,
                )

                # Log trial-level params
                for k, v in data_params.items():
                    try:
                        client.log_param(trial_run_id, f"data.{k}", str(v)[:250])
                    except Exception:
                        pass
                for k, v in search_params.items():
                    try:
                        client.log_param(trial_run_id, f"search.{k}", str(v)[:250])
                    except Exception:
                        pass

                client.set_terminated(trial_run_id, status="FINISHED")

                if margin is None:
                    logger.bind(category="Margin-Opt").warning(
                        f"Trial {trial.number}: margin computation failed."
                    )
                    return float("-inf")

                logger.bind(category="Margin-Opt").success(
                    f"Trial {trial.number}: margin = {margin:+.4f}"
                )
                return margin

            except Exception as exc:
                client.set_terminated(trial_run_id, status="FAILED")
                logger.bind(category="Margin-Opt").error(
                    f"Trial {trial.number} failed: {exc}"
                )
                raise optuna.TrialPruned()

        # Create Optuna study — maximize margin (our model winning more)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # n_jobs=1 → sequential trials for Bayesian feedback
        # (within each trial, multi-model comparison runs models in parallel)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        # Log best trial results to parent run
        best = study.best_trial
        client = MlflowClient()
        client.log_metric(parent_run_id, "best_margin", best.value)
        client.log_param(parent_run_id, "best_trial_number", best.number)

        for k, v in best.params.items():
            try:
                client.log_param(parent_run_id, f"best.{k}", str(v)[:250])
            except Exception:
                pass

        # Log full study summary
        import tempfile

        study_summary = {
            "best_margin": best.value,
            "best_trial_number": best.number,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(study_summary, f, indent=2, default=str)
            tmp = f.name
        client.log_artifact(parent_run_id, tmp, artifact_path="")
        os.remove(tmp)

        logger.bind(category="Margin-Opt").success(
            f"Study complete! Best margin: {best.value:+.4f} "
            f"(trial {best.number}, params: {best.params})"
        )

        return {
            "best_margin": best.value,
            "best_trial_number": best.number,
            "best_params": dict(best.params),
            "parent_run_id": parent_run_id,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Margin Optimization Study with MLflow Tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    margin_optimization_experiment(cfg, config_path=args.config)
