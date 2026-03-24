import os
import sys
import subprocess

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from s2t_fs.utils.dict_utils import flatten_dict
from s2t_fs.utils.logger import custom_logger as logger


def _client() -> MlflowClient:
    """Return a MlflowClient bound to the currently configured tracking URI."""
    return MlflowClient()


def log_experiment_metadata(cfg, config_path=None, run_id: str | None = None):
    """
    Logs configuration, artifacts, and git commit details to MLflow.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration dictionary.
    config_path : str, optional
        Path to the config JSON file on disk (logged as an artifact).
    run_id : str, optional
        Explicit MLflow run ID. When provided (multi-model parallel mode),
        uses MlflowClient directly to avoid thread-local state conflicts.
        When None, falls back to the fluent API (standalone / single-process mode).
    """
    if run_id is not None:
        client = _client()
        # Log flattened params
        for k, v in flatten_dict(cfg).items():
            try:
                client.log_param(run_id, k, v)
            except Exception:
                pass

        # Config artifact
        if config_path and os.path.exists(config_path):
            client.log_artifact(run_id, config_path, artifact_path="")
        else:
            import json, tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(cfg, f, indent=2)
                tmp = f.name
            client.log_artifact(run_id, tmp, artifact_path="")
            os.remove(tmp)

        # Codebase
        s2t_fs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if os.path.exists(s2t_fs_dir):
            client.log_artifacts(run_id, s2t_fs_dir, artifact_path="s2t_fs")

        # Git commit
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
            if commit_hash:
                client.set_tag(run_id, "mlflow.source.git.commit", commit_hash)
        except Exception:
            pass

    else:
        # Fluent API path (standalone single-model runs)
        mlflow.log_params(flatten_dict(cfg))

        if config_path and os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="")
        else:
            mlflow.log_dict(cfg, "config.json")

        main_script = sys.argv[0]
        if os.path.exists(main_script):
            mlflow.log_artifact(main_script, artifact_path="codebase")

        s2t_fs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if os.path.exists(s2t_fs_dir):
            mlflow.log_artifacts(s2t_fs_dir, artifact_path="s2t_fs")

        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
            if commit_hash:
                mlflow.set_tag("mlflow.source.git.commit", commit_hash)
        except Exception:
            pass


def log_experiment_results(search, wer_test, dataset_stats, run_id: str | None = None):
    """
    Extracts the best trial's CV results and logs metrics/artifacts to MLflow.

    Parameters
    ----------
    search : OptunaSearchCV
        Fitted search object.
    wer_test : float
        Final test WER of the best estimator.
    dataset_stats : dict
        Dataset statistics returned by the data loader.
    run_id : str, optional
        Explicit MLflow run ID (parallel model mode).
    """
    best_idx = search.best_index_
    cv_res = search.cv_results_

    def get_cv_stat(key):
        val = cv_res.get(key)
        return float(val[best_idx]) if val is not None else None

    def get_cv_stat_neg(key):
        val = get_cv_stat(key)
        return -val if val is not None else None

    results_dict = {
        "cv_mean_train_time": get_cv_stat("mean_fit_time"),
        "cv_mean_test_time": get_cv_stat("mean_score_time"),
        "cv_mean_train_wer": get_cv_stat_neg("mean_train_score"),
        "cv_mean_test_wer": get_cv_stat_neg("mean_test_score"),
        "wer_test": float(wer_test),
        "dataset_statistics": dataset_stats,
    }
    results_dict = {k: v for k, v in results_dict.items() if v is not None}

    if run_id is not None:
        client = _client()
        import json, tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(results_dict, f, indent=2)
            tmp = f.name
        client.log_artifact(run_id, tmp, artifact_path="results")
        os.remove(tmp)

        for key, val in results_dict.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                client.log_metric(run_id, key, val)
    else:
        mlflow.log_dict(results_dict, "results.json")
        for key, val in results_dict.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                mlflow.log_metric(key, val)


def compute_and_log_margin(results, target_model_name, run_id):
    """
    Compute the WER margin between our model and the best competitor,
    and log the results to an MLflow run.

    Margin = best_competitor_wer - our_wer (positive means we win).

    Parameters
    ----------
    results : dict[str, float]
        Mapping of model_name → best_wer.
    target_model_name : str
        Name of our model in the results dict.
    run_id : str
        MLflow run ID to log metrics/artifacts to.

    Returns
    -------
    float or None
        The computed margin, or None if computation wasn't possible.
    """
    if target_model_name not in results:
        logger.error(f"Target model '{target_model_name}' not found in results!")
        return None

    our_wer = results[target_model_name]
    competitors = {n: w for n, w in results.items() if n != target_model_name}

    if not competitors:
        logger.warning("No competitors found to calculate margin.")
        return None

    best_competitor_name = min(competitors, key=competitors.get)
    best_competitor_wer = competitors[best_competitor_name]
    margin = best_competitor_wer - our_wer

    client = MlflowClient()
    client.log_metric(run_id, "our_wer", our_wer)
    client.log_metric(run_id, "best_competitor_wer", best_competitor_wer)
    client.log_metric(run_id, "wer_margin", margin)

    import json as _json
    import tempfile

    summary = {
        "target_model": target_model_name,
        "our_wer": our_wer,
        "best_competitor": best_competitor_name,
        "best_competitor_wer": best_competitor_wer,
        "wer_margin": margin,
        "all_results": results,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(summary, f, indent=2)
        tmp = f.name
    client.log_artifact(run_id, tmp, artifact_path="")
    os.remove(tmp)

    logger.bind(category="Margin").success(
        f"Our WER ({target_model_name}): {our_wer:.4f} | "
        f"Best Competitor ({best_competitor_name}): {best_competitor_wer:.4f} | "
        f"Margin: {margin:+.4f}"
    )

    return margin
