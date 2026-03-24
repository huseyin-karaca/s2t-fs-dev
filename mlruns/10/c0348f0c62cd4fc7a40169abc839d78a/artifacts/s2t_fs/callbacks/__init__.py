"""
Post-experiment callback system.

Callbacks are config-driven classes invoked after model fitting.
Each callback receives the fitted estimator, MLflow run context, and dataset info,
and can log custom artifacts (figures, metrics, JSON) without polluting experiment code.

Usage in config:
    "callbacks": [
        {
            "class_path": "s2t_fs.callbacks.plot_feature_importances.PlotFeatureImportances",
            "init_args": {"n_informative": 5}
        }
    ]
"""

from s2t_fs.models.registry import instantiate_from_path
from s2t_fs.utils.logger import custom_logger as logger


def run_callbacks(callbacks_config, best_estimator, run_id, dataset_stats=None):
    """
    Invoke all callbacks defined in the config.

    Parameters
    ----------
    callbacks_config : list[dict]
        List of callback specs, each with 'class_path' and optional 'init_args'.
    best_estimator : BaseEstimator
        The fitted best estimator from HPT.
    run_id : str
        MLflow run ID for artifact logging.
    dataset_stats : dict, optional
        Dataset statistics from the data loader.
    """
    if not callbacks_config:
        return

    for cb_spec in callbacks_config:
        class_path = cb_spec["class_path"]
        init_args = cb_spec.get("init_args", {})

        try:
            callback = instantiate_from_path(class_path, **init_args)
            callback(best_estimator, run_id, dataset_stats)
            logger.bind(category="Callback").success(
                f"Callback {class_path.rsplit('.', 1)[-1]} completed."
            )
        except Exception as exc:
            logger.bind(category="Callback").error(
                f"Callback {class_path} failed: {exc}"
            )
