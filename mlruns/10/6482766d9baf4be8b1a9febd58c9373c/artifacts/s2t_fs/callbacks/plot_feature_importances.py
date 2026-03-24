"""
Callback: Plot learned feature importances and log to MLflow.

Extracted from the former experiment/synthetic/test_transform.py.
Works with any model that exposes a `get_feature_importances()` method.
"""

import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from s2t_fs.utils.logger import custom_logger as logger


class PlotFeatureImportances:
    """
    Post-experiment callback that extracts feature importances from the
    fitted model and logs a feature importance plot to MLflow.

    Parameters
    ----------
    n_informative : int
        Number of informative features (used to draw the boundary line).
    """

    def __init__(self, n_informative: int = 5):
        self.n_informative = n_informative

    def __call__(self, best_estimator, run_id, dataset_stats=None):
        """
        Parameters
        ----------
        best_estimator : BaseEstimator
            Fitted model with `get_feature_importances()` method.
        run_id : str
            MLflow run ID for artifact logging.
        dataset_stats : dict, optional
            Dataset statistics.
        """
        if not hasattr(best_estimator, "get_feature_importances"):
            logger.bind(category="Callback").warning(
                "Model does not expose get_feature_importances(); skipping plot."
            )
            return

        feature_importances = best_estimator.get_feature_importances()
        if feature_importances is None:
            return

        client = MlflowClient()

        # Log raw weights as JSON artifact
        if isinstance(feature_importances, list):
            weights_data = {f"round_{i}": w.tolist() for i, w in enumerate(feature_importances)}
        else:
            weights_data = {"feature_importances": feature_importances.tolist()}

        import json
        import tempfile
        import os

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(weights_data, f, indent=2)
            tmp_json = f.name
        client.log_artifact(run_id, tmp_json, artifact_path="")
        os.remove(tmp_json)

        # Create figure
        n_features = (
            len(feature_importances[0])
            if isinstance(feature_importances, list)
            else len(feature_importances)
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        if isinstance(feature_importances, list):
            for i, gw in enumerate(feature_importances):
                ax.plot(gw, marker="o", alpha=0.9, label=f"Round {i + 1}")
        else:
            ax.plot(
                feature_importances,
                marker="o",
                alpha=0.9,
                color="green",
                label="Feature Importances",
            )

        boundary = self.n_informative - 0.5
        ax.axvline(
            x=boundary,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Informative / Noise Boundary",
        )
        ax.set_title("Learned Feature Importances", fontsize=14)
        ax.set_xlabel("Feature Index", fontsize=12)
        ax.set_ylabel("Importance Magnitude", fontsize=12)
        ax.set_xticks(range(n_features))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Save to temp file and log via MlflowClient
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, dpi=150)
            tmp_png = f.name
        client.log_artifact(run_id, tmp_png, artifact_path="")
        os.remove(tmp_png)

        plt.close(fig)
        logger.bind(category="Callback").info("Feature importances plot logged to MLflow.")
