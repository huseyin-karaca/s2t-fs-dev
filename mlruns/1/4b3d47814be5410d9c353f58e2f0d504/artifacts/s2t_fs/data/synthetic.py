"""
Synthetic data generation for FASTT framework validation.

Produces data in the same format as load_and_prepare_data:
    (X_train, Y_train, X_test, Y_test, dataset_stats)

The synthetic setup: N samples, p features (some informative, rest noise),
L experts. Expert performance depends only on the informative features.
This simulates high-dimensional acoustic embeddings where only a subset
of features is relevant for predicting which STT model is best.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from s2t_fs.utils.logger import custom_logger as logger


def generate_synthetic_data(data_params):
    """Generate synthetic features and WER matrix with known structure.

    Parameters (from data_params)
    ----------
    n_samples : int
    n_informative : int
    n_noise : int
    n_experts : int
    test_size : float
    seed : int

    Returns
    -------
    X_train, Y_train, X_test, Y_test, dataset_stats
        Same format as load_and_prepare_data for seamless integration.
    """
    n_samples = data_params["n_samples"]
    n_informative = data_params["n_informative"]
    n_noise = data_params["n_noise"]
    n_experts = data_params["n_experts"]
    test_size = data_params["test_size"]
    seed = data_params["seed"]

    rng = np.random.default_rng(seed)
    n_features = n_informative + n_noise

    logger.bind(category="Data").info(
        f"Generating synthetic data: {n_samples} samples, "
        f"{n_features} features ({n_informative} informative, {n_noise} noise), "
        f"{n_experts} experts"
    )

    X_informative = rng.standard_normal((n_samples, n_informative)).astype(np.float32)
    X_noise = rng.standard_normal((n_samples, n_noise)).astype(np.float32) * 0.5
    X = np.hstack([X_informative, X_noise])

    # Each expert has a preference direction in the informative subspace
    expert_directions = rng.standard_normal((n_experts, n_informative)).astype(np.float32)
    for j in range(n_experts):
        expert_directions[j] /= np.linalg.norm(expert_directions[j]) + 1e-8

    suitability = X_informative @ expert_directions.T * 1.5

    base_wer = 0.25
    wer_range = 0.50
    Y = base_wer + wer_range * _sigmoid(-suitability)
    Y += rng.normal(0, 0.01, Y.shape).astype(np.float32)
    Y = np.clip(Y, 0.01, 1.0).astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )

    dataset_stats = {
        "source": "synthetic",
        "num_features": n_features,
        "num_informative": n_informative,
        "num_noise": n_noise,
        "num_experts": n_experts,
        "num_total_rows": n_samples,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    logger.bind(category="Data").success(
        f"Synthetic data ready. Train: {len(X_train)}, Test: {len(X_test)}, "
        f"Oracle WER: {Y_test.min(axis=1).mean():.4f}"
    )

    return X_train, Y_train, X_test, Y_test, dataset_stats


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
