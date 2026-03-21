import argparse
import json
import os
import sys

# Add the project directory to the python path so that imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import numpy as np

from s2t_fs.data.synthetic import generate_synthetic_data
from s2t_fs.models.fastt.fastt_boosted import FASTTBoosted
from s2t_fs.utils.logger import custom_logger as logger
from dotenv import load_dotenv

load_dotenv()


def test_diagonalgating():
    # Setup test parameters
    n_samples = 4000
    n_informative = 5
    n_noise = 25
    n_experts = 3
    test_size = 0.2
    seed = 42

    data_params = {
        "n_samples": n_samples,
        "n_informative": n_informative,
        "n_noise": n_noise,
        "n_experts": n_experts,
        "test_size": test_size,
        "seed": seed
    }

    # Generate synthetic data where only the first n_informative features are relevant
    logger.bind(category="Test-Data").info("Generating synthetic data for DiagonalGating test...")
    X_train, Y_train, X_test, Y_test, dataset_stats = generate_synthetic_data(data_params)

    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlruns_nested_try.db")
    mlflow.set_experiment("FASTT_DiagonalGating_Test")

    with mlflow.start_run(run_name="FASTTBoosted_Diagonal_Test"):
        mlflow.log_params(data_params)

        logger.bind(category="Test-Model").info("Initializing FASTTBoosted with diagonal transform...")
        model = FASTTBoosted(
            num_rounds=3,
            transform_type="diagonal",
            transform_kwargs={"lambda1": 0.05, "lambda2": 0.01}, # Moderate L1 to zero out noise
            num_trees=10,
            depth=3,
            lmbda=0.1,
            lr=1e-2, # Increased LR so transform weights can converge
            weight_decay=1e-4,
            epochs=100,
            patience=20,
            random_state=seed,
            batch_size=128
        )

        logger.bind(category="Test-Model").info("Fitting model...")
        model.fit(X_train, Y_train)

        # Evaluate performance
        preds = model.predict(X_test)
        test_wer = float(Y_test[np.arange(len(Y_test)), preds].mean())
        oracle_wer = float(Y_test.min(axis=1).mean())
        
        mlflow.log_metric("test_wer", test_wer)
        mlflow.log_metric("oracle_wer", oracle_wer)

        logger.bind(category="Test-Eval").success(f"Test WER: {test_wer:.4f} (Oracle: {oracle_wer:.4f})")

        # Extract and verify gating weights
        gating_weights = model.get_gating_weights()
        if gating_weights is None:
            logger.bind(category="Test-Gating").error("Gating weights are None, expected a list of weights per round.")
            assert False, "Gating weights should not be None"

        logger.bind(category="Test-Gating").info("Verifying feature selection via gating weights...")
        
        # Log and evaluate weights from each round
        passed = True
        for round_idx, round_weights in enumerate(gating_weights):
            abs_weights = np.abs(round_weights)
            
            informative_weights = abs_weights[:n_informative]
            noise_weights = abs_weights[n_informative:]

            mean_informative = float(informative_weights.mean())
            mean_noise = float(noise_weights.mean())
            
            mlflow.log_metric(f"mean_informative_w_round_{round_idx}", mean_informative)
            mlflow.log_metric(f"mean_noise_w_round_{round_idx}", mean_noise)

            logger.bind(category="Test-Gating").info(
                f"Round {round_idx} | Mean Informative W: {mean_informative:.4f} | Mean Noise W: {mean_noise:.4f}"
            )

            # We expect the model to learn that informative features are useful 
            # and push their weights higher, while noise features should be pushed to ~0
            # Let's enforce a strict margin, e.g., informative mean >= 1.5 * noise mean
            if mean_informative <= 1.5 * mean_noise:
                logger.bind(category="Test-Gating").warning(
                    f"Round {round_idx}: Informative mean is not significantly larger than Noise mean ({mean_informative:.4f} <= 1.5 * {mean_noise:.4f})"
                )
                passed = False
            
            # Additional check: zeroed out features
            num_zeroed_noise = np.sum(noise_weights < 0.1) # using 0.1 as near-zero threshold
            logger.bind(category="Test-Gating").info(
                f"Round {round_idx} | Small Noise Features (< 0.1): {num_zeroed_noise} / {n_noise}"
            )

        if passed:
            logger.bind(category="Test-Result").success("SUCCESS: DiagonalGating successfully favors informative features over noise.")
        else:
            logger.bind(category="Test-Result").error("FAILURE: DiagonalGating failed to adequately separate informative and noise features.")
            
        assert passed, "Failed to appropriately weight informative features greater than noise features."


if __name__ == "__main__":
    logger.bind(category="Test").info("Starting FASTTBoosted DiagonalGating Test Script.")
    test_diagonalgating()
