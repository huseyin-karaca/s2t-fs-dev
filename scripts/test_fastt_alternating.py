import argparse
import json
import os
import sys

# Add the project directory to the python path so that imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import numpy as np

from s2t_fs.data.synthetic import generate_synthetic_data
from s2t_fs.models.fastt.fastt_alternating import FASTTAlternating
from s2t_fs.models.adastt_xgboost import AdaSTTXGBoost
from s2t_fs.utils.logger import custom_logger as logger
from dotenv import load_dotenv

load_dotenv()


def test_fastt_alternating():
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

    # Generate synthetic data
    logger.bind(category="Test-Data").info("Generating synthetic data for FASTTAlternating test...")
    X_train, Y_train, X_test, Y_test, dataset_stats = generate_synthetic_data(data_params)

    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlruns_nested_try.db")
    mlflow.set_experiment("FASTT_Alternating_Test")

    with mlflow.start_run(run_name="FASTTAlternating_Diagonal_Test"):
        mlflow.log_params(data_params)

        logger.bind(category="Test-Model").info("Initializing FASTTAlternating with diagonal transform...")
        
        base_selector = AdaSTTXGBoost(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            early_stopping_rounds=10
        )
        
        # Use tuned values to encourage gating shrinkage
        model = FASTTAlternating(
            base_selector=base_selector,
            transform_type="diagonal",
            transform_kwargs={"lambda1": 0.05, "lambda2": 0.01}, 
            num_iterations=5,
            transform_lr=1e-2, # slightly higher learning rate for surrogate transform update
            transform_steps=100,
            random_state=seed
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

        # Extract and verify gating weights (returns a single 1D array)
        gating_weights = model.get_gating_weights()
        if gating_weights is None:
            logger.bind(category="Test-Gating").error("Gating weights are None, expected a 1D array.")
            assert False, "Gating weights should not be None"

        logger.bind(category="Test-Gating").info("Verifying feature selection via alternating gating weights...")
        
        abs_weights = np.abs(gating_weights)
        
        informative_weights = abs_weights[:n_informative]
        noise_weights = abs_weights[n_informative:]

        mean_informative = float(informative_weights.mean())
        mean_noise = float(noise_weights.mean())
        
        mlflow.log_metric("mean_informative_w", mean_informative)
        mlflow.log_metric("mean_noise_w", mean_noise)

        logger.bind(category="Test-Gating").info(
            f"Mean Informative W: {mean_informative:.4f} | Mean Noise W: {mean_noise:.4f}"
        )

        passed = True
        if mean_informative <= 1.5 * mean_noise:
            logger.bind(category="Test-Gating").warning(
                f"Informative mean is not significantly larger than Noise mean ({mean_informative:.4f} <= 1.5 * {mean_noise:.4f})"
            )
            passed = False
        
        num_zeroed_noise = int(np.sum(noise_weights < 0.1)) 
        logger.bind(category="Test-Gating").info(
            f"Small Noise Features (< 0.1): {num_zeroed_noise} / {n_noise}"
        )

        if passed:
            logger.bind(category="Test-Result").success("SUCCESS: FASTTAlternating DiagonalGating successfully favors informative features over noise.")
        else:
            logger.bind(category="Test-Result").error("FAILURE: FASTTAlternating DiagonalGating failed to adequately separate informative and noise features.")
            
        assert passed, "Failed to appropriately weight informative features greater than noise features."


if __name__ == "__main__":
    logger.bind(category="Test").info("Starting FASTTAlternating DiagonalGating Test Script.")
    test_fastt_alternating()
