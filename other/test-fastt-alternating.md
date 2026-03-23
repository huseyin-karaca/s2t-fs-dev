# Testing FASTT-Alternating DiagonalGating

This document explains the validation process for the `FASTTAlternating` model utilizing the `diagonal` feature transform. In contrast to the end-to-end differentiable FASTTBoosted approach, this algorithm optimizes a non-differentiable selector (XGBoost) iteratively alongside the learnable feature transform using a surrogate gradient step. 

## Overview

The `scripts/test_fastt_alternating.py` script validates the effectiveness of the `DiagonalGating` module on synthetic feature sets. By explicitly generating a dataset with a known distribution of informative vs. noise features, we can evaluate whether the alternating algorithm learns the true underlying distributions.

## Pipeline Architecture

The test script natively adopts MLflow tracking and Loguru logging conventions.

### 1. Synthetic Data Generation
We leverage `s2t_fs.data.synthetic.generate_synthetic_data` to simulate a testing environment:
- **Total Samples**: 4000 (split into 3200 train and 800 test)
- **Features**: 30 (comprising exactly 5 informative directions and 25 pure noise dimensions)

### 2. Model Initialization and Alternating Training
The `FASTTAlternating` algorithm is instantiated specifying `transform_type="diagonal"`. The base selector is defined as an `AdaSTTXGBoost` estimator.
- **Alternations**: `num_iterations=5` toggles between fitting the XGBoost baseline and optimizing the transform gating vectors.
- **Surrogate Optimization**: The `transform_lr` is set to `1e-2` and `transform_steps=100` to assure the surrogate accurately approximates the decision space mapping before pulling gradients.
- **Regularization**: `lambda1=0.05` and `lambda2=0.01` properly zero out the noise.

### 3. Assertion and Verification Logic
Unlike FASTTBoosted, the alternating model produces a **single** consolidated 1D optimal gating vector representing the global transformation prior to the final XGBoost base ensemble.
- The weights associated with the 5 randomly mapped informative features are averaged.
- The weights associated with the 25 noise features are averaged.
- The script asserts that the model assigns a **strict magnitude advantage** (`> 1.5x`) to the informative set compared to the noise set.
- A secondary check asserts that all 25 noise feature dimensions fall below an absolute magnitude of `0.1`.

## Execution

To execute this test locally, utilize the standard module execution approach from the project's root:

```bash
python -m scripts.test_fastt_alternating
```

This will automatically instantiate a new tracking ID under the `FASTT_Alternating_Test` experiment in MLflow.
