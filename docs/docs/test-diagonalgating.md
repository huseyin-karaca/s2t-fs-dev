# Testing FASTT-Boosted DiagonalGating

This document explains the validation process for the `FASTTBoosted` model utilizing the `diagonal` feature transform. This gating mechanism aims to zero out or apply a minimal weight penalty to uninformative (noise) features while assigning significantly larger weights to informative features in high-dimensional datasets.

## Overview

The `scripts/test_fastt_diagonalgating.py` script validates the effectiveness of the `DiagonalGating` module on synthetic feature sets. By explicitly generating a dataset with a known distribution of informative vs. noise features, we can evaluate whether the deep model learns the true underlying distributions.

## Pipeline Architecture

The test script follows the project's design language, natively adopting MLflow tracking and Loguru logging conventions.

### 1. Synthetic Data Generation
We leverage `s2t_fs.data.synthetic.generate_synthetic_data` to simulate a testing environment that is identical in format to `load_and_prepare_data`. For the validation procedure, we generate:
- **Total Samples**: 4000 (split into 3200 train and 800 test)
- **Features**: 30 (comprising exactly 5 informative directions and 25 pure noise dimensions)
- **Experts**: 3

### 2. Model Initialization and Training
The `FASTTBoosted` algorithm is instantiated specifying `transform_type="diagonal"`. To ensure the L1/L2 regularization correctly achieves sparsity iteratively across the ensemble's boosts:
- `learning_rate` is moderately set to `0.01` with weight decay.
- `lambda1=0.05` and `lambda2=0.01` constraints are enforced for the diagonal scaling weights to strictly zero-out useless noise features over 100 epochs.

### 3. Assertion and Verification Logic
After training, the validation logic assesses the gating coefficients across every tree-boosting round:
- The weights associated with the 5 randomly mapped informative features are averaged.
- The weights associated with the 25 noise features are averaged.
- The script asserts that the model assigns **significantly greater average magnitude** (by a strict 1.5x margin) to the informative set compared to the noise set.
- A secondary check asserts that the vast majority of the noise feature dimension weights decay close to zero (`< 0.1`).

## Execution

To execute this test locally, utilize the standard module execution approach from the project's root:

```bash
python -m scripts.test_fastt_diagonalgating
```

This will automatically instantiate a new instance under the `FASTT_DiagonalGating_Test` experiment in MLflow and log the weight-convergence diagnostics incrementally across each round.
