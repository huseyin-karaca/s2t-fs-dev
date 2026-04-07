# Transform Experiments Rationale

This page explains the design choices behind the synthetic validation suite for FASTT's four
transform types, and why each experiment uses a slightly different dataset configuration.

---

## Why Synthetic Validation?

Before running the full benchmark on real speech datasets, we validate each transform variant
in isolation using synthetic data. Synthetic experiments let us control the ground-truth
structure of the feature space — the number of informative features, noise features, and
target experts — and verify that the transform correctly identifies and exploits that
structure.

The 16-model matrix (`configs/16_exp/`) covers all combinations of:

- **4 transform types**: `diagonal`, `linear`, `low_rank`, `nonlinear`
- **2 FASTT variants**: `FASTTAlternating`, `FASTTBoosted`
- **2 base selectors**: `AdaSTTXGBoost` / `AdaSTTLightGBM` (for Alternating), `BoostedSDTR` / `AdaSTTMLP` (for Boosted)

Run any individual experiment:

```bash
make run CONFIG=configs/16_exp/exp_alt_XGBoost_nonlinear.json
make run CONFIG=configs/16_exp/exp_boosted_MLP_low_rank.json
```

---

## Rationale for Non-Diagonal Dataset Variants

The diagonal transform (`configs/synt-exp_fastt-alternating-diagonal.json`) uses the default
synthetic dataset configuration. When we move to matrix-based transforms, we deliberately
adjust the dataset to match the structural assumptions each transform is designed to exploit.

### 1 — Linear Transform

**Config:** `synt-exp_fastt-alternating-linear.json`

**Dataset shift:** `n_samples=4500`, `n_informative=6`, `n_noise=30`

A dense linear projection mixes all feature dimensions simultaneously. By providing
a larger number of informative features embedded in a sizeable noise set, we test whether the
learned projection matrix correctly isolates the informative subspace from the noise columns.
If the transform works correctly, the column norms of \( W \) should concentrate on the
informative feature indices.

### 2 — Low-Rank Transform

**Config:** `synt-exp_fastt-boosted-lowrank.json`

**Dataset shift:** `n_samples=5000`, `n_noise=45`, `n_experts=4`

A low-rank factorisation \( u = W_2 \cdot \text{GELU}(W_1 z) \) enforces a strict
informational bottleneck — the bottleneck dimension is smaller than the input dimension.
We increase noise aggressively and add more experts to stress-test whether the bottleneck
forces the model to discard redundant dimensions while preserving discriminative signal.
This is the hardest configuration: the model must compress more aggressively under stronger
noise.

### 3 — Nonlinear Transform

**Config:** `synt-exp_fastt-alternating-nonlinear.json`

**Dataset shift:** `n_samples=3000`, `n_noise=10`

The nonlinear transform builds complex feature interactions via a two-layer GELU MLP. Because
it has the highest capacity of the four transforms, we deliberately keep the dataset simpler
(fewer samples, less noise). This ensures the convergence dynamics of the surrogate training
loop (in the Alternating variant) are observable without being confounded by a hard
optimization landscape. The goal is to verify that the GELU activations add value over the
linear baseline, not to maximally stress-test the optimizer.

### 4 — Diagonal Transform

**Config:** `synt-exp_fastt-alternating-diagonal.json`

The diagonal transform is the simplest: it applies a per-feature scalar gate \( u_i = q_i z_i \).
It is expected to produce sparse solutions (via L1 regularisation) that directly identify
the informative feature indices. This is the reference configuration and uses the default
synthetic dataset parameters.

---

## What "Correct" Looks Like

A transform experiment is considered successful if:

1. **WER improves** over the raw (untransformed) baseline with the same selector.
2. **Feature importances concentrate** on the informative feature indices in the synthetic
   ground truth (verifiable because we know which features are informative).
3. **Training converges** without gradient explosion (monitored via the surrogate loss curve
   in the Alternating variant).

All three criteria are logged to MLflow as part of the experiment artifacts, including the
feature importance plot produced by the `plot_feature_importances` callback.

---

## Compatibility Guarantee

All synthetic validation configs use the same `class_path`-driven registry system as the
real-data experiments. No internal logic is substituted. The transforms, selectors, and
training loops are identical. This guarantees that results on synthetic data transfer
directly to the production benchmarks.
