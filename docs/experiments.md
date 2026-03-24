# Base-Model Agnostic Scalability

The fundamental requirement of the `FASTT` joint feature-model dimensionality reduction sequence is **Base-Model Agnosticism**. 

Our framework proves this through 16 distinct experiments executing across completely variant configurations without relying on any internal logic substitutions in the pipeline:

### Framework Matrix
There are **4 Dimensional Transforms**:
- `diagonal`: Static scalar multiplication resolving per-feature.
- `linear`: Full-dimension linear abstraction handling column spaces.
- `low_rank`: Dimensional bottleneck matrix abstraction forcing informational limits.
- `nonlinear`: Deep interactions utilizing intermediate `GELU` non-linear projections.

There are **2 Wrapper Strategies** resolving internal backpropagation barriers:
- **`FASTTAlternating`**: Bridges gradients sequentially masking **Non-Differentiable Learner Models** using surrogate predictions.
- **`FASTTBoosted`**: Bridges gradients continuously over **Differentiable Back-Propagating Sequential Layers**.

Because `FASTT` operates as an exterior wrapper, the sub-components (`base_selectors`, `base_estimators`) are entirely modularized via PyTorch factories over dynamic JSON `class_path` resolutions.

### The 16 Models Executed
To fully validate structural independence, we tested the 4 transformations across all configurations utilizing the completely alternative learner backends:
1. **Alternating / XGBoost:** Uses the classical `AdaSTTXGBoost` expected-WER mapper.
2. **Alternating / LightGBM:** Uses the new `AdaSTTLightGBM` custom multivariant fobj gradient.
3. **Boosted / SDTR:** Uses the custom Soft Decision Tree Regressor as PyTorch nn.module limits.
4. **Boosted / MLP:** Uses a flat `_MLPDifferentiable` fully connected deep structure.

## Execution Syntax

We expose reproducible entrypoints through the root `MLproject` file.

### Model Comparison & Hyperparameter Tuning
To execute the multi-model comparison across varying architectures natively via MLflow tracking, prefix the tracking URI and supply the experiment name so `mlflow` orchestrates it to the correct backend store:

```bash
# Example: Running the full VoxPopuli model comparison
MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db" mlflow run . \
  -e model_comparison \
  --env-manager=local \
  -P config=configs/model_comparison_voxpopuli.json \
  --experiment-name "Model Comparison (VoxPopuli)"
```

### Single-Model HPT (Deep Tuning a Specific Model)
When you want to invest all trials into improving one model with a richer search space, use the `single_model_hpt` entrypoint. In standalone mode it automatically uses all available CPU cores (`n_jobs=-1`) since Optuna no longer needs to coordinate with other models.

```bash
# Example: Deep-tuning FASTT_Alternating with 30 Bayesian trials
MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db" mlflow run . \
  -e single_model_hpt \
  --env-manager=local \
  -P config=configs/try_to_improve_our_models_voxpopuli.json \
  --experiment-name "Try to Improve Our Models (VoxPopuli)"
```

> By default, both trials and models run **sequentially** (`trial_parallel: false`, `model_parallel: false`).
> This is optimal for TPE-based sampling and safe on all hardware.
> See [Orchestration & Parallelism](experiment_orchestration.md) for details.

### Synthetic Sub-Component Verification
All JSON arrays automatically load specific learner topologies dynamically based on their object pointers!

To verify the matrix individually, you can natively orchestrate standard `mlflow` invocations without altering internal logic using any of the 16 generated template mappings found in `configs/16_exp/`:

```bash
# Example: Running the Boosted Multi-layer perceptron resolving Non-Linear combinations!
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test_transform -P config=configs/16_exp/exp_boosted_MLP_nonlinear.json
```

```bash
# Example: Running the Alternating LightGBM abstraction mapping low-rank properties!
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test_transform -P config=configs/16_exp/exp_alt_LightGBM_low_rank.json
```

## Parallelization Strategy

By default, all execution is **fully sequential**: models run one at a time,
and each model's HPT trials run sequentially (`n_jobs=1`). This is optimal for
TPE-based Bayesian sampling and safe across all hardware (Mac MPS, CUDA, CPU).

Each model gets **exclusive access to all hardware resources** during its
execution:

- XGBoost / LightGBM use all CPU cores (`nthread = os.cpu_count()`)
- GPU models (FASTT_Boosted, BoostedSDTR, AdaSTTMLP) get sole GPU access

Two independent config flags allow opting into parallelism when appropriate:

- `trial_parallel: true` — parallel Optuna trials (degrades TPE, use only with
  non-Bayesian samplers)
- `model_parallel: true` — parallel model execution (CPU-only models only; a
  GPU safety guard prevents unsafe GPU contention)

For full details, see [Orchestration & Parallelism](experiment_orchestration.md).
