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
