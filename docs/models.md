# Model Catalog

All models in `s2t_fs.models` follow the scikit-learn `BaseEstimator` / `ClassifierMixin`
interface. Every model can be used with `OptunaSearchCV` and instantiated from a JSON config
via the model registry.

---

## Model Summary

| Class | Module | Hardware | Task |
|-------|--------|----------|------|
| `DummyModel` | `dummy_model` | CPU | Baseline — always predicts one fixed target index |
| `RandomModel` | `random_ensemble` | CPU | Baseline — random model selection |
| `MultiTargetSelectKBest` | `multi_target_selector` | CPU | Feature pre-filter (sklearn Pipeline step) |
| `AdaSTTXGBoost` | `adastt_xgboost` | CPU | XGBoost with custom expected-WER objective |
| `AdaSTTLightGBM` | `adastt_lightgbm` | CPU | LightGBM with custom multivariate WER fobj |
| `AdaSTTMLP` | `adastt_mlp` | GPU (MPS/CUDA) | PyTorch MLP with expected-WER loss |
| `SingleSDTR` | `sdtr_models` | GPU (MPS/CUDA) | Single Soft Decision Tree Regressor |
| `BoostedSDTR` | `sdtr_models` | GPU (MPS/CUDA) | Gradient-boosted stack of SDTRs |
| `FASTTAlternating` | `fastt.fastt_alternating` | GPU (MPS/CUDA) | Joint transform + non-differentiable selector |
| `FASTTBoosted` | `fastt.fastt_boosted` | GPU (MPS/CUDA) | Joint transform + differentiable boosted layers |

---

## Baselines

### `DummyModel`

Always predicts a single fixed target index regardless of input. Use to establish the lower
bound of performance for one specific S2T model being universally preferred.

```json
{
  "class_path": "s2t_fs.models.dummy_model.DummyModel",
  "init_args": { "target_index": 0 },
  "hyperparameters": {}
}
```

### `RandomModel`

Selects a model index uniformly at random for each prediction. Provides a stochastic lower
bound.

```json
{
  "class_path": "s2t_fs.models.random_ensemble.RandomModel",
  "init_args": { "seed": 42 },
  "hyperparameters": {}
}
```

---

## Tree-Based Selectors

### `AdaSTTXGBoost`

An XGBoost multi-output regressor wrapped to predict expected WER across all candidate
models. Uses a custom gradient/hessian objective that directly minimises expected WER instead
of squared error. The `predict()` method returns the argmin column (the model predicted to
achieve the lowest WER).

Key hyperparameters:

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Step size shrinkage |
| `max_depth` | Maximum tree depth |
| `subsample` | Row subsampling ratio |
| `colsample_bytree` | Feature subsampling ratio per tree |
| `min_child_weight` | Minimum sum of instance weight in a leaf |
| `reg_lambda` | L2 regularisation term |

```json
{
  "class_path": "s2t_fs.models.adastt_xgboost.AdaSTTXGBoost",
  "init_args": { "random_state": 42 },
  "hyperparameters": {
    "n_estimators": [10, 20, 50, 100],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7, 10]
  }
}
```

### `AdaSTTLightGBM`

Equivalent to `AdaSTTXGBoost` but built on LightGBM with a custom multi-target WER `fobj`.
Faster on large datasets and more memory-efficient.

Key hyperparameters: `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`,
`min_child_samples`, `reg_alpha`, `reg_lambda`.

### `MultiTargetSelectKBest`

A scikit-learn-compatible feature selector that applies `SelectKBest` independently for each
WER target column and takes the union of selected features. Used as a Pipeline pre-processing
step to reduce dimensionality before a downstream selector.

```json
{
  "class_path": "sklearn.pipeline.Pipeline",
  "init_args": {
    "steps": [
      ["selector", {
        "class_path": "s2t_fs.models.multi_target_selector.MultiTargetSelectKBest",
        "init_args": {}
      }],
      ["model", {
        "class_path": "s2t_fs.models.adastt_xgboost.AdaSTTXGBoost",
        "init_args": { "random_state": 42 }
      }]
    ]
  },
  "hyperparameters": {
    "selector__k": [5, 10, 20, 30],
    "model__n_estimators": [20, 50, 100],
    "model__max_depth": [3, 5, 7]
  }
}
```

---

## Neural Selectors

### `AdaSTTMLP`

A fully connected feed-forward neural network trained end-to-end with expected-WER loss.
Runs on GPU (MPS or CUDA) via PyTorch.

Key hyperparameters: `hidden_dims` (list of layer widths), `lr`, `epochs`, `batch_size`,
`dropout`, `weight_decay`.

### `BoostedSDTR`

A gradient-boosted sequence of Soft Decision Tree Regressors (SDTR). Each tree is a
differentiable PyTorch `nn.Module` trained with backpropagation. Stacking multiple layers
(`num_boosting_layers`) implements a form of functional gradient boosting.

Key hyperparameters:

| Parameter | Description |
|-----------|-------------|
| `num_boosting_layers` | Number of SDTR layers in the boosted stack |
| `depth` | Depth of each soft decision tree |
| `num_trees` | Number of trees per layer |
| `lr` | Learning rate |
| `lmbda` | Regularisation strength |
| `epochs` | Training epochs per layer |
| `batch_size` | Mini-batch size |

---

## FASTT Models

FASTT (Feature-Adaptive Selector with Trainable Transforms) is the core contribution of this
work. Both variants wrap a **parametric feature transform** \( T_\theta \) around a base
selector, jointly optimising the transform and the selector for minimal expected WER.

### Feature Transforms

All FASTT models share the same four transform families, implemented in
`s2t_fs.models.fastt.transforms` and constructed via `build_transform()`:

| `transform_type` | Description | Parameters |
|-----------------|-------------|------------|
| `diagonal` | Scalar per-feature gating (element-wise multiplication). Differentiable. Sparse via L1/L2 regularisation. | `lambda1` (L1), `lambda2` (L2) |
| `linear` | Full-rank linear projection \( u = Wz \). Learns a global rotation of the feature space. | `weight_decay` |
| `low_rank` | Bottleneck factorisation \( u = W_2 \cdot \text{GELU}(W_1 z) \), \( \dim(W_1) < \dim(z) \). Enforces informational compression. | `bottleneck_dim`, `weight_decay` |
| `nonlinear` | Two-layer MLP with GELU activation. Captures non-linear feature interactions. | `hidden_dim`, `weight_decay` |

Feature importance after fitting is derived from the transform weights:

- **Diagonal**: absolute value of the gate vector \( |q_i| \)
- **Linear / Low-rank / Nonlinear**: \( \ell_2 \) column norm of the first weight matrix \( W_1 \)

### `FASTTAlternating`

Implements **Algorithm 3** — alternating optimisation for a non-differentiable selector
(e.g., XGBoost).

**Why alternating?** XGBoost is not differentiable: gradients cannot be backpropagated
through it to update \( T_\theta \). `FASTTAlternating` solves this by fitting a small
differentiable **linear surrogate** \( S_\phi(u) \approx \hat{q}(u) \) that approximates the
selector's logit mapping. Gradients flow through the surrogate, not the selector.

**One alternating iteration \( t \):**

1. **Transform**: compute \( u = T_\theta(z) \) (forward pass, no grad)
2. **Selector update**: train a fresh clone of `base_selector` on \( u \) (XGBoost fit)
3. **Surrogate fit**: fit a linear model \( S_\phi \) to match the selector's logits on \( u \)
4. **Transform update**: minimise \( \mathbb{E}_{\text{WER}}[\text{softmax}(S_\phi(T_\theta(z)))] \) w.r.t. \( \theta \) via gradient descent

After all iterations, a final selector is trained on the converged transformed features.

Key hyperparameters:

| Parameter | Description |
|-----------|-------------|
| `base_selector` | The non-differentiable selector (config-driven, e.g., `AdaSTTXGBoost`) |
| `transform_type` | One of `diagonal`, `linear`, `low_rank`, `nonlinear` |
| `num_iterations` | Number of alternating iterations \( T \) |
| `transform_lr` | Learning rate for transform update gradient steps |
| `transform_steps` | Number of gradient steps per transform update |
| `transform_weight_decay` | Weight decay for linear/low_rank/nonlinear transforms |
| `transform_lambda1/2` | L1/L2 regularisation for diagonal gating |

```json
{
  "class_path": "s2t_fs.models.fastt.fastt_alternating.FASTTAlternating",
  "init_args": {
    "transform_type": "nonlinear",
    "num_iterations": 5,
    "transform_steps": 200,
    "random_state": 42,
    "base_selector": {
      "class_path": "s2t_fs.models.adastt_xgboost.AdaSTTXGBoost",
      "init_args": { "n_estimators": 20, "max_depth": 10 }
    }
  },
  "hyperparameters": {
    "num_iterations": [5, 8, 10],
    "transform_lr": [0.001, 0.005, 0.01],
    "transform_steps": [100, 150, 200]
  }
}
```

### `FASTTBoosted`

Implements **Algorithm 4** — joint optimisation for a differentiable selector
(`BoostedSDTR` or `AdaSTTMLP`). Because the selector is fully differentiable, gradients can
be backpropagated directly from the WER loss through the selector and into the transform.

`FASTTBoosted` trains the transform and the boosted SDTR stack end-to-end in a single
gradient-based loop. This is simpler than the alternating approach but requires that the
selector be implemented as a PyTorch `nn.Module`.

Key hyperparameters:

| Parameter | Description |
|-----------|-------------|
| `transform_type` | One of `diagonal`, `linear`, `low_rank`, `nonlinear` |
| `num_rounds` | Number of boosting rounds |
| `num_trees` | Trees per round |
| `depth` | Tree depth |
| `lr` | Joint learning rate |
| `epochs` | Training epochs |
| `patience` | Early stopping patience |

---

## The 16-Model Validation Matrix

To validate that FASTT is truly base-model agnostic, we test all 4 transforms against all
combinations of the 2 variants × 2 base selectors:

| Variant | Base Selector | Configs |
|---------|--------------|---------|
| `FASTTAlternating` | `AdaSTTXGBoost` | `configs/16_exp/exp_alt_XGBoost_*.json` |
| `FASTTAlternating` | `AdaSTTLightGBM` | `configs/16_exp/exp_alt_LightGBM_*.json` |
| `FASTTBoosted` | `BoostedSDTR` | `configs/16_exp/exp_boosted_SDTR_*.json` |
| `FASTTBoosted` | `AdaSTTMLP` | `configs/16_exp/exp_boosted_MLP_*.json` |

Each config is named `exp_<variant>_<base>_<transform>.json`. Run any one of them:

```bash
make run CONFIG=configs/16_exp/exp_alt_XGBoost_nonlinear.json
```
