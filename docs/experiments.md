# Experimental Design

---

## Overview

All experiments share the same execution entry point:

```bash
python -m s2t_fs.experiment --config configs/<config>.json [--mode single|multi|margin]
```

Or, equivalently, via `make`:

```bash
make run-single     CONFIG=configs/<config>.json
make run-comparison CONFIG=configs/<config>.json
make run-margin     CONFIG=configs/<config>.json
make run            CONFIG=configs/<config>.json   # auto-detect mode
```

When `--mode` is omitted, the mode is **auto-detected** from the config structure:

1. Config has `optimization_params` or list-valued `data_params`/`search_params` → `margin`
2. Config has multiple entries in `models` → `multi`
3. Otherwise → `single`

---

## The 3-Tier Hierarchy

### Tier 1 — Single-Model HPT (`make run-single`)

**Script:** `s2t_fs/experiment/train_single_model.py`

Runs `OptunaSearchCV` for exactly one model. This is the workhorse of the system.
Every higher tier ultimately calls this function.

What it does:

1. Loads and prepares data via `load_and_prepare_data(data_params)`.
2. Instantiates the model from its `class_path` via the model registry.
3. Constructs an `OptunaSearchCV` object with the model's hyperparameter space.
4. Calls `search.fit(X_train, Y_train)`.
5. Evaluates `best_estimator_.predict(X_test)` and computes test WER.
6. Logs everything to MLflow via `log_experiment_metadata()` and `log_experiment_results()`.

Use this tier when you want to invest the full trial budget into one model:

```bash
make run-single CONFIG=configs/try_to_improve_our_models_voxpopuli.json
```

### Tier 2 — Multi-Model Comparison (`make run-comparison`)

**Script:** `s2t_fs/experiment/train_multi_model.py`

Runs Tier 1 for every model defined in the config's `models` dict. Data is loaded once and
shared across all models. The MLflow run structure is:

```
Multi_Model_Benchmark (parent run)
  ├── Model_Tuning_Dummy_0
  ├── Model_Tuning_Raw_XGBoost
  ├── Model_Tuning_SelectKBest_SDT
  ├── Model_Tuning_FASTT_Boosted
  └── Model_Tuning_FASTT_Alternating
```

After all models finish, the **WER margin** is computed and logged to the parent run:

```
margin = best_competitor_wer − our_best_wer
```

A positive margin means our model(s) outperform the best competitor.

```bash
make run-comparison CONFIG=configs/model_comparison_voxpopuli.json
```

### Tier 3 — Competitive Margin Maximization (`make run-margin`)

**Script:** `s2t_fs/experiment/train_margin_optimization.py`

**The research question answered by this tier:** _Under which experimental conditions
(data sampling, search budget, etc.) does the gap between our models and competitors grow
largest?_

This tier uses Optuna's `TPESampler` as an outer optimizer over the experimental setup
itself. Parameters in `data_params` or `search_params` that are specified as JSON arrays
(lists) become the search space. Each Optuna trial resolves one concrete set of values,
runs a full Tier 2 comparison, and returns the WER margin as the objective value.

The study maximises margin across all trials.

```
Margin_Optimization_Study (parent run)
  ├── Trial_0_Comparison
  │     ├── Model_Tuning_... (per model)
  │     └── ...
  ├── Trial_1_Comparison
  │     └── ...
  └── ...
```

!!! tip "Why sequential outer trials?"
    The outer Optuna study always uses `n_jobs=1`. TPE (Tree-structured Parzen Estimator)
    is a Bayesian algorithm — each new trial must see the results of all previous trials
    to make an informed suggestion. Parallelising the outer loop would destroy this feedback
    and reduce it to random search.

```bash
make run-margin CONFIG=configs/margin_optimization_voxpopuli.json
```

---

## Config Schema Reference

Every experiment config is a JSON file with five top-level keys.

### `mlflow_params`

```json
"mlflow_params": {
  "tracking_uri": "sqlite:///s2t-fs-experiments.db",
  "experiment_name": "Model Comparison (VoxPopuli)"
}
```

| Key | Description |
|-----|-------------|
| `tracking_uri` | SQLite database path (relative to the project root) |
| `experiment_name` | MLflow experiment name — creates the experiment if it does not exist |

### `data_params`

```json
"data_params": {
  "dataset": "voxpopuli",
  "row_subsample": 1.0,
  "feature_subsample": 1.0,
  "standard_normalize": true,
  "seed": 42,
  "test_size": 0.15
}
```

| Key | Type | Description |
|-----|------|-------------|
| `dataset` | string | One of `voxpopuli`, `librispeech`, `ami`, `common_voice`, or `synthetic` |
| `row_subsample` | float [0,1] | Fraction of rows to randomly retain |
| `feature_subsample` | float [0,1] | Fraction of feature columns to randomly retain |
| `standard_normalize` | bool | Apply `StandardScaler` to features |
| `seed` | int | Random seed for subsampling and train/test split |
| `test_size` | float [0,1] | Fraction of data held out for final evaluation |

In **Tier 3 configs**, any value can be a list to make it a search dimension:

```json
"data_params": {
  "dataset": ["voxpopuli", "librispeech"],
  "row_subsample": [0.5, 0.7, 1.0],
  "standard_normalize": true,
  "seed": 42,
  "test_size": 0.15
}
```

### `search_params`

```json
"search_params": {
  "seed": 0,
  "num_samples": 10,
  "num_folds": 1,
  "test_size": 0.1,
  "refit": true,
  "trial_parallel": false,
  "model_parallel": false
}
```

| Key | Description |
|-----|-------------|
| `num_samples` | Number of Optuna HPT trials per model |
| `num_folds` | Number of CV splits in `ShuffleSplit` |
| `test_size` | CV validation split size (within the training set) |
| `refit` | Whether `OptunaSearchCV` refits on the full training set after search |
| `trial_parallel` | Run Optuna trials in parallel (`n_jobs=-1`). Degrades TPE. |
| `model_parallel` | Run models in parallel processes. CPU-only; GPU models fall back to sequential. |

### `evaluation_params`

```json
"evaluation_params": {
  "target_model_names": ["FASTT_Alternating", "FASTT_Boosted"]
}
```

Identifies which models are "ours" for margin computation. The margin is:
`best_competitor_wer − min(our_model_wers)`.

### `models`

A dictionary mapping human-readable model names to their specs. Each spec has three keys:

```json
"FASTT_Alternating": {
  "class_path": "s2t_fs.models.fastt.fastt_alternating.FASTTAlternating",
  "init_args": { ... },
  "hyperparameters": {
    "num_iterations": [5, 8, 10],
    "transform_lr": [0.001, 0.005]
  }
}
```

| Key | Description |
|-----|-------------|
| `class_path` | Fully qualified Python class path — resolved by the model registry |
| `init_args` | Fixed constructor arguments (not tuned). Supports nested model specs. |
| `hyperparameters` | Dict of `{param_name: [candidate_values]}` for OptunaSearchCV |

---

## Available Configs

### Multi-Model Comparison (Tier 2)

| File | Dataset | Models |
|------|---------|--------|
| `model_comparison_voxpopuli.json` | VoxPopuli | Dummy, Random, XGBoost, SDT, SelectKBest+SDT, SelectKBest+XGBoost, FASTT_Boosted, FASTT_Alternating |
| `model_comparison_ami.json` | AMI | Same model set |
| `model_comparison_librispeech.json` | LibriSpeech | Same model set |
| `model_comparison_common_voice.json` | Common Voice | Same model set |

### Margin Optimization (Tier 3)

| File | Dataset | Description |
|------|---------|-------------|
| `margin_optimization_voxpopuli.json` | VoxPopuli | Outer Optuna search over data/search params |
| `margin_optimization_ami.json` | AMI | Same study on AMI |

### Single-Model Deep Tuning (Tier 1)

| File | Description |
|------|-------------|
| `try_to_improve_our_models_voxpopuli.json` | High-trial-budget tuning of FASTT models on VoxPopuli |
| `xgboost_hpt_config.json` | Deep XGBoost HPT |

### 16-Model Base-Agnosticism Matrix (Tier 2)

All 16 files in `configs/16_exp/` run the full FASTT × transform × base-selector matrix
on synthetic data. See [Transform Experiments Rationale](synthetic_transforms_rationale.md).

---

## MLproject Entry Points

The `MLproject` file at the repo root exposes three named entry points for `mlflow run`:

```bash
# Multi-model comparison
MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db" mlflow run . \
  -e model_comparison --env-manager=local \
  -P config=configs/model_comparison_voxpopuli.json \
  --experiment-name "Model Comparison (VoxPopuli)"

# Single-model HPT
MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db" mlflow run . \
  -e single_model_hpt --env-manager=local \
  -P config=configs/try_to_improve_our_models_voxpopuli.json

# Margin optimization
MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db" mlflow run . \
  -e margin_optimization --env-manager=local \
  -P config=configs/margin_optimization_voxpopuli.json
```

!!! warning "Always export `MLFLOW_TRACKING_URI` when using `mlflow run`"
    When you invoke `mlflow run`, the MLflow CLI creates the parent run **before** your
    Python script starts, using its own default tracking URI. Your script's `tracking_uri`
    from the JSON config is only applied as a fallback when no active run exists.
    To direct `mlflow run` to your SQLite database, always prefix the command with
    `MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db"`.
