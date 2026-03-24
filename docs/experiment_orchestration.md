# Experiment Orchestration Architecture

This document describes the 3-tier experiment execution hierarchy, the MLflow
tracking structure, parallelism configuration, and hardware considerations.

## Execution Entry Point

All experiments are launched through a single unified CLI:

```bash
python -m s2t_fs.experiment --config configs/<config>.json [--mode single|multi|margin]
```

Or equivalently via `make`:

```bash
make run-single   CONFIG=configs/some_config.json
make run-comparison CONFIG=configs/some_config.json
make run-margin   CONFIG=configs/some_config.json
make run          CONFIG=configs/some_config.json   # auto-detect mode
```

Mode auto-detection logic (when `--mode` is omitted):

1. Config contains `optimization_params` or list-valued `data_params`/`search_params` &rarr; **margin**
2. Config contains multiple entries in `models` &rarr; **multi**
3. Otherwise &rarr; **single**

---

## 3-Tier Hierarchy

```
Level 3: Margin Optimization Study
  └── Optuna study.optimize() with TPESampler (sequential trials)
      └── Each trial resolves list-valued data/search params
          └── Level 2: Multi-Model Comparison
              └── Sequential loop over models (default)
                  └── Level 1: Single-Model HPT
                      └── OptunaSearchCV (sequential trials by default)
                          └── Model.fit() per trial
```

### Level 1 — Single-Model HPT (`train_single_model.py`)

Runs `OptunaSearchCV` for one model. Can be invoked standalone or called by
Level 2 with pre-loaded data and an explicit MLflow `run_id`.

Key behavior:

- Reads `trial_parallel` from `search_params` (default: `false`)
- `false` &rarr; `OptunaSearchCV(n_jobs=1)` — sequential trials, optimal for TPE
- `true` &rarr; `OptunaSearchCV(n_jobs=-1)` — parallel trials (degrades TPE)

### Level 2 — Multi-Model Comparison (`train_multi_model.py`)

Runs Level 1 for each model in the config. Can be invoked standalone or called
by Level 3.

Key behavior:

- Reads `model_parallel` from `search_params` (default: `false`)
- `false` &rarr; sequential loop over models (safe on all hardware)
- `true` &rarr; process pool with `spawn` context, **but only if all models are
  CPU-only**; a GPU safety guard detects GPU-accelerated models and
  automatically falls back to sequential with a warning

### Level 3 — Margin Optimization (`train_margin_optimization.py`)

Uses Optuna `study.optimize()` to search over data/search hyperparameters
(specified as lists in config). Each trial runs a full Level 2 comparison.
The objective is to maximize the WER margin (find configurations where our
proprietary models outperform competitors the most).

- Always uses `n_jobs=1` for the outer Optuna study (sequential, TPE)
- Delegates all model execution to Level 2

---

## MLflow Tracking Hierarchy

Each tier creates an appropriate parent-child run structure:

```
Level 3 (Margin Optimization):
  Margin_Optimization_Study (parent)
    ├── Trial_0_Comparison (child)
    │   ├── Model_Tuning_Dummy_0 (grandchild)
    │   ├── Model_Tuning_XGBoost (grandchild)
    │   └── Model_Tuning_FASTT_Boosted (grandchild)
    ├── Trial_1_Comparison (child)
    │   └── ...
    └── ...

Level 2 (Multi-Model):
  Multi_Model_Benchmark (parent)
    ├── Model_Tuning_Dummy_0 (child)
    ├── Model_Tuning_XGBoost (child)
    └── Model_Tuning_FASTT_Boosted (child)

Level 1 (Single-Model):
  Model_Tuning_<name> (standalone run)
```

All child/grandchild runs are created via `MlflowClient` (process-safe API),
not the fluent `mlflow.start_run()` API, ensuring correctness in both
sequential and parallel modes.

---

## Parallelism Configuration

The `search_params` section of the config JSON supports two independent flags:

```json
"search_params": {
    "trial_parallel": false,
    "model_parallel": false,
    "num_samples": 10,
    "num_folds": 1,
    "test_size": 0.1,
    "refit": true,
    "seed": 0
}
```

### `trial_parallel` (default: `false`)

Controls `n_jobs` in `OptunaSearchCV`:

| Value   | Behavior |
|---------|----------|
| `false` | `n_jobs=1` — sequential trials. **Recommended for TPE.** Each trial sees the full history of past trials, maximizing sample efficiency. |
| `true`  | `n_jobs=-1` — parallel trials. Faster wall-clock but degrades TPE's Bayesian feedback loop. Only use with non-Bayesian samplers (e.g., `RandomSampler`). |

### `model_parallel` (default: `false`)

Controls model-level execution in Level 2:

| Value   | Behavior |
|---------|----------|
| `false` | Sequential model loop. Each model gets full access to all hardware (GPU + all CPU cores). **Recommended.** |
| `true`  | Attempts to run models in parallel via a process pool (`spawn` context). A GPU safety guard checks the config: if any GPU-accelerated model is present, it **automatically falls back to sequential** with a warning. Only effective for CPU-only model sets. |

### Backward Compatibility

The legacy `"parallel"` key is still supported. If `trial_parallel` /
`model_parallel` are absent but `"parallel"` exists, both flags inherit its
value and a deprecation warning is logged.

### Recommended Configurations

| Scenario | `trial_parallel` | `model_parallel` | Notes |
|----------|:-:|:-:|-------|
| **Default / production** | `false` | `false` | Safe on all hardware. Optimal TPE. Each model gets full resources. |
| CPU-only models, many models | `false` | `true` | Models run in parallel process pool. Modest wall-clock improvement. |
| Non-TPE sampler (e.g., Random) | `true` | `false` | Parallel trials per model. Faster but no Bayesian guidance. |

---

## Hardware & Accelerator Support

### Device Auto-Detection

`s2t_fs.utils.torch_utils.get_torch_device()` returns the best available device:

1. **Apple Metal (MPS)** — Apple Silicon via Metal Performance Shaders
2. **CUDA** — NVIDIA GPU
3. **CPU** — fallback

### GPU vs CPU Model Classification

Models are classified by whether they call `get_torch_device()` and move
tensors to an accelerator:

| Model Class | Hardware | Module |
|-------------|----------|--------|
| `FASTTBoosted` | **GPU** (MPS/CUDA) | `s2t_fs.models.fastt.fastt_boosted` |
| `BoostedSDTR` / `SingleSDTR` | **GPU** (MPS/CUDA) | `s2t_fs.models.sdtr_models` |
| `AdaSTTMLP` | **GPU** (MPS/CUDA) | `s2t_fs.models.adastt_mlp` |
| `FASTTAlternating` | **GPU** (MPS/CUDA) | `s2t_fs.models.fastt.fastt_alternating` |
| `AdaSTTXGBoost` | **CPU only** | `s2t_fs.models.adastt_xgboost` |
| `AdaSTTLightGBM` | **CPU only** | `s2t_fs.models.adastt_lightgbm` |
| `DummyModel` | **CPU only** | `s2t_fs.models.dummy_model` |
| `RandomModel` | **CPU only** | `s2t_fs.models.random_ensemble` |

### Resource Utilization in Sequential Mode

When both `trial_parallel` and `model_parallel` are `false` (default), each
model at each trial gets **full exclusive access** to all hardware:

- **XGBoost**: `nthread = os.cpu_count()` (all CPU cores)
- **LightGBM**: `num_threads = os.cpu_count()` (all CPU cores)
- **GPU models**: sole process using MPS/CUDA (no memory contention)

### GPU Safety Guard

When `model_parallel: true`, the orchestrator checks model `class_path` values
(including nested Pipeline steps) against the known set of GPU classes. If any
GPU model is found:

- A warning is logged explaining why parallel execution is unsafe
- Execution automatically falls back to sequential

This prevents:

- **MPS deadlocks**: Apple Metal cannot be shared across processes
- **CUDA fork corruption**: `fork()` duplicates CUDA context, causing silent
  corruption on Linux (the `spawn` context is used regardless, but GPU memory
  contention remains a risk)
- **OOM errors**: Multiple models competing for limited GPU memory

---

## Speed Expectations by Hardware

### Apple Silicon (MPS)

- GPU models run on Metal Performance Shaders — functional but 3-7x slower
  than equivalent CUDA hardware for typical PyTorch training loops
- `FASTTAlternating` is CPU-bound regardless (no GPU acceleration in its
  current implementation)
- XGBoost / LightGBM use all CPU cores natively

### NVIDIA GPU (CUDA)

- `FASTTBoosted`: expect **3-7x speedup** vs MPS (~15 min &rarr; ~2-5 min)
- `BoostedSDTR` / `AdaSTTMLP`: modest speedup (already fast)
- `FASTTAlternating`: **significant speedup** for transform and surrogate gradient
  steps (GPU-accelerated); base selector (XGBoost) remains CPU-bound
- XGBoost / LightGBM: no change (CPU-only)

### CPU-Only

- All models fall back to CPU automatically
- GPU models train on CPU (slower but functional)
- No special configuration needed

---

## Future Optimizations

### FASTTAlternating Architecture Note

`FASTTAlternating` uses GPU acceleration for its torch components (feature
transform, linear surrogate, gradient updates) while the base selector
(e.g., XGBoost) remains CPU-bound. Each alternating iteration involves:

1. **GPU**: Transform features via `Tθ(z)` on device
2. **CPU**: Transfer to numpy, fit base selector (XGBoost)
3. **GPU**: Fit surrogate and update transform via backpropagation

The CPU-GPU data transfer per iteration is minimal (feature matrices), and the
GPU acceleration of the gradient-heavy surrogate fitting and transform update
steps provides meaningful speedup on CUDA.

### Hybrid Parallel Execution

A more advanced strategy would partition models into GPU and CPU-only groups,
run GPU models sequentially (sole GPU access), then run CPU models in parallel
via process pool. The current architecture supports adding this behind a
`"model_parallel": "auto"` mode if the marginal time savings (~5 min in typical
configs) justify the added complexity.
