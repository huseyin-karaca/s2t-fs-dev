# Architecture

This document describes the module layout of `s2t_fs`, the design principles behind it,
and the standards every contributor must uphold.

---

## Module Overview

```
s2t_fs/
├── __init__.py
├── data/
│   ├── loader.py          ← load_and_prepare_data()
│   └── synthetic.py       ← generate_synthetic_data() for transform validation
├── experiment/
│   ├── __main__.py        ← Unified CLI entry point (mode auto-detection)
│   ├── train_single_model.py   ← Level 1: Single-model HPT via OptunaSearchCV
│   ├── train_multi_model.py    ← Level 2: Multi-model benchmark orchestrator
│   └── train_margin_optimization.py  ← Level 3: Competitive margin maximization
├── models/
│   ├── registry.py        ← Dynamic class loading from class_path strings
│   ├── dummy_model.py     ← DummyModel (always predicts one fixed target)
│   ├── random_ensemble.py ← RandomModel (random selection baseline)
│   ├── multi_target_selector.py ← MultiTargetSelectKBest (feature pre-filter)
│   ├── adastt_xgboost.py  ← AdaSTTXGBoost (XGBoost with custom WER objective)
│   ├── adastt_lightgbm.py ← AdaSTTLightGBM (LightGBM with custom WER fobj)
│   ├── adastt_mlp.py      ← AdaSTTMLP (PyTorch MLP as sklearn estimator)
│   ├── sdtr_models.py     ← BoostedSDTR / SingleSDTR (Soft Decision Tree)
│   └── fastt/
│       ├── transforms.py        ← build_transform(): diagonal/linear/low_rank/nonlinear
│       ├── fastt_alternating.py ← FASTTAlternating (Algorithm 3)
│       └── fastt_boosted.py     ← FASTTBoosted (Algorithm 4)
├── utils/
│   ├── logger.py          ← Loguru-based structured logger (custom_logger)
│   ├── mlflow_utils.py    ← log_experiment_metadata(), log_experiment_results(), compute_and_log_margin()
│   ├── torch_utils.py     ← get_torch_device(), seed_device(), log_hardware_info()
│   ├── dict_utils.py      ← flatten_dict() for MLflow param logging
│   └── io.py              ← (reserved)
└── callbacks/
    ├── __init__.py        ← run_callbacks() dispatcher
    └── plot_feature_importances.py
```

---

## Design Principles

This codebase is **strictly modular and config-driven**. The following principles govern
every design decision and every line of code.

### Rule 1 — Strict Modularity (DRY)

Do not reimplement what already exists. Every experiment script imports from `s2t_fs.models`,
`s2t_fs.data`, and `s2t_fs.utils`. The modules are designed for reuse, not to be copied.

Correct:

```python
from s2t_fs.models.registry import prepare_model_from_config
from s2t_fs.data.loader import load_and_prepare_data
```

Incorrect:

```python
# ✗ Never instantiate a model inline in an experiment script
model = xgb.XGBClassifier(...)
```

### Rule 2 — The Scikit-Learn Contract

Every model in `s2t_fs.models` implements the scikit-learn `BaseEstimator` and
`ClassifierMixin` interface. This means:

- All hyperparameters are constructor arguments.
- `fit(X, y)` trains the model.
- `predict(X)` returns class indices.
- `score(X, y)` returns negative mean WER (higher is better, sklearn convention).

This contract is what makes `OptunaSearchCV` work seamlessly with every model —
including complex ones like `FASTTAlternating` and `BoostedSDTR`.

### Rule 3 — Optuna for All Hyperparameter Search

We use `OptunaSearchCV` (from `optuna-integration`) exclusively. Do not write manual
tuning loops. The entire HPT loop is:

```python
search = OptunaSearchCV(
    estimator=model_instance,
    param_distributions=model_hpt_space,   # dict of optuna.distributions.*
    cv=cv,
    n_trials=n_trials,
    n_jobs=1,
    refit=True,
)
search.fit(X_train, Y_train)
best_estimator = search.best_estimator_
```

Because `OptunaSearchCV` is itself a scikit-learn estimator, it can be nested inside pipelines.

### Rule 4 — MLflow Is Mandatory

Every experiment run logs:

- All config parameters (flattened via `flatten_dict`)
- The config JSON file as an artifact
- A snapshot of the `s2t_fs/` source directory as an artifact
- The current git commit hash as a tag
- CV metrics (`cv_mean_train_wer`, `cv_mean_test_wer`)
- Final test WER (`custom_test_wer`)

Use `log_experiment_metadata()` and `log_experiment_results()` from
`s2t_fs.utils.mlflow_utils`. Do not write ad-hoc `mlflow.log_*` calls in experiment scripts.

### Rule 5 — Config-Driven Execution

Execution is a pure function: `config → results`. Nothing is hardcoded.
Models, datasets, hyperparameter spaces, search budgets, and tracking destinations are all
specified in JSON config files.

```
Config Input (JSON)
      │
      ▼
s2t_fs.experiment.__main__.py
      │
      ├─ loads model via registry.py (class_path → instance)
      ├─ loads data via loader.py
      ├─ runs OptunaSearchCV
      └─ logs everything to MLflow
```

---

## The Model Registry

`s2t_fs.models.registry` provides dynamic class loading. Any Python class reachable by import
path can be instantiated from a JSON config entry:

```json
{
  "class_path": "s2t_fs.models.fastt.fastt_alternating.FASTTAlternating",
  "init_args": {
    "transform_type": "nonlinear",
    "num_iterations": 5
  },
  "hyperparameters": {
    "transform_lr": [0.001, 0.005, 0.01],
    "num_iterations": [5, 8, 10]
  }
}
```

The registry resolves **nested configs** recursively. This is how `FASTTAlternating`'s
`base_selector` (itself a full model spec) is instantiated from JSON:

```json
"base_selector": {
  "class_path": "s2t_fs.models.adastt_xgboost.AdaSTTXGBoost",
  "init_args": { "n_estimators": 20, "max_depth": 10 }
}
```

`resolve_nested_configs()` walks the `init_args` tree, detects any dict with a `class_path`
key, and replaces it with a live Python object. This enables arbitrarily deep composition —
including `sklearn.pipeline.Pipeline` steps — without any custom deserialisation logic.

---

## Data Flow

```
data/processed/<dataset>.parquet
        │
        ▼
load_and_prepare_data(data_params)
        │   ├─ reads feature columns (f0, f1, ...)
        │   ├─ reads WER target columns (wer_<model>)
        │   ├─ optional row/feature subsampling
        │   ├─ optional StandardScaler normalization
        │   └─ train/test split
        ▼
(X_train, Y_train, X_test, Y_test, dataset_stats)
        │
        ▼
OptunaSearchCV.fit(X_train, Y_train)
        │
        ▼
best_estimator.predict(X_test)  →  WER evaluation
```

`Y_train` and `Y_test` are 2D arrays of shape `(n_samples, n_models)`, where each column is
the WER of one candidate S2T model on that utterance. The prediction task is to select the
column index (model) with the lowest WER.

---

## Logging Infrastructure

All logging uses the structured Loguru logger exported as `custom_logger` from
`s2t_fs.utils.logger`. Log records carry a `category` field for filtering:

```python
from s2t_fs.utils.logger import custom_logger as logger

logger.bind(category="Data").info("Loading dataset...")
logger.bind(category="HPT-Detail").success(f"Test WER: {wer:.4f}")
```

Log files are written to `logs/s2t_fs_<date>.log`. Do not use `print()` or the standard
`logging` module in any `s2t_fs` module.

---

## Code Style

The project uses `ruff` for linting and formatting, configured in `pyproject.toml`.

```toml
[tool.ruff]
line-length = 99
src = ["s2t_fs"]

[tool.ruff.lint]
extend-select = ["I"]  # enforce import sorting
```

Run `make lint` to check. Run `make format` to auto-fix. All pull requests must pass lint.

Key standards:

- **PEP 8** compliance enforced by ruff.
- **Import order**: standard library → third-party → `s2t_fs` (enforced by isort via ruff).
- **Type hints** on all public function signatures.
- **Docstrings** in NumPy style on all public functions and classes.
- **No `print()`**: use `logger.bind(category=...).info(...)`.
