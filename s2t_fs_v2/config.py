"""
Locked experimental constants. Do not change without updating
reports/EXPERIMENTAL_PLAN.md §2 and §7 (Change Log).
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Splits (per-dataset, stratified by source dataset is moot since we eval per
# dataset; we use simple shuffled splits with the locked seed)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
# Resolve at import time relative to this file. Override via $S2T_FS_DATA_DIR
# (e.g. for Colab where parquet files live on Drive).
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
_default_data = REPO_ROOT / "data" / "processed"
DATA_DIR = Path(os.environ.get("S2T_FS_DATA_DIR", str(_default_data)))
RESULTS_DIR = REPO_ROOT / "results"

PARQUET_FILES = {
    "ami": "ami.parquet",
    "librispeech": "librispeech.parquet",
    "common_voice": "common_voice.parquet",
    "voxpopuli": "voxpopuli.parquet",
}

# Manuscript Table I default training sizes (used as default N for Phase 4
# when no Phase 3 sweep was run on that dataset)
MANUSCRIPT_DEFAULT_N_TRAIN = {
    "ami": 3272,
    "librispeech": 8400,
    "common_voice": 6560,
    "voxpopuli": 5120,
}

# WER label columns and feature column prefix
WER_COLS = ["wer_whisper", "wer_parakeet", "wer_canary"]
EXPERT_NAMES = ["whisper", "parakeet", "canary"]
NUM_FEATURES = 1672  # f0..f1671
N_CLASSES = len(WER_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search spaces (manuscript Table II, exact)
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACES = {
    # Selectors --------------------------------------------------------------
    "xgboost": {
        "n_estimators": [100, 500, 1000],
        "learning_rate": [1e-2, 2e-2, 5e-2, 1e-1],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "min_child_weight": [1, 4, 10],
        "reg_lambda": [0.1, 1.0, 10.0],
    },
    "mlp": {
        "lr": [1e-4, 5e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [512, 1024, 2048],
        "dropout_rate": [0.1, 0.2, 0.3],
    },
    "sdt": {
        "num_trees": [10, 50, 100],
        "depth": [3, 4, 5],
        "lmbda": [0.01, 0.1, 1.0],
        "temperature": [0.5, 1.0, 2.0],
    },
    # Feature selection ------------------------------------------------------
    "selectkbest": {
        "k": [50, 100, 200, 500],
    },
    # FASTT framework-level --------------------------------------------------
    "fastt": {
        "num_rounds": [2, 3, 5],
        "lr": [1e-4, 5e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "num_iterations": [3, 5, 10],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Trial budgets per phase
# ─────────────────────────────────────────────────────────────────────────────
TRIALS_PER_PHASE = {
    "phase2": 8,
    "phase3": 8,
    "phase4": 15,
    "smoke": 2,  # for local sanity testing
}
