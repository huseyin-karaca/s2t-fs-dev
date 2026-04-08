"""
Data loading and stratified train/val/test splitting.

The same (seed, dataset, n_train) tuple always returns the same indices —
this is the contract that makes per-job runs comparable across machines and
across phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from s2t_fs_v2 import config as C


@dataclass
class Split:
    """A train/val/test split with feature matrices and per-expert WER labels.

    X_*  : (N, p)        feature matrices
    Y_*  : (N, L)        per-expert WER matrices (Y[i, j] = WER of expert j on clip i)
    idx_*: (N,)          original row indices in the source parquet (for traceability)
    dataset_name: str
    """

    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    dataset_name: str
    n_train_full: int  # total rows the dataset has after split before subsampling

    @property
    def shapes(self) -> dict:
        return {
            "n_train": int(self.X_train.shape[0]),
            "n_val": int(self.X_val.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "feature_dim": int(self.X_train.shape[1]),
            "n_classes": int(self.Y_train.shape[1]),
        }


def _feature_columns() -> list[str]:
    return [f"f{i}" for i in range(C.NUM_FEATURES)]


def load_parquet(dataset: str) -> pd.DataFrame:
    if dataset not in C.PARQUET_FILES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Known: {sorted(C.PARQUET_FILES)}"
        )
    path = C.DATA_DIR / C.PARQUET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet not found at {path}. Set $S2T_FS_DATA_DIR if files live "
            f"elsewhere (e.g., on Colab Drive)."
        )
    return pd.read_parquet(path)


def build_split(
    dataset: str,
    n_train: int | None = None,
    seed: int = C.SEED,
) -> Split:
    """Build a train/val/test split for a single dataset.

    Pipeline:
    1. Load the parquet
    2. Shuffle once with `seed`
    3. Slice into train/val/test by TRAIN_FRAC/VAL_FRAC/TEST_FRAC
    4. If `n_train` is set and smaller than the train slice, subsample the
       FIRST `n_train` rows of the (shuffled) train slice. This is deterministic
       and avoids re-shuffling, so all methods see the SAME training subset.

    Note: val/test sizes always derive from VAL_FRAC/TEST_FRAC of the FULL
    dataset — i.e., the val/test split does not depend on `n_train`. This is
    intentional: the test set is fixed for all training-curve points.
    """
    df = load_parquet(dataset)
    n_total = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_train_full = int(round(C.TRAIN_FRAC * n_total))
    n_val = int(round(C.VAL_FRAC * n_total))
    # n_test = remainder
    train_idx = perm[:n_train_full]
    val_idx = perm[n_train_full : n_train_full + n_val]
    test_idx = perm[n_train_full + n_val :]

    if n_train is not None:
        if n_train > n_train_full:
            raise ValueError(
                f"Requested n_train={n_train} > available train portion "
                f"{n_train_full} for dataset {dataset!r}"
            )
        train_idx = train_idx[:n_train]

    feat_cols = _feature_columns()
    X = df[feat_cols].to_numpy(dtype=np.float32)
    Y = df[C.WER_COLS].to_numpy(dtype=np.float32)

    return Split(
        X_train=X[train_idx],
        Y_train=Y[train_idx],
        X_val=X[val_idx],
        Y_val=Y[val_idx],
        X_test=X[test_idx],
        Y_test=Y[test_idx],
        idx_train=train_idx,
        idx_val=val_idx,
        idx_test=test_idx,
        dataset_name=dataset,
        n_train_full=n_train_full,
    )
