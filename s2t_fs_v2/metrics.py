"""
Selection-based WER computation.

The protocol metric is: given a per-expert WER matrix Y of shape (N, L) and a
selection vector of shape (N,) with values in {0, ..., L-1}, the method's
mean WER is mean_i Y[i, selection[i]].
"""

from __future__ import annotations

import numpy as np


def selected_wer(Y: np.ndarray, selection: np.ndarray) -> float:
    """Mean WER under the given selection."""
    Y = np.asarray(Y)
    selection = np.asarray(selection, dtype=np.int64)
    if Y.ndim != 2 or selection.shape != (Y.shape[0],):
        raise ValueError(
            f"Shape mismatch: Y={Y.shape}, selection={selection.shape}"
        )
    return float(Y[np.arange(len(Y)), selection].mean())


def oracle_wer(Y: np.ndarray) -> float:
    """Per-clip best (lower-bound)."""
    return float(np.asarray(Y).min(axis=1).mean())


def per_expert_wer(Y: np.ndarray) -> dict[int, float]:
    """Mean WER if you always picked expert j."""
    Y = np.asarray(Y)
    return {j: float(Y[:, j].mean()) for j in range(Y.shape[1])}


def best_single_wer(Y: np.ndarray) -> tuple[int, float]:
    """Index and value of the best single expert (mean WER over the eval set)."""
    means = np.asarray(Y).mean(axis=0)
    j = int(means.argmin())
    return j, float(means[j])
