"""
Optuna-based hyperparameter search.

The driver is a thin wrapper around optuna.create_study. We use the TPE sampler
(default) with no pruner — fair comparison across methods within an equal
trial budget.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna

from s2t_fs_v2.methods import METHODS, fit_and_score


def tune(
    method: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Run an Optuna study for a single (tunable) method.

    Returns a dict with: best_hp, best_val_wer, n_trials, tuning_seconds, all_trials.
    Reference methods are not allowed here (raises ValueError).
    """
    spec = METHODS[method]
    if not spec.needs_tuning:
        raise ValueError(f"Method {method!r} does not need tuning")

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        hp = spec.sample_hp(trial)
        try:
            wer, fit_t, score_t = fit_and_score(
                method, hp, X_train, Y_train, X_val, Y_val, seed=seed
            )
        except Exception as exc:  # noqa: BLE001
            # Mark as bad — Optuna will avoid this region next time.
            trial.set_user_attr("error", repr(exc))
            return float("inf")
        trial.set_user_attr("fit_time", fit_t)
        trial.set_user_attr("score_time", score_t)
        return wer

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    tuning_seconds = time.perf_counter() - t0

    all_trials = [
        dict(
            number=t.number,
            params=t.params,
            value=t.value,
            user_attrs=t.user_attrs,
            state=str(t.state),
        )
        for t in study.trials
    ]

    return dict(
        best_hp=dict(study.best_params) if study.best_trial is not None else {},
        best_val_wer=float(study.best_value) if study.best_trial is not None else float("inf"),
        n_trials=n_trials,
        tuning_seconds=tuning_seconds,
        all_trials=all_trials,
    )
