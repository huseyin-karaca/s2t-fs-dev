"""
Method registry: 11 methods total (5 reference, 4 classical baselines, 2 FASTT).

Each method exposes a tiny interface:

    sample_hp(trial)  -> dict   # Optuna HP sampling (or {} for reference methods)
    train_and_score(  # full pipeline: fit on train, score on val (HP search) or test (final)
        hp, X_train, Y_train, X_eval, Y_eval, *, seed
    ) -> tuple[float, dict]      # (mean_wer_on_eval, fit_time_dict)

Reference methods ignore X_train/Y_train and HP entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from s2t_fs.models.adastt_mlp import AdaSTTMLP
from s2t_fs.models.adastt_xgboost import AdaSTTXGBoost
from s2t_fs.models.fastt import FASTTAlternating, FASTTBoosted
from s2t_fs_v2 import config as C
from s2t_fs_v2.metrics import selected_wer


# ─────────────────────────────────────────────────────────────────────────────
# Feature normalization wrapper
#
# The raw 1672-d feature vector contains columns with very different scales
# (std≈399, max≈16000) because preprocessing concatenates MFCCs, prosodic
# stats, and SSL embeddings without per-column standardization. Tree-based
# selectors (XGBoost) are scale-invariant, but neural FASTT-SDT immediately
# saturates: the first nn.Linear → sigmoid hits 1.0 exactly → log(1-prob) =
# log(0) = -inf → NaN logits. AdaSTTMLP works because it does its own
# StandardScaler internally (s2t_fs/models/adastt_mlp.py:92-93). FASTTBoosted
# does NOT, so we apply the same standardization at the wrapper level here
# instead of patching s2t_fs/.
# ─────────────────────────────────────────────────────────────────────────────


class _StandardScaledEstimator:
    """Sklearn-style wrapper that StandardScales X before fit/predict.

    Y is left untouched (it's a per-row WER target vector, already in [0, 1]).
    """

    def __init__(self, base):
        self.base = base
        self.scaler_ = None

    def fit(self, X, Y):
        self.scaler_ = StandardScaler().fit(X)
        self.base.fit(self.scaler_.transform(X), Y)
        return self

    def predict(self, X):
        return self.base.predict(self.scaler_.transform(X))

    def predict_proba(self, X):
        return self.base.predict_proba(self.scaler_.transform(X))

# ─────────────────────────────────────────────────────────────────────────────
# HP samplers (manuscript Table II)
# ─────────────────────────────────────────────────────────────────────────────


def _sample_xgb(trial: optuna.Trial) -> dict:
    s = C.SEARCH_SPACES["xgboost"]
    return dict(
        n_estimators=trial.suggest_categorical("xgb_n_estimators", s["n_estimators"]),
        learning_rate=trial.suggest_categorical("xgb_lr", s["learning_rate"]),
        max_depth=trial.suggest_categorical("xgb_max_depth", s["max_depth"]),
        subsample=trial.suggest_categorical("xgb_subsample", s["subsample"]),
        colsample_bytree=trial.suggest_categorical("xgb_colsample", s["colsample_bytree"]),
        min_child_weight=trial.suggest_categorical("xgb_min_child_weight", s["min_child_weight"]),
        reg_lambda=trial.suggest_categorical("xgb_reg_lambda", s["reg_lambda"]),
    )


def _sample_mlp(trial: optuna.Trial) -> dict:
    s = C.SEARCH_SPACES["mlp"]
    return dict(
        lr=trial.suggest_categorical("mlp_lr", s["lr"]),
        weight_decay=trial.suggest_categorical("mlp_wd", s["weight_decay"]),
        batch_size=trial.suggest_categorical("mlp_batch_size", s["batch_size"]),
        dropout_rate=trial.suggest_categorical("mlp_dropout", s["dropout_rate"]),
    )


def _sample_sdt(trial: optuna.Trial) -> dict:
    s = C.SEARCH_SPACES["sdt"]
    return dict(
        num_trees=trial.suggest_categorical("sdt_num_trees", s["num_trees"]),
        depth=trial.suggest_categorical("sdt_depth", s["depth"]),
        lmbda=trial.suggest_categorical("sdt_lmbda", s["lmbda"]),
        # temperature is not exposed by FASTTBoosted's _SDTR ctor; we keep it
        # in the search space for documentation consistency with the manuscript
        # but do not actually pass it through (no-op)
    )


def _sample_selectkbest(trial: optuna.Trial) -> dict:
    return dict(k=trial.suggest_categorical("kbest_k", C.SEARCH_SPACES["selectkbest"]["k"]))


def _sample_fastt_framework(trial: optuna.Trial) -> dict:
    s = C.SEARCH_SPACES["fastt"]
    return dict(
        num_rounds=trial.suggest_categorical("fastt_num_rounds", s["num_rounds"]),
        num_iterations=trial.suggest_categorical("fastt_num_iter", s["num_iterations"]),
        lr=trial.suggest_categorical("fastt_lr", s["lr"]),
        weight_decay=trial.suggest_categorical("fastt_wd", s["weight_decay"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tunable method factories: hp -> fitted estimator on X_train
# ─────────────────────────────────────────────────────────────────────────────


def _fit_raw_xgb(hp, X_train, Y_train, *, seed):
    est = AdaSTTXGBoost(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        reg_lambda=hp["reg_lambda"],
        random_state=seed,
    )
    est.fit(X_train, Y_train)
    return est


def _fit_raw_mlp(hp, X_train, Y_train, *, seed):
    est = AdaSTTMLP(
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"],
        dropout_rate=hp["dropout_rate"],
        random_state=seed,
    )
    est.fit(X_train, Y_train)
    return est


class _SelectKBestWrapper:
    """Composes SelectKBest (fit on X_train with argmin-of-WER targets) with
    a downstream tunable estimator. predict() applies the same feature mask."""

    def __init__(self, k: int, downstream):
        self.k = k
        self.downstream = downstream
        self.selector_ = None

    def fit(self, X, Y):
        # Discrete target = best expert index per training row
        target = np.asarray(Y).argmin(axis=1)
        # Cap k at the number of features available
        k_eff = min(self.k, X.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k_eff)
        X_sel = self.selector_.fit_transform(X, target)
        self.downstream.fit(X_sel, Y)
        return self

    def predict(self, X):
        return self.downstream.predict(self.selector_.transform(X))

    def predict_proba(self, X):
        return self.downstream.predict_proba(self.selector_.transform(X))


def _fit_selectkbest_xgb(hp, X_train, Y_train, *, seed):
    downstream = AdaSTTXGBoost(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        reg_lambda=hp["reg_lambda"],
        random_state=seed,
    )
    est = _SelectKBestWrapper(k=hp["k"], downstream=downstream)
    est.fit(X_train, Y_train)
    return est


def _fit_selectkbest_mlp(hp, X_train, Y_train, *, seed):
    downstream = AdaSTTMLP(
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"],
        dropout_rate=hp["dropout_rate"],
        random_state=seed,
    )
    est = _SelectKBestWrapper(k=hp["k"], downstream=downstream)
    est.fit(X_train, Y_train)
    return est


def _fit_fastt_sdt(hp, X_train, Y_train, *, seed):
    base = FASTTBoosted(
        num_rounds=hp["num_rounds"],
        transform_type="diagonal",
        transform_kwargs={"lambda1": 0.01, "lambda2": 0.01},
        num_trees=hp["num_trees"],
        depth=hp["depth"],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"],
        random_state=seed,
    )
    # FASTT-SDT REQUIRES standardized inputs (see _StandardScaledEstimator
    # docstring). Without it, the SDT's first nn.Linear saturates and the
    # logits collapse to NaN, making predictions degenerate (all-class-0).
    est = _StandardScaledEstimator(base)
    est.fit(X_train, Y_train)
    return est


def _fit_fastt_xgb(hp, X_train, Y_train, *, seed):
    inner = AdaSTTXGBoost(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        reg_lambda=hp["reg_lambda"],
        random_state=seed,
    )
    base = FASTTAlternating(
        base_selector=inner,
        transform_type="diagonal",
        num_iterations=hp["num_iterations"],
        transform_lr=hp["lr"],
        random_state=seed,
    )
    # FASTT-XGB benefits from standardization too: the diagonal-gating
    # transform sits in front of XGBoost as a torch nn.Module trained via
    # surrogate gradients. With raw inputs (max≈16000) the surrogate weights
    # need to be tiny and gradient updates to the gating vector q can be
    # numerically unstable. XGBoost itself is scale-invariant, so this only
    # stabilizes the transform-update half of the alternating loop.
    est = _StandardScaledEstimator(base)
    est.fit(X_train, Y_train)
    return est


# ─────────────────────────────────────────────────────────────────────────────
# Method spec
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MethodSpec:
    name: str
    needs_tuning: bool
    sample_hp: Callable[[optuna.Trial], dict] | None
    fit: Callable[..., Any] | None  # signature: (hp, X_train, Y_train, *, seed) -> estimator

    def predict(self, est, X):
        return est.predict(X)


def _combine_samplers(*fns):
    def sampler(trial: optuna.Trial) -> dict:
        out: dict = {}
        for fn in fns:
            out.update(fn(trial))
        return out

    return sampler


METHODS: dict[str, MethodSpec] = {
    # Reference -------------------------------------------------------------
    "whisper":   MethodSpec("whisper",   False, None, None),
    "parakeet":  MethodSpec("parakeet",  False, None, None),
    "canary":    MethodSpec("canary",    False, None, None),
    "random":    MethodSpec("random",    False, None, None),
    "oracle":    MethodSpec("oracle",    False, None, None),
    # Classical baselines ---------------------------------------------------
    "raw_xgb":           MethodSpec("raw_xgb",           True, _sample_xgb,            _fit_raw_xgb),
    "raw_mlp":           MethodSpec("raw_mlp",           True, _sample_mlp,            _fit_raw_mlp),
    "selectkbest_xgb":   MethodSpec("selectkbest_xgb",   True, _combine_samplers(_sample_xgb, _sample_selectkbest), _fit_selectkbest_xgb),
    "selectkbest_mlp":   MethodSpec("selectkbest_mlp",   True, _combine_samplers(_sample_mlp, _sample_selectkbest), _fit_selectkbest_mlp),
    # FASTT -----------------------------------------------------------------
    "fastt_sdt": MethodSpec(
        "fastt_sdt", True,
        _combine_samplers(_sample_sdt, _sample_fastt_framework, _sample_mlp),
        _fit_fastt_sdt,
    ),
    "fastt_xgb": MethodSpec(
        "fastt_xgb", True,
        _combine_samplers(_sample_xgb, _sample_fastt_framework),
        _fit_fastt_xgb,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Reference selection (used for non-tunable methods)
# ─────────────────────────────────────────────────────────────────────────────


def reference_selection(method: str, n_rows: int, Y: np.ndarray, seed: int) -> np.ndarray:
    if method == "whisper":
        return np.zeros(n_rows, dtype=np.int64)
    if method == "parakeet":
        return np.ones(n_rows, dtype=np.int64)
    if method == "canary":
        return np.full(n_rows, 2, dtype=np.int64)
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, C.N_CLASSES, size=n_rows).astype(np.int64)
    if method == "oracle":
        return np.asarray(Y).argmin(axis=1).astype(np.int64)
    raise ValueError(f"Unknown reference method {method!r}")


# ─────────────────────────────────────────────────────────────────────────────
# High-level convenience: fit-and-score (used by runner)
# ─────────────────────────────────────────────────────────────────────────────


def fit_and_score(
    method: str,
    hp: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Fit a tunable method on (X_train, Y_train) and score on (X_eval, Y_eval).

    Returns: (eval_wer, fit_time_seconds, score_time_seconds).
    Reference methods short-circuit fit_time=0.
    """
    spec = METHODS[method]
    if not spec.needs_tuning:
        t0 = time.perf_counter()
        sel = reference_selection(method, len(X_eval), Y_eval, seed=seed)
        score_time = time.perf_counter() - t0
        return selected_wer(Y_eval, sel), 0.0, score_time

    t0 = time.perf_counter()
    est = spec.fit(hp, X_train, Y_train, seed=seed)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sel = est.predict(X_eval).astype(np.int64)
    score_time = time.perf_counter() - t0

    return selected_wer(Y_eval, sel), fit_time, score_time
