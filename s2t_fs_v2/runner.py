"""
Stateless single-job runner.

Each invocation runs ONE (dataset, method, n_train, seed) tuple end-to-end:
  1. Build the locked train/val/test split for the dataset
  2. (Tunable methods only) Run Optuna with `trials` budget on (train, val)
  3. Refit on train ∪ val with best HP
  4. Score on the held-out test set
  5. Write a JSON result file

Run identically on local CPU or Colab GPU. The result file is the only
external state — copy/sync it back when done.

Usage:
    python -m s2t_fs_v2.runner --dataset ami --method fastt_sdt --phase phase2

Env vars:
    S2T_FS_DATA_DIR  — override parquet directory (default: <repo>/data/processed)
    S2T_FS_RESULTS_DIR — override results dir (default: <repo>/results)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

from s2t_fs_v2 import config as C
from s2t_fs_v2.data import build_split
from s2t_fs_v2.methods import METHODS, fit_and_score
from s2t_fs_v2.metrics import (
    best_single_wer,
    oracle_wer,
    per_expert_wer,
)
from s2t_fs_v2.search import tune


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=C.REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _device_summary() -> dict:
    try:
        import torch

        cuda = torch.cuda.is_available()
        return {
            "torch": torch.__version__,
            "cuda": cuda,
            "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"torch_error": repr(exc)}


def _env_summary() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "device": _device_summary(),
    }


def _output_path(phase: str, dataset: str, method: str, n_train: int, seed: int) -> Path:
    results_dir = Path(os.environ.get("S2T_FS_RESULTS_DIR", str(C.RESULTS_DIR)))
    out_dir = results_dir / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}__{method}__n{n_train}__seed{seed}.json"
    return out_dir / fname


def run_job(
    *,
    phase: str,
    dataset: str,
    method: str,
    n_train: int | None,
    trials: int | None,
    seed: int = C.SEED,
    overwrite: bool = False,
) -> dict:
    """Run one job and return the result dict (also written to disk)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Known: {sorted(METHODS)}")

    spec = METHODS[method]
    if trials is None:
        trials = C.TRIALS_PER_PHASE.get(phase, C.TRIALS_PER_PHASE["smoke"])

    split = build_split(dataset, n_train=n_train, seed=seed)

    # Resolve effective n_train AFTER the split is built (so Phase 4 default
    # paths get a real number, not None, in the filename and JSON)
    eff_n_train = int(split.X_train.shape[0])
    out_path = _output_path(phase, dataset, method, eff_n_train, seed)
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists. Pass --overwrite to redo.")
        with open(out_path) as f:
            return json.load(f)

    started_at = datetime.now(timezone.utc).isoformat()
    job_t0 = time.perf_counter()

    # ── Reference: per-expert / oracle / random / single -------------------
    per_expert_test = per_expert_wer(split.Y_test)
    oracle_test = oracle_wer(split.Y_test)
    best_idx, best_test = best_single_wer(split.Y_test)
    best_name = C.EXPERT_NAMES[best_idx]

    record: dict = dict(
        phase=phase,
        dataset=dataset,
        method=method,
        seed=seed,
        n_train_requested=n_train,
        n_train_effective=eff_n_train,
        n_val=int(split.X_val.shape[0]),
        n_test=int(split.X_test.shape[0]),
        feature_dim=int(split.X_train.shape[1]),
        n_classes=int(split.Y_train.shape[1]),
        trials_budget=trials,
        started_at=started_at,
        env=_env_summary(),
        per_expert_wer_test={
            C.EXPERT_NAMES[i]: float(per_expert_test[i]) for i in range(C.N_CLASSES)
        },
        oracle_wer_test=float(oracle_test),
        best_single_test={"name": best_name, "index": int(best_idx), "wer": float(best_test)},
    )

    error: dict | None = None

    try:
        if not spec.needs_tuning:
            # Reference method: no tuning, no fit
            test_wer, _, score_time = fit_and_score(
                method, {}, split.X_train, split.Y_train, split.X_test, split.Y_test, seed=seed
            )
            val_wer, _, _ = fit_and_score(
                method, {}, split.X_train, split.Y_train, split.X_val, split.Y_val, seed=seed
            )
            record.update(
                best_hp={},
                best_val_wer=float(val_wer),
                test_wer=float(test_wer),
                refit_fit_time_seconds=0.0,
                refit_score_time_seconds=float(score_time),
                tuning_seconds=0.0,
                all_trials=[],
            )
        else:
            # ── Tunable: Optuna tune on (train, val) -----------------------
            study = tune(
                method,
                split.X_train,
                split.Y_train,
                split.X_val,
                split.Y_val,
                n_trials=trials,
                seed=seed,
            )

            # Map flat optuna param names back into the canonical hp dict
            best_hp = study["best_hp"]
            canonical_hp = _canonical_hp(best_hp)

            # Refit on train ∪ val with best HP, score on test
            X_combined = np.concatenate([split.X_train, split.X_val], axis=0)
            Y_combined = np.concatenate([split.Y_train, split.Y_val], axis=0)
            test_wer, refit_fit_t, refit_score_t = fit_and_score(
                method,
                canonical_hp,
                X_combined,
                Y_combined,
                split.X_test,
                split.Y_test,
                seed=seed,
            )

            record.update(
                best_hp_optuna=best_hp,
                best_hp=canonical_hp,
                best_val_wer=float(study["best_val_wer"]),
                test_wer=float(test_wer),
                refit_fit_time_seconds=float(refit_fit_t),
                refit_score_time_seconds=float(refit_score_t),
                tuning_seconds=float(study["tuning_seconds"]),
                all_trials=study["all_trials"],
            )
    except Exception as exc:  # noqa: BLE001
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        record.update(test_wer=float("nan"), error=error)

    record["wallclock_seconds"] = time.perf_counter() - job_t0
    record["finished_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"[done] {out_path}")
    if error is None:
        rel_vs_best_single = (
            (record["best_single_test"]["wer"] - record["test_wer"]) / record["best_single_test"]["wer"] * 100
            if record["best_single_test"]["wer"] > 0
            else float("nan")
        )
        print(
            f"       test_wer={record['test_wer']:.4f}  "
            f"vs best_single ({best_name}={best_test:.4f})  "
            f"rel_gain={rel_vs_best_single:+.2f}%  "
            f"oracle={oracle_test:.4f}  "
            f"wallclock={record['wallclock_seconds']:.1f}s"
        )
    else:
        print(f"       ERROR: {error['type']}: {error['message']}")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Optuna param-name canonicalization
# Optuna trial.params keys are flat (e.g. 'xgb_n_estimators'); the fit
# functions expect canonical names (e.g. 'n_estimators'). This map is the
# inverse of the prefix scheme used in methods._sample_*.
# ─────────────────────────────────────────────────────────────────────────────


_OPTUNA_KEY_MAP = {
    "xgb_n_estimators": "n_estimators",
    "xgb_lr": "learning_rate",
    "xgb_max_depth": "max_depth",
    "xgb_subsample": "subsample",
    "xgb_colsample": "colsample_bytree",
    "xgb_min_child_weight": "min_child_weight",
    "xgb_reg_lambda": "reg_lambda",
    "mlp_lr": "lr",
    "mlp_wd": "weight_decay",
    "mlp_batch_size": "batch_size",
    "mlp_dropout": "dropout_rate",
    "sdt_num_trees": "num_trees",
    "sdt_depth": "depth",
    "sdt_lmbda": "lmbda",
    "kbest_k": "k",
    "fastt_num_rounds": "num_rounds",
    "fastt_num_iter": "num_iterations",
    "fastt_lr": "lr",
    "fastt_wd": "weight_decay",
}


def _canonical_hp(optuna_params: dict) -> dict:
    """Translate Optuna's prefixed param names back to canonical names.

    For methods that mix two HP groups (e.g. fastt_sdt uses both sdt and mlp
    spaces, both of which define 'lr' / 'wd'), the LATER prefix wins. Order
    of precedence: fastt > sdt > mlp > xgb. We pick this so that fastt_*
    framework-level lr/wd take precedence over the inner SDT/MLP defaults
    (matching the manuscript's framework-level FASTT hyperparameters).
    """
    out: dict = {}
    # Lowest precedence first
    precedence = ["xgb_", "mlp_", "sdt_", "fastt_"]
    for prefix in precedence:
        for k, v in optuna_params.items():
            if k.startswith(prefix):
                canonical = _OPTUNA_KEY_MAP[k]
                out[canonical] = v
    # SelectKBest 'k' is its own thing, no prefix collision
    if "kbest_k" in optuna_params:
        out["k"] = optuna_params["kbest_k"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one (dataset, method, n_train, seed) job.")
    p.add_argument("--phase", required=True, choices=sorted(C.TRIALS_PER_PHASE) + ["phase2", "phase3", "phase4"])
    p.add_argument("--dataset", required=True, choices=sorted(C.PARQUET_FILES))
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="If omitted, use the manuscript Table I default for the dataset.",
    )
    p.add_argument("--trials", type=int, default=None, help="Override default trials for the phase.")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n_train = args.n_train
    if n_train is None and args.method != "oracle":
        # Default to manuscript value
        n_train = C.MANUSCRIPT_DEFAULT_N_TRAIN.get(args.dataset)
    run_job(
        phase=args.phase,
        dataset=args.dataset,
        method=args.method,
        n_train=n_train,
        trials=args.trials,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
