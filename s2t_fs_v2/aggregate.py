"""
Aggregate result JSONs from a phase directory into a markdown summary.

Usage:
    python -m s2t_fs_v2.aggregate --phase phase2
    python -m s2t_fs_v2.aggregate --phase phase4 --out results/phase4/SUMMARY.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from s2t_fs_v2 import config as C


def _phase_dir(phase: str) -> Path:
    results_dir = Path(os.environ.get("S2T_FS_RESULTS_DIR", str(C.RESULTS_DIR)))
    return results_dir / phase


def load_phase(phase: str) -> list[dict]:
    rows = []
    for fpath in sorted(glob.glob(str(_phase_dir(phase) / "*.json"))):
        with open(fpath) as f:
            rows.append(json.load(f))
    return rows


def _key(row: dict) -> tuple:
    return (row["dataset"], row["method"], row.get("n_train_effective"))


def render_markdown(phase: str, rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# Results — {phase}")
    lines.append("")
    if not rows:
        lines.append("_No result files found._")
        return "\n".join(lines)

    # ─── Per-dataset WER table ───────────────────────────────────────────
    datasets = sorted({r["dataset"] for r in rows})
    methods = sorted({r["method"] for r in rows})

    lines.append("## Test WER per dataset")
    lines.append("")
    header = "| Method | " + " | ".join(datasets) + " | macro |"
    sep = "|---|" + "|".join(["---:"] * (len(datasets) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    by_key = {(_key(r)): r for r in rows}

    def _wer(method: str, dataset: str) -> float:
        # Use the largest n_train_effective for that (dataset, method) pair
        candidates = [
            r for r in rows if r["dataset"] == dataset and r["method"] == method
        ]
        if not candidates:
            return float("nan")
        best = max(candidates, key=lambda r: r.get("n_train_effective", 0))
        v = best.get("test_wer")
        return float(v) if v is not None else float("nan")

    for m in methods:
        cells = []
        per_dataset_wers = []
        for d in datasets:
            v = _wer(m, d)
            cells.append("---" if np.isnan(v) else f"{v:.4f}")
            if not np.isnan(v):
                per_dataset_wers.append(v)
        macro = float(np.mean(per_dataset_wers)) if per_dataset_wers else float("nan")
        cells.append("---" if np.isnan(macro) else f"{macro:.4f}")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")

    # ─── Relative gap (best classical baseline vs best FASTT) ────────────
    lines.append("")
    lines.append("## Primary criterion: FASTT vs best classical baseline (per dataset)")
    lines.append("")
    lines.append("| Dataset | best classical | classical WER | best FASTT | FASTT WER | Δ rel |")
    lines.append("|---|---|---:|---|---:|---:|")
    classical_methods = [m for m in methods if m in {"raw_xgb", "raw_mlp", "selectkbest_xgb", "selectkbest_mlp"}]
    fastt_methods = [m for m in methods if m in {"fastt_sdt", "fastt_xgb"}]

    for d in datasets:
        c_pairs = [(m, _wer(m, d)) for m in classical_methods]
        c_pairs = [(m, v) for m, v in c_pairs if not np.isnan(v)]
        f_pairs = [(m, _wer(m, d)) for m in fastt_methods]
        f_pairs = [(m, v) for m, v in f_pairs if not np.isnan(v)]
        if not c_pairs or not f_pairs:
            lines.append(f"| {d} | --- | --- | --- | --- | --- |")
            continue
        bc, bcw = min(c_pairs, key=lambda kv: kv[1])
        bf, bfw = min(f_pairs, key=lambda kv: kv[1])
        rel = (bcw - bfw) / bcw * 100 if bcw > 0 else float("nan")
        marker = " ✅" if rel >= 5 else (" ⚠️" if rel >= 0 else " ❌")
        lines.append(f"| {d} | `{bc}` | {bcw:.4f} | `{bf}` | {bfw:.4f} | **{rel:+.2f}%**{marker} |")

    # ─── Wallclock summary ───────────────────────────────────────────────
    lines.append("")
    lines.append("## Wallclock per job")
    lines.append("")
    lines.append("| Dataset | Method | n_train | trials | wallclock | tuning | refit fit | refit score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: (r["dataset"], r["method"], r.get("n_train_effective", 0))):
        lines.append(
            f"| {r['dataset']} | `{r['method']}` | {r.get('n_train_effective', '-')} | "
            f"{r.get('trials_budget', '-')} | "
            f"{r.get('wallclock_seconds', float('nan')):.1f}s | "
            f"{r.get('tuning_seconds', float('nan')):.1f}s | "
            f"{r.get('refit_fit_time_seconds', float('nan')):.2f}s | "
            f"{r.get('refit_score_time_seconds', float('nan')):.4f}s |"
        )

    # ─── Errors ──────────────────────────────────────────────────────────
    errored = [r for r in rows if r.get("error")]
    if errored:
        lines.append("")
        lines.append(f"## ⚠️ Errors ({len(errored)} job(s))")
        lines.append("")
        for r in errored:
            lines.append(f"- `{r['dataset']}/{r['method']}`: **{r['error']['type']}** — {r['error']['message']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    p.add_argument("--out", default=None, help="Output path. Default: results/<phase>/SUMMARY.md")
    args = p.parse_args()

    rows = load_phase(args.phase)
    md = render_markdown(args.phase, rows)
    out = Path(args.out) if args.out else _phase_dir(args.phase) / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
