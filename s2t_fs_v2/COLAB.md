# Running on Google Colab

This package is designed to run identically on local CPU and on Colab GPU.
A single command runs one `(dataset, method, n_train, seed)` job and writes
a JSON result file. Sync the JSON files back to the local repo when done.

## Recommended workflow (per phase)

**Phase 2 (AMI smoke test, ~30-60 min on A100):**

1. Open a new Colab notebook, A100 runtime.
2. Mount Drive (or upload parquet files manually — see "Data placement" below).
3. Paste the **setup cell** (next section).
4. Paste the **Phase 2 cell**.
5. After completion, download `results/phase2/` as a zip and extract into the
   local repo at `s2t-fs/results/phase2/`.
6. Tell Claude in the chat: "Phase 2 done, results in `results/phase2/`."

**Phase 3** and **Phase 4** follow the same pattern with their own cells below.

---

## Data placement

The runner expects parquet files at one of:

- `<repo>/data/processed/<dataset>.parquet`  (default)
- A directory specified by the env var `S2T_FS_DATA_DIR`

Two ways to get them onto Colab:

### Option A — Google Drive (recommended for repeated runs)

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['S2T_FS_DATA_DIR'] = '/content/drive/MyDrive/s2t-fs-data'
```

Upload `ami.parquet`, `librispeech.parquet`, `common_voice.parquet`,
`voxpopuli.parquet` to that Drive folder once.

### Option B — Direct upload (one-shot)

```python
from google.colab import files
import os, pathlib
pathlib.Path('/content/data').mkdir(exist_ok=True)
os.environ['S2T_FS_DATA_DIR'] = '/content/data'
print('Upload the four .parquet files now (multi-select):')
uploaded = files.upload()
for name in uploaded:
    pathlib.Path('/content/data', name).write_bytes(uploaded[name])
```

---

## Setup cell (paste once at the top of each notebook)

```python
# 1) Clone the repo (or pull latest if you already have it)
import os, subprocess
REPO = '/content/s2t-fs'
if not os.path.exists(REPO):
    !git clone https://github.com/huseyin-karaca/s2t-fs.git /content/s2t-fs
else:
    !cd /content/s2t-fs && git pull
%cd /content/s2t-fs

# 2) Install minimal deps (skipping nemo / transformers — not needed for the harness)
!pip install -q numpy pandas pyarrow scikit-learn xgboost==2.* torch optuna lightgbm loguru

# 3) Make sure both packages are importable
import sys
if '/content/s2t-fs' not in sys.path:
    sys.path.insert(0, '/content/s2t-fs')

# 4) Sanity-check
from s2t_fs_v2 import config as C
print('SEED:', C.SEED, 'DATA_DIR:', C.DATA_DIR)
```

---

## Phase 2 cell — AMI smoke test (5 methods, 8 trials each)

```python
import subprocess, time

JOBS = [
    # (method, n_train) — phase=phase2, dataset=ami
    ('oracle',          None),
    ('canary',          None),
    ('selectkbest_xgb', None),
    ('fastt_xgb',       None),
    ('fastt_sdt',       None),
]

t0 = time.time()
for method, n in JOBS:
    cmd = [
        'python', '-m', 's2t_fs_v2.runner',
        '--phase', 'phase2',
        '--dataset', 'ami',
        '--method', method,
        '--trials', '8',
    ]
    if n is not None:
        cmd += ['--n-train', str(n)]
    print(f'\n>>> {method}')
    subprocess.run(cmd, check=False)
print(f'\n[Phase 2 complete] wallclock: {time.time()-t0:.0f}s')

# Aggregate
!python -m s2t_fs_v2.aggregate --phase phase2

# Zip results for download
!cd results && zip -r phase2_results.zip phase2/
from google.colab import files
files.download('results/phase2_results.zip')
```

After downloading, extract into the local repo:
```bash
unzip phase2_results.zip -d s2t-fs/results/
```

Then tell Claude: "Phase 2 bitti, sonuçlar `results/phase2/` altında."

---

## Phase 3 cell — Mini learning-curve sweep (AMI + VoxPopuli, ~3-5h on A100)

```python
import subprocess

# Datasets and N points to sweep
DATASETS = ['ami', 'voxpopuli']
N_POINTS = [1000, 2000, None]  # None = manuscript default
METHODS = ['selectkbest_xgb', 'raw_xgb', 'fastt_sdt', 'fastt_xgb']

for dataset in DATASETS:
    for n in N_POINTS:
        for method in METHODS:
            cmd = [
                'python', '-m', 's2t_fs_v2.runner',
                '--phase', 'phase3',
                '--dataset', dataset,
                '--method', method,
                '--trials', '8',
            ]
            if n is not None:
                cmd += ['--n-train', str(n)]
            print(f'\n>>> {dataset} {method} n={n}')
            subprocess.run(cmd, check=False)

!python -m s2t_fs_v2.aggregate --phase phase3
!cd results && zip -r phase3_results.zip phase3/
from google.colab import files
files.download('results/phase3_results.zip')
```

---

## Phase 4 cell — Final WER table (4 datasets × ~10 methods, ~3-6h on A100)

Phase 4 should be run AFTER Phase 3 picks N* per dataset. The N values below
are placeholders — replace them with what Claude tells you after Phase 3.

```python
import subprocess

# Replace these with the per-dataset N* picked from the Phase 3 sweep
N_STAR = {
    'ami':          None,  # e.g. 2000 if Phase 3 says so
    'voxpopuli':    None,
    'librispeech':  None,  # default = manuscript Table I
    'common_voice': None,
}

METHODS = [
    'oracle', 'whisper', 'parakeet', 'canary', 'random',
    'raw_xgb', 'raw_mlp',
    'selectkbest_xgb', 'selectkbest_mlp',
    'fastt_sdt', 'fastt_xgb',
]

for dataset, n in N_STAR.items():
    for method in METHODS:
        cmd = [
            'python', '-m', 's2t_fs_v2.runner',
            '--phase', 'phase4',
            '--dataset', dataset,
            '--method', method,
            '--trials', '15',
        ]
        if n is not None:
            cmd += ['--n-train', str(n)]
        print(f'\n>>> {dataset} {method} n={n}')
        subprocess.run(cmd, check=False)

!python -m s2t_fs_v2.aggregate --phase phase4
!cd results && zip -r phase4_results.zip phase4/
from google.colab import files
files.download('results/phase4_results.zip')
```

---

## Splitting work between local CPU and Colab

The runner is fully stateless: each job is independent. To run two datasets
in parallel:

- **Local terminal:** `python -m s2t_fs_v2.runner --phase phase4 --dataset librispeech --method fastt_sdt --trials 15`
- **Colab simultaneously:** Runs the Phase 4 cell above with `N_STAR = {'voxpopuli': ...}` (i.e., one dataset only).

Result files have unique names (`<dataset>__<method>__n<N>__seed<seed>.json`),
so there are no collisions when you merge them.

---

## Resuming after a Colab disconnect

Each completed job writes its JSON immediately. If Colab disconnects mid-run,
just rerun the same cell — the runner skips files that already exist (unless
you pass `--overwrite`).

---

## Quick smoke test (verifies env is set up correctly)

```python
!python -m s2t_fs_v2.runner --phase smoke --dataset ami --method oracle --n-train 200 --trials 0 --overwrite
!python -m s2t_fs_v2.runner --phase smoke --dataset ami --method selectkbest_xgb --n-train 1000 --trials 2 --overwrite
!python -m s2t_fs_v2.runner --phase smoke --dataset ami --method fastt_sdt --n-train 500 --trials 2 --overwrite
!python -m s2t_fs_v2.runner --phase smoke --dataset ami --method fastt_xgb --n-train 500 --trials 2 --overwrite
!python -m s2t_fs_v2.aggregate --phase smoke
```

If all jobs complete and the aggregator prints a table, the environment
is good. If you see import errors for `s2t_fs.models.*`, make sure you
ran the setup cell (`%cd /content/s2t-fs` and the path insert).

---

## Notes on environment fixes baked into the harness

Two issues that bit us during local sanity testing — both are already
handled inside `s2t_fs_v2`, you do not need to do anything extra on Colab,
but it's worth knowing what they are in case something looks weird later.

**1. FASTT-SDT requires standardized features.** The raw 1672-d feature
vector mixes MFCCs, prosodic stats, and SSL embeddings without per-column
normalization (`std≈399`, `max≈16000`). Tree-based selectors (XGBoost) are
scale-invariant, but the SDT's first `nn.Linear → sigmoid` saturates and
collapses to NaN logits → all-class-0 predictions. `AdaSTTMLP` already
self-normalizes internally; `FASTTBoosted` does not. We fix this at the
wrapper level via `_StandardScaledEstimator` in
`s2t_fs_v2/methods.py` — applied to both `fastt_sdt` and `fastt_xgb` so
they sit on the same input footing. No changes to `s2t_fs/`.

**2. macOS torch + xgboost OMP segfault.** On macOS, PyTorch and XGBoost
each ship their own libomp, and loading both into the same process causes
SIGSEGV the moment XGBoost runs after a torch op (which is exactly what
`FASTTAlternating` does: torch transform → xgboost selector). We force
single-thread OMP only on Darwin via `s2t_fs_v2/__init__.py`. **On Colab
Linux this is a no-op** — XGBoost gets full multi-thread parallelism there,
which is what makes the time budget feasible.
