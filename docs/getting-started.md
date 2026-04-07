# Getting Started

This guide takes you from a clean machine to a running experiment in four steps.

---

## Prerequisites

- **Conda** (Miniconda or Anaconda) — [install guide](https://docs.conda.io/en/latest/miniconda.html)
- **Git**
- **Python 3.10** (managed automatically by Conda)
- An internet connection to download the datasets (~300 MB total)

!!! note "GPU Support"
    The project runs on CPU, Apple Silicon (MPS), and NVIDIA GPU (CUDA 12.1+) without any
    configuration changes. Hardware is detected automatically at runtime.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/huseyin-karaca/s2t-fs-dev.git
cd s2t-fs-dev
```

---

## Step 2 — Create the Conda Environment

```bash
make create_environment
```

This runs `conda env create --name s2t-fs -f environment.yml` and installs all dependencies,
including PyTorch, scikit-learn, XGBoost, LightGBM, Optuna, MLflow, and the `s2t_fs` package
itself (in editable mode via `-e .`).

Once the environment is created, activate it:

```bash
conda activate s2t-fs
```

!!! warning "Always activate the environment before running anything"
    The project uses specific, pinned versions of MLflow, Optuna, and scikit-learn.
    Running scripts outside this environment will cause severe version mismatch errors
    that are difficult to diagnose.

To update the environment after pulling new changes:

```bash
make requirements
```

---

## Step 3 — Download the Datasets

```bash
make download_processed_data
```

This downloads four pre-processed Parquet files into `data/processed/`:

| File | Dataset |
|------|---------|
| `voxpopuli.parquet` | European Parliament speech |
| `librispeech.parquet` | Audiobook read speech |
| `ami.parquet` | Meeting room recordings |
| `common_voice.parquet` | Crowd-sourced speech |

Each file contains acoustic feature columns (`f0`, `f1`, …) and per-model WER target columns
(`wer_<model_name>`).

!!! note
    The download uses `gdown` to fetch files from Google Drive. If you are behind a proxy,
    ensure your environment is configured accordingly.

---

## Step 4 — Run Your First Experiment

Run the full multi-model benchmark on VoxPopuli (the default dataset):

```bash
make run-comparison CONFIG=configs/model_comparison_voxpopuli.json
```

This launches all models defined in the config through the 3-tier experiment pipeline and logs
everything to the local MLflow database at `s2t-fs-experiments.db`.

To view the results:

```bash
mlflow ui --backend-store-uri sqlite:///s2t-fs-experiments.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Available `make` Commands

Run `make help` to print all available commands at any time.

### Environment & Data

| Command | Description |
|---------|-------------|
| `make create_environment` | Create the `s2t-fs` Conda environment from `environment.yml` |
| `make requirements` | Update the environment after dependency changes |
| `make download_processed_data` | Download all four datasets to `data/processed/` |
| `make clean` | Delete compiled `.pyc` files and `__pycache__` directories |

### Linting & Formatting

| Command | Description |
|---------|-------------|
| `make lint` | Check code style with `ruff` |
| `make format` | Auto-fix style issues with `ruff` |

### Running Experiments

All experiment commands accept an optional `CONFIG=` override. The default config is
`configs/model_comparison_voxpopuli.json`.

| Command | Mode | Description |
|---------|------|-------------|
| `make run CONFIG=<path>` | Auto-detect | Detect mode from config structure and run |
| `make run-single CONFIG=<path>` | Single | Hyperparameter tuning for one model |
| `make run-comparison CONFIG=<path>` | Multi | Benchmark all models defined in the config |
| `make run-margin CONFIG=<path>` | Margin | Optuna study to maximize the WER margin |

### Documentation

| Command | Description |
|---------|-------------|
| `make docs-serve` | Serve the documentation locally at [http://127.0.0.1:8000](http://127.0.0.1:8000) |
| `make docs-build` | Build the static documentation site |
| `make docs-deploy` | Deploy to GitHub Pages |

### Remote Execution (Vast.ai)

| Command | Description |
|---------|-------------|
| `make vast-launch` | Rent a GPU instance and start the experiment environment |
| `make vast-dry-run` | Preview the best matching offer without creating an instance |
| `make vast-list` | List your active instances |
| `make vast-stop INSTANCE=<id>` | Destroy a specific instance (omit `INSTANCE` to destroy all) |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

The following variables are used:

| Variable | Required | Description |
|----------|----------|-------------|
| `VAST_API_KEY` | For remote runs | Your Vast.ai API key |
| `GITHUB_ACTOR` | For remote runs | Your GitHub username (lowercase) |
| `GIT_REPO_URL` | For remote runs | Full HTTPS URL of this repository |
| `GPU` | Optional | GPU model for Vast.ai search (default: `RTX_3090`) |
| `MAX_PRICE` | Optional | Max $/hr for Vast.ai (default: `0.30`) |
| `DISK` | Optional | Disk size in GB (default: `30`) |

!!! warning
    Never commit `.env` to version control. It is listed in `.gitignore`.

---

## Project Layout

```
s2t-fs/
├── Makefile                  ← Single source of truth for all commands
├── environment.yml           ← Conda environment specification
├── pyproject.toml            ← Package metadata and ruff config
├── MLproject                 ← MLflow project entry points
├── configs/                  ← Experiment configuration files (JSON)
├── data/
│   └── processed/            ← Downloaded Parquet datasets
├── docs/                     ← This documentation
├── s2t_fs/
│   ├── data/                 ← Data loading and synthetic data generation
│   ├── experiment/           ← 3-tier experiment runners
│   ├── models/               ← All model implementations + registry
│   │   └── fastt/            ← FASTT transform variants
│   ├── utils/                ← Logger, MLflow utils, torch utils
│   └── callbacks/            ← Post-experiment callbacks (e.g., feature importance plots)
├── scripts/
│   ├── vast_launch.sh        ← Vast.ai instance provisioning
│   └── onstart.sh            ← Instance startup script (clone + install + download)
└── s2t-fs-experiments.db     ← Local MLflow SQLite database
```
