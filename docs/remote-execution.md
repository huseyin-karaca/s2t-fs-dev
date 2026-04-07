# Remote Execution (Vast.ai)

This guide walks you through renting a GPU on [Vast.ai](https://vast.ai), deploying the
S2T-FS environment, and running experiments remotely — all from `make` commands.

---

## Overview

The remote execution workflow is fully automated:

1. `make vast-launch` rents a GPU instance and provisions it with the pre-built Docker image.
2. The instance runs `onstart.sh` on startup: it clones the repository, installs the package,
   and downloads all four datasets.
3. You SSH in and run experiments with `make run CONFIG=...` exactly as you would locally.
4. Download the MLflow database and inspect results locally.
5. `make vast-stop` destroys the instance when you are done.

---

## Prerequisites

### 1 — Install the Vast.ai CLI

```bash
pip install vastai
```

### 2 — Create a Vast.ai Account

Sign up at [cloud.vast.ai](https://cloud.vast.ai). Navigate to **Account → API Key** and
copy your key.

### 3 — Configure `.env`

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required for remote execution
VAST_API_KEY=your_vast_api_key_here
GITHUB_ACTOR=your_github_username   # lowercase
GIT_REPO_URL=https://github.com/your_github_username/s2t-fs.git

# Optional overrides (these are the defaults)
GPU=RTX_3090
MAX_PRICE=0.30
DISK=30
INET_DOWN=500
INET_UP=100
GIT_BRANCH=main
```

!!! warning
    Never commit `.env` to version control. It is listed in `.gitignore`.

### 4 — Build and Push the Docker Image

The Docker image contains all Python dependencies but **no code and no data**. Code is
cloned at startup, and data is downloaded at startup. This keeps the image small and ensures
it only needs to be rebuilt when `environment.yml` changes.

The image is automatically built and published to
`ghcr.io/<your_github_username>/s2t-fs:latest` by the GitHub Actions workflow at
`.github/workflows/docker-publish.yml` whenever you push to `main`.

To build and push manually:

```bash
docker build -t ghcr.io/<your_github_username>/s2t-fs:latest .
docker push ghcr.io/<your_github_username>/s2t-fs:latest
```

---

## Step-by-Step: Launching an Instance

### Step 1 — Preview Available Offers (Dry Run)

Before committing, inspect the best available offer without creating an instance:

```bash
make vast-dry-run
```

Or with custom options:

```bash
make vast-dry-run GPU=RTX_4090 MAX_PRICE=0.60
```

This prints the best matching offer — GPU model, price per hour, CUDA version, and network
speeds — without spending any money.

### Step 2 — Launch the Instance

```bash
make vast-launch
```

Override defaults from the command line as needed:

```bash
make vast-launch GPU=A100_SXM4 MAX_PRICE=1.50 DISK=50
```

The script will:

1. Search for available offers matching your GPU, price, and bandwidth requirements.
2. Create an instance from the best match using the pre-built Docker image.
3. Pass `GIT_REPO_URL` and `GIT_BRANCH` as environment variables to the instance.
4. Start `onstart.sh`, which runs automatically:
   - Clones the repository to `/workspace/s2t-fs`
   - Runs `pip install -e .`
   - Downloads all four datasets via `gdown`
5. Poll every 10 seconds until SSH is ready (typically 30–90 seconds).
6. Print the SSH connection command when ready.

!!! note "Expected startup time"
    The instance is typically SSH-ready within 1–2 minutes. Dataset downloads
    add another 2–5 minutes depending on network speed. Check progress by SSH-ing in
    and watching the logs in `/workspace/s2t-fs/`.

### Step 3 — Connect via SSH

Use the SSH URL printed by `make vast-launch`:

```bash
ssh -p <port> root@<host>
```

### Step 4 — Run Experiments

Once connected:

```bash
cd /workspace/s2t-fs

# Verify the environment is ready
python -m s2t_fs.experiment --help

# Run the full multi-model benchmark
make run-comparison CONFIG=configs/model_comparison_voxpopuli.json

# Or run any other experiment
make run CONFIG=configs/margin_optimization_ami.json
```

!!! note "No Conda activation needed on the instance"
    The Docker image installs all packages into the base Python environment.
    You do not need to activate a Conda environment on the instance.

### Step 5 — Retrieve Results

Download the MLflow database to your local machine:

```bash
# On your local machine:
scp -P <port> root@<host>:/workspace/s2t-fs/s2t-fs-experiments.db .
```

Then view the results locally:

```bash
conda activate s2t-fs
mlflow ui --backend-store-uri sqlite:///s2t-fs-experiments.db
```

### Step 6 — Destroy the Instance

Always destroy the instance when you are done to avoid unnecessary charges:

```bash
make vast-stop INSTANCE=<instance_id>
```

To destroy all active instances:

```bash
make vast-stop
```

---

## Makefile Command Reference

| Command | Description |
|---------|-------------|
| `make vast-dry-run` | Preview the best offer without creating an instance |
| `make vast-launch` | Launch a GPU instance |
| `make vast-list` | List all your active instances |
| `make vast-stop INSTANCE=<id>` | Destroy a specific instance |
| `make vast-stop` | Destroy all active instances |

All commands accept the following overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU` | `RTX_3090` | GPU model name (as shown in `vastai search offers`) |
| `MAX_PRICE` | `0.30` | Maximum price in $/hr |
| `DISK` | `30` | Disk size in GB |

Example:

```bash
make vast-launch GPU=A100_SXM4 MAX_PRICE=1.20 DISK=60
```

---

## Under the Hood

### `scripts/vast_launch.sh`

This script:

1. Loads `.env` from the project root.
2. Parses CLI flags to override defaults.
3. Calls `vastai search offers` with GPU/price/bandwidth filters, sorted by price.
4. Selects the cheapest matching offer.
5. Creates the instance with the pre-built Docker image, the specified disk size, and
   `onstart.sh` as the startup command.
6. Polls `vastai ssh-url` until SSH is reachable.

### `scripts/onstart.sh`

This script runs once inside the container at instance startup:

1. Clones the repository (`GIT_REPO_URL`, `GIT_BRANCH`) to `/workspace/s2t-fs`.
2. Installs the package in editable mode: `pip install -e .`
3. Downloads the four datasets via `gdown` into `data/processed/`.

The script is baked into the Docker image at `/onstart.sh` so it is always available even
before the repository is cloned.

---

## Troubleshooting

**SSH timeout after `make vast-launch`:**
The instance may still be starting. Run `make vast-list` to check its status. If it shows
`running`, try connecting manually using `vastai ssh-url <instance_id>`.

**Dataset download fails:**
The `gdown` download may time out on slow instances. SSH in and re-run:
```bash
cd /workspace/s2t-fs && make download_processed_data
```

**Out of disk space:**
Increase the disk allocation: `make vast-launch DISK=60`.

**GPU not visible:**
Verify the instance is running and CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
