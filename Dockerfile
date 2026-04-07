# Runtime environment image — no code, no data.
# Code is git-cloned and data is downloaded by onstart.sh at instance startup.
# Rebuild only when environment.yml (dependencies) changes.

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        rsync \
        curl \
        jq \
        make \
        vim \
    && rm -rf /var/lib/apt/lists/*

# ── All Python packages via pip (conda solver is too slow for CI) ─────────────
# torch is already in the base image; install everything else via pip.
RUN pip install --no-cache-dir \
        "numpy>=1.26" \
        "pandas>=2.1" \
        "pyarrow>=14" \
        "scikit-learn>=1.4" \
        "xgboost>=2.0" \
        "lightgbm>=4.0" \
        "optuna>=4.3" \
        "optuna-integration>=4.3" \
        "mlflow>=3.10" \
        "python-dateutil>=2.7" \
        python-dotenv \
        loguru \
        gdown \
        ruff \
        ipykernel \
        mkdocs \
        mkdocs-material

# ── Data is intentionally NOT baked in ────────────────────────────────────────
# Datasets are downloaded by onstart.sh at instance startup via gdown.
# This keeps the image small and CI fast; data persists on the instance disk.

# ── Onstart script ────────────────────────────────────────────────────────────
# Copied here so it survives inside the container without needing the repo.
COPY scripts/onstart.sh /onstart.sh
RUN chmod +x /onstart.sh

WORKDIR /workspace
