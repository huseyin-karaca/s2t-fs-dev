# Stage: full runtime environment with data baked in.
# Code is intentionally excluded — it is git-cloned at instance startup via onstart.sh.
# Rebuild this image only when environment.yml or data sources change.

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        rsync \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# ── Conda packages (into pre-existing base env) ────────────────────────────────
RUN conda install -y -c conda-forge \
        "numpy>=1.26" \
        "pandas>=2.1" \
        "pyarrow>=14" \
        "scikit-learn>=1.4" \
        "xgboost>=2.0" \
        "lightgbm>=4.0" \
        ruff \
        ipykernel \
    && conda clean -afy

# ── Pip packages (torch already provided by base image) ───────────────────────
RUN pip install --no-cache-dir \
        "optuna>=4.3" \
        "optuna-integration>=4.3" \
        "mlflow>=2.10" \
        python-dotenv \
        loguru \
        gdown \
        mkdocs \
        mkdocs-material

# ── Bake datasets into image ───────────────────────────────────────────────────
# Download IDs must stay in sync with Makefile:download_processed_data target.
RUN mkdir -p /data/processed && \
    gdown --fuzzy 1SzQLGqwpKuLEUf_4tIzvneg9T_RBNVtv -O /data/processed/voxpopuli.parquet    && \
    gdown --fuzzy 1gUPWooWpyNx-mbSB-mFDUqVZtIzD8Q7U -O /data/processed/librispeech.parquet && \
    gdown --fuzzy 1EzfaIOovXBY5pfxYdp9Pgq2YAXRWD50Q -O /data/processed/ami.parquet          && \
    gdown --fuzzy 1hpqNdUI4y_4lD2Gj3tC2QWnsUbK0lOxZ -O /data/processed/common_voice.parquet

# ── Onstart script ────────────────────────────────────────────────────────────
# Copied here so it survives inside the container without needing the repo.
COPY scripts/onstart.sh /onstart.sh
RUN chmod +x /onstart.sh

WORKDIR /workspace
