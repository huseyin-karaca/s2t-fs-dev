#!/usr/bin/env bash
# Runs once when the vast.ai instance starts.
# Clones the project repo, wires up the pre-baked data, and installs the package.
# Environment variables expected (passed via vastai create instance --env):
#   GIT_REPO_URL  — e.g. https://github.com/yourname/s2t-fs.git
#   GIT_BRANCH    — branch to checkout (default: main)

set -euo pipefail

REPO_DIR="/workspace/s2t-fs"
BRANCH="${GIT_BRANCH:-main}"

echo "[onstart] Cloning ${GIT_REPO_URL} (branch: ${BRANCH})..."
git clone --branch "${BRANCH}" --depth 1 "${GIT_REPO_URL}" "${REPO_DIR}"

echo "[onstart] Installing package in editable mode..."
cd "${REPO_DIR}"
pip install --no-cache-dir -e .

echo "[onstart] Downloading datasets..."
mkdir -p "${REPO_DIR}/data/processed"
gdown --fuzzy 1SzQLGqwpKuLEUf_4tIzvneg9T_RBNVtv -O "${REPO_DIR}/data/processed/voxpopuli.parquet"
gdown --fuzzy 1gUPWooWpyNx-mbSB-mFDUqVZtIzD8Q7U -O "${REPO_DIR}/data/processed/librispeech.parquet"
gdown --fuzzy 1EzfaIOovXBY5pfxYdp9Pgq2YAXRWD50Q -O "${REPO_DIR}/data/processed/ami.parquet"
gdown --fuzzy 1hpqNdUI4y_4lD2Gj3tC2QWnsUbK0lOxZ -O "${REPO_DIR}/data/processed/common_voice.parquet"

echo "[onstart] Done. Environment is ready."
echo "[onstart] Working directory: ${REPO_DIR}"
echo "[onstart] Run experiments with: make run CONFIG=configs/<your_config>.json"
