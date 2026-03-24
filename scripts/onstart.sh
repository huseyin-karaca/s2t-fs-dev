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

echo "[onstart] Symlinking pre-baked data..."
mkdir -p "${REPO_DIR}/data"
ln -sfn /data/processed "${REPO_DIR}/data/processed"

echo "[onstart] Installing package in editable mode..."
cd "${REPO_DIR}"
pip install --no-cache-dir -e .

echo "[onstart] Done. Environment is ready."
echo "[onstart] Working directory: ${REPO_DIR}"
echo "[onstart] Run experiments with: make run CONFIG=configs/<your_config>.json"
