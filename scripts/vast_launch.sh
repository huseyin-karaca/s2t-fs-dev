#!/usr/bin/env bash
# Launch a vast.ai GPU instance pre-loaded with the s2t-fs Docker image.
# Reads credentials and defaults from .env in the project root.
#
# Usage:
#   bash scripts/vast_launch.sh [OPTIONS]
#
# Options (all override .env defaults):
#   --gpu        GPU model name to search for   (default: $GPU or RTX_3090)
#   --max-price  Max price in $/hr              (default: $MAX_PRICE or 0.30)
#   --disk       Disk size in GB                (default: $DISK or 30)
#   --branch     Git branch to clone            (default: $GIT_BRANCH or main)
#   --inet-down  Min download speed in Mbps     (default: $INET_DOWN or 500)
#   --inet-up    Min upload speed in Mbps       (default: $INET_UP or 100)
#   --dry-run    Print the offer without creating the instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Load .env ────────────────────────────────────────────────────────────────
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +o allexport
fi

# ── Defaults (can be overridden by .env or CLI flags) ────────────────────────
GPU="${GPU:-RTX_3090}"
MAX_PRICE="${MAX_PRICE:-0.30}"
DISK="${DISK:-30}"
GIT_BRANCH="${GIT_BRANCH:-main}"
INET_DOWN="${INET_DOWN:-500}"
INET_UP="${INET_UP:-100}"
VAST_IMAGE="${VAST_IMAGE:-ghcr.io/${GITHUB_ACTOR}/s2t-fs:latest}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/${GITHUB_ACTOR}/s2t-fs.git}"
DRY_RUN=false

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2";        shift 2 ;;
        --max-price)  MAX_PRICE="$2";  shift 2 ;;
        --disk)       DISK="$2";       shift 2 ;;
        --branch)     GIT_BRANCH="$2"; shift 2 ;;
        --inet-down)  INET_DOWN="$2";  shift 2 ;;
        --inet-up)    INET_UP="$2";    shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Validate required vars ────────────────────────────────────────────────────
: "${VAST_API_KEY:?Missing VAST_API_KEY in .env}"
: "${GITHUB_ACTOR:?Missing GITHUB_ACTOR in .env}"
: "${GIT_REPO_URL:?Missing GIT_REPO_URL in .env}"

# VAST_API_KEY is already exported via `set -o allexport` above.
# vastai reads it directly from the environment — no `vastai set api-key` needed.

echo "┌─────────────────────────────────────────┐"
echo "│          vast.ai instance launch         │"
echo "├─────────────────────────────────────────┤"
echo "│  GPU        : ${GPU}"
echo "│  Max price  : \$${MAX_PRICE}/hr"
echo "│  Disk       : ${DISK} GB"
echo "│  Down/Up    : ${INET_DOWN}/${INET_UP} Mbps"
echo "│  Image      : ${VAST_IMAGE}"
echo "│  Branch     : ${GIT_BRANCH}"
echo "└─────────────────────────────────────────┘"
echo ""

# ── Search for offers ─────────────────────────────────────────────────────────
echo "Searching for offers..."
OFFER_JSON=$(vastai search offers \
    "gpu_name=${GPU} dph<${MAX_PRICE} cuda_vers>=12.1 rentable=True num_gpus=1 inet_down>=${INET_DOWN} inet_up>=${INET_UP}" \
    -o dph_total --raw 2>/dev/null)

OFFER_COUNT=$(echo "${OFFER_JSON}" | jq 'length')

if [ "${OFFER_COUNT}" -eq 0 ]; then
    echo "No matching offers found."
    echo "Try: --max-price higher, a different --gpu, or check: vastai search offers --help"
    exit 1
fi

OFFER=$(echo "${OFFER_JSON}" | jq '.[0]')
OFFER_ID=$(echo "${OFFER}" | jq -r '.id')
OFFER_PRICE=$(echo "${OFFER}" | jq -r '.dph_total')
OFFER_GPU=$(echo "${OFFER}" | jq -r '.gpu_name')
OFFER_CUDA=$(echo "${OFFER}" | jq -r '.cuda_max_good')
OFFER_DOWN=$(echo "${OFFER}" | jq -r '.inet_down // "?"')
OFFER_UP=$(echo "${OFFER}" | jq -r '.inet_up // "?"')

echo "Best match:"
echo "  ID: ${OFFER_ID}  |  GPU: ${OFFER_GPU}  |  \$${OFFER_PRICE}/hr  |  CUDA ${OFFER_CUDA}  |  ${OFFER_DOWN}↓ / ${OFFER_UP}↑ Mbps"
echo ""

if [ "${DRY_RUN}" = "true" ]; then
    echo "[dry-run] Skipping instance creation."
    exit 0
fi

# ── Build env string for the instance ────────────────────────────────────────
ENV_FLAGS="-e GIT_REPO_URL=${GIT_REPO_URL} -e GIT_BRANCH=${GIT_BRANCH}"

# ── Create instance ───────────────────────────────────────────────────────────
echo "Creating instance..."
CREATE_OUTPUT=$(vastai create instance "${OFFER_ID}" \
    --image "${VAST_IMAGE}" \
    --disk "${DISK}" \
    --onstart-cmd "bash /onstart.sh" \
    --env "${ENV_FLAGS}" \
    --raw 2>&1)

INSTANCE_ID=$(echo "${CREATE_OUTPUT}" | jq -r '.new_contract // empty')

if [ -z "${INSTANCE_ID}" ]; then
    echo "Failed to create instance. vastai output:"
    echo "${CREATE_OUTPUT}"
    exit 1
fi

echo "Instance created: ${INSTANCE_ID}"
echo ""

# ── Poll for SSH readiness ────────────────────────────────────────────────────
echo "Waiting for SSH (this takes ~30–90s)..."
for i in $(seq 1 36); do
    SSH_URL=$(vastai ssh-url "${INSTANCE_ID}" 2>/dev/null || true)
    if [ -n "${SSH_URL}" ] && [ "${SSH_URL}" != "null" ]; then
        echo ""
        echo "┌─────────────────────────────────────────────────────────────┐"
        echo "│  Instance ready!                                            │"
        echo "│  Connect: ${SSH_URL}"
        echo "│                                                             │"
        echo "│  Once inside:                                               │"
        echo "│    cd /workspace/s2t-fs                                     │"
        echo "│    make run CONFIG=configs/<your_config>.json               │"
        echo "│                                                             │"
        echo "│  Destroy when done:                                         │"
        echo "│    make vast-stop INSTANCE=${INSTANCE_ID}"
        echo "└─────────────────────────────────────────────────────────────┘"
        exit 0
    fi
    printf "."
    sleep 10
done

echo ""
echo "Timed out waiting for SSH. Instance ${INSTANCE_ID} may still be starting."
echo "Check status: vastai show instances"
echo "Get SSH URL:  vastai ssh-url ${INSTANCE_ID}"
echo "Destroy:      make vast-stop INSTANCE=${INSTANCE_ID}"
