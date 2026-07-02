#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_VALUE="${PYTHON:-python}"
MODEL_FILE_VALUE="${MODEL_FILE:-Resonate_GRPO.pth}"
WEIGHTS_DIR_VALUE="${WEIGHTS_DIR:-weights}"
HF_REPO_VALUE="${HF_REPO:-AndreasXi/Resonate}"
HF_REVISION_VALUE="${HF_REVISION:-main}"

exec "${PYTHON_VALUE}" drifting/Resonate/download_resonate_model.py \
  --repo-id "${HF_REPO_VALUE}" \
  --model-file "${MODEL_FILE_VALUE}" \
  --weights-dir "${WEIGHTS_DIR_VALUE}" \
  --revision "${HF_REVISION_VALUE}" \
  "$@"
