#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

args=(
  --repo-id "${HF_REPO_ID:-AndreasXi/MeanAudio}"
  --weights-dir "${WEIGHTS_DIR:-weights}"
)

if [[ -n "${INCLUDE_OPTIONAL_WEIGHTS:-}" ]]; then
  args+=(--include-optional)
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  args+=(--token "$HF_TOKEN")
fi

python tst/download_phase0_assets.py "${args[@]}"
