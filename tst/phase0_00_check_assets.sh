#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"

python tst/check_phase0_assets.py \
  --teacher-weights "${TEACHER_WEIGHTS:-${WEIGHTS_DIR}/fluxaudio_s_full.pth}" \
  --baseline-weights "${BASELINE_WEIGHTS:-${WEIGHTS_DIR}/meanaudio_s_full.pth}" \
  --vae-weights "${VAE_WEIGHTS:-${WEIGHTS_DIR}/v1-16.pth}" \
  --vocoder-weights "${VOCODER_WEIGHTS:-${WEIGHTS_DIR}/best_netG.pt}"
