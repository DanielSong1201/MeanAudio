#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_VALUE="${PYTHON:-python}"
CHECKPOINT_VALUE="${CHECKPOINT:-weights/Resonate_GRPO.pth}"
DEVICE_VALUE="${DEVICE:-cpu}"
FLUX_BASELINE_VALUE="${FLUX_BASELINE:-a7f5239}"

exec "${PYTHON_VALUE}" drifting/Resonate/validate_phase_0_3.py \
  --checkpoint "${CHECKPOINT_VALUE}" \
  --device "${DEVICE_VALUE}" \
  --flux-baseline "${FLUX_BASELINE_VALUE}" \
  "$@"
