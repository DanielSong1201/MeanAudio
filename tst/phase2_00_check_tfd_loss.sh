#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" tst/phase2_check_tfd_loss.py \
  --device "$DEVICE" \
  --dtype "$DTYPE"
