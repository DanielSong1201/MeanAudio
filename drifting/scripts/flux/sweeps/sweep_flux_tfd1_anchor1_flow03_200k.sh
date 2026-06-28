#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

EXP_ID="${EXP_ID:-flux_sweep_tfd1_anchor1_flow03_200k}" \
ITERATIONS="${ITERATIONS:-200000}" \
LEARNING_RATE="${LEARNING_RATE:-5e-5}" \
LAMBDA_FLOW="${LAMBDA_FLOW:-0.3}" \
LAMBDA_TFD="${LAMBDA_TFD:-1.0}" \
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}" \
FEATURE_NOISE="${FEATURE_NOISE:-0.1}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}" \
EMA_DECAY="${EMA_DECAY:-0.9999}" \
bash "${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_1x4090.sh}"
