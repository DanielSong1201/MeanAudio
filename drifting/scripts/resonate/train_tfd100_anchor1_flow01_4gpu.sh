#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
EXP_ID="${EXP_ID:-resonate_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k}" \
LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}" \
ITERATIONS="${ITERATIONS:-200000}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-4}" \
LAMBDA_TFD="${LAMBDA_TFD:-100.0}" \
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}" \
LAMBDA_FLOW="${LAMBDA_FLOW:-0.1}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}" \
bash drifting/scripts/resonate/train.sh
