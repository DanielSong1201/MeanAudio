#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

export EVAL_CONSOLE_OUTPUT="${EVAL_CONSOLE_OUTPUT:-0}"
export EVAL_FAILURE_FATAL="${EVAL_FAILURE_FATAL:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
EXP_ID="${EXP_ID:-flux_lr5e5_tfd100_anchor1_flow01_cond4_warmup500_200k_4gpu}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-4}" \
ITERATIONS="${ITERATIONS:-200000}" \
LEARNING_RATE="${LEARNING_RATE:-5e-5}" \
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}" \
LAMBDA_FLOW="${LAMBDA_FLOW:-0.1}" \
LAMBDA_TFD="${LAMBDA_TFD:-100.0}" \
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}" \
FEATURE_NOISE="${FEATURE_NOISE:-0.1}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}" \
EMA_DECAY="${EMA_DECAY:-0.9999}" \
bash "${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_4gpu_custom_loss.sh}"

if [[ "${RUN_CFG_COMPARISON:-1}" == "1" ]]; then
  EXP_ID="${EXP_ID:-flux_lr5e5_tfd100_anchor1_flow01_cond4_warmup500_200k_4gpu}" \
  bash drifting/scripts/flux/compare_cfg_one_forward.sh
fi
