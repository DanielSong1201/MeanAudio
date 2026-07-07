#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

export EVAL_CONSOLE_OUTPUT=0
export EVAL_FAILURE_FATAL="${EVAL_FAILURE_FATAL:-0}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"

TEACHER_POSITIVE_DIR_VALUE="${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6}"
TEACHER_POSITIVE_COUNT_VALUE="${TEACHER_POSITIVE_COUNT:-3}"
TRAIN_GPUS_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_NPROC_VALUE="${NPROC_PER_NODE:-2}"

if [[ "${PREPARE_TEACHER_POSITIVES:-1}" == "1" \
  && ! -f "${TEACHER_POSITIVE_DIR_VALUE}/complete.json" ]]; then
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS_VALUE}" \
  NPROC_PER_NODE="${TRAIN_NPROC_VALUE}" \
  TEACHER_POSITIVE_DIR="${TEACHER_POSITIVE_DIR_VALUE}" \
  TEACHER_POSITIVE_COUNT="${TEACHER_POSITIVE_COUNT_VALUE}" \
  TEACHER_POSITIVE_NUM_STEPS="${TEACHER_POSITIVE_NUM_STEPS:-25}" \
  TEACHER_POSITIVE_CFG_STRENGTH="${TEACHER_POSITIVE_CFG_STRENGTH:-6.0}" \
  bash drifting/scripts/flux/build_teacher_positive_bank_4gpu.sh
elif [[ ! -f "${TEACHER_POSITIVE_DIR_VALUE}/complete.json" ]]; then
  printf 'Missing completed teacher-positive bank: %s/complete.json\n' \
    "${TEACHER_POSITIVE_DIR_VALUE}" >&2
  exit 1
else
  printf 'Using completed teacher-positive bank: %s\n' \
    "${TEACHER_POSITIVE_DIR_VALUE}"
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS_VALUE}" \
NPROC_PER_NODE="${TRAIN_NPROC_VALUE}" \
EXP_ID="${EXP_ID:-flux_lr1e6_tfd100_anchor1_flow005_hybridpos4_warmup1000_200k_2gpu}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-4}" \
TEACHER_POSITIVE_DIR="${TEACHER_POSITIVE_DIR_VALUE}" \
TEACHER_POSITIVE_COUNT="${TEACHER_POSITIVE_COUNT_VALUE}" \
ITERATIONS="${ITERATIONS:-200000}" \
LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}" \
LAMBDA_FLOW="${LAMBDA_FLOW:-0.05}" \
LAMBDA_TFD="${LAMBDA_TFD:-100.0}" \
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}" \
FEATURE_NOISE="${FEATURE_NOISE:-0.1}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}" \
EMA_DECAY="${EMA_DECAY:-0.9999}" \
bash "${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}"

if [[ "${RUN_CFG_COMPARISON:-1}" == "1" ]]; then
  EXP_ID="${EXP_ID:-flux_lr1e6_tfd100_anchor1_flow005_hybridpos4_warmup1000_200k_2gpu}" \
  bash drifting/scripts/flux/compare_cfg_one_forward.sh
fi
