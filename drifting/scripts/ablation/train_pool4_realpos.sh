#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export EXP_ID="${EXP_ID:-flux_ablate_pool4_fluxpos_flow005_tfd100}"
export ITERATIONS="${ITERATIONS:-200000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-4}"
export TEACHER_POSITIVE_DIR="${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-fluxaudio-s-full-25step-cfg6}"
export TEACHER_POSITIVE_COUNT="${TEACHER_POSITIVE_COUNT:-3}"
export POOL_TOKENS="${POOL_TOKENS:-4}"
export FEATURE_NOISE="${FEATURE_NOISE:-0.1}"
export FEATURE_LAYERS="${FEATURE_LAYERS:-joint_3,fused_3,fused_7}"
export LAMBDA_FLOW="${LAMBDA_FLOW:-0.05}"
export LAMBDA_TFD="${LAMBDA_TFD:-100.0}"
export LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export EVAL_CFG_STRENGTH="${EVAL_CFG_STRENGTH:-4.5}"
export AUTO_RESUME="${AUTO_RESUME:-1}"

if [[ ! -f "${TEACHER_POSITIVE_DIR}/complete.json" ]]; then
  TEACHER_POSITIVE_VARIANT=fluxaudio_s \
  TEACHER_POSITIVE_WEIGHTS="${TEACHER_POSITIVE_WEIGHTS:-weights/fluxaudio_s_full.pth}" \
  TEACHER_POSITIVE_DIR="${TEACHER_POSITIVE_DIR}" \
  TEACHER_POSITIVE_COUNT="${TEACHER_POSITIVE_COUNT}" \
  TEACHER_POSITIVE_NUM_STEPS="${TEACHER_POSITIVE_NUM_STEPS:-25}" \
  TEACHER_POSITIVE_CFG_STRENGTH="${TEACHER_POSITIVE_CFG_STRENGTH:-6.0}" \
  TEACHER_POSITIVE_OVERWRITE=0 \
  bash drifting/scripts/flux/build_teacher_positive_bank_4gpu.sh
fi

bash drifting/scripts/flux/train_flux_4gpu_custom_loss.sh
