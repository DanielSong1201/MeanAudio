#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-2}"

EXP_ID_VALUE="${EXP_ID:-flux_drifting_s_2x4090}"
TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/fluxaudio_s_full.pth}"
STUDENT_INIT_VALUE="${STUDENT_INIT:-weights/fluxaudio_s_full.pth}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-4}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-4}"
ITERATIONS_VALUE="${ITERATIONS:-1000000}"
LEARNING_RATE_VALUE="${LEARNING_RATE:-5e-5}"
LOG_INTERVAL_VALUE="${LOG_INTERVAL:-20}"
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL_VALUE="${EVAL_INTERVAL:-10000}"
EVAL_OUTPUT_ROOT_VALUE="${EVAL_OUTPUT_ROOT:-exps/drifting_flux_eval}"
EVAL_GT_CACHE_VALUE="${EVAL_GT_CACHE:-data/audiocaps/test-features}"
EVAL_NUM_STEPS_VALUE="${EVAL_NUM_STEPS:-1}"
EVAL_CFG_STRENGTH_VALUE="${EVAL_CFG_STRENGTH:-4.5}"
EMA_DECAY_VALUE="${EMA_DECAY:-0.9999}"
EMA_START_VALUE="${EMA_START:-0}"
EMA_UPDATE_INTERVAL_VALUE="${EMA_UPDATE_INTERVAL:-1}"
EMA_DEVICE_VALUE="${EMA_DEVICE:-cpu}"
FEATURE_LAYERS_VALUE="${FEATURE_LAYERS:-joint_3,fused_3,fused_7}"
LAMBDA_FLOW_VALUE="${LAMBDA_FLOW:-1.0}"
LAMBDA_TFD_VALUE="${LAMBDA_TFD:-0.2}"
LAMBDA_ANCHOR_VALUE="${LAMBDA_ANCHOR:-0.05}"
FEATURE_NOISE_VALUE="${FEATURE_NOISE:-0.1}"
POOL_TOKENS_VALUE="${POOL_TOKENS:-64}"
EFFECTIVE_BATCH_SIZE_VALUE=$((BATCH_SIZE_VALUE * NPROC_PER_NODE_VALUE))
AUTO_RESUME_VALUE="${AUTO_RESUME:-1}"
AUTO_RESUME_ARGS=()
if [[ "${AUTO_RESUME_VALUE}" == "0" ]]; then
  AUTO_RESUME_ARGS=(--no-auto-resume)
fi

printf 'train_config EXP_ID=%s\n' "${EXP_ID_VALUE}"
printf 'train_config GPU_COUNT=%s\n' "${NPROC_PER_NODE_VALUE}"
printf 'train_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'train_config TRAIN_SCRIPT=%s\n' "drifting/scripts/flux/train_flux_2x4090.sh"
printf 'train_config TEACHER_WEIGHTS=%s\n' "${TEACHER_WEIGHTS_VALUE}"
printf 'train_config STUDENT_INIT=%s\n' "${STUDENT_INIT_VALUE}"
printf 'train_config BATCH_SIZE_PER_GPU=%s\n' "${BATCH_SIZE_VALUE}"
printf 'train_config EFFECTIVE_BATCH_SIZE=%s\n' "${EFFECTIVE_BATCH_SIZE_VALUE}"
printf 'train_config NUM_WORKERS=%s\n' "${NUM_WORKERS_VALUE}"
printf 'train_config ITERATIONS=%s\n' "${ITERATIONS_VALUE}"
printf 'train_config LEARNING_RATE=%s\n' "${LEARNING_RATE_VALUE}"
printf 'train_config LOG_INTERVAL=%s\n' "${LOG_INTERVAL_VALUE}"
printf 'train_config SAVE_INTERVAL=%s\n' "${SAVE_INTERVAL_VALUE}"
printf 'train_config EVAL_INTERVAL=%s\n' "${EVAL_INTERVAL_VALUE}"
printf 'train_config EVAL_OUTPUT_ROOT=%s\n' "${EVAL_OUTPUT_ROOT_VALUE}"
printf 'train_config EVAL_GT_CACHE=%s\n' "${EVAL_GT_CACHE_VALUE}"
printf 'train_config EVAL_NUM_STEPS=%s\n' "${EVAL_NUM_STEPS_VALUE}"
printf 'train_config EVAL_CFG_STRENGTH=%s\n' "${EVAL_CFG_STRENGTH_VALUE}"
printf 'train_config EMA_DECAY=%s\n' "${EMA_DECAY_VALUE}"
printf 'train_config EMA_START=%s\n' "${EMA_START_VALUE}"
printf 'train_config EMA_UPDATE_INTERVAL=%s\n' "${EMA_UPDATE_INTERVAL_VALUE}"
printf 'train_config EMA_DEVICE=%s\n' "${EMA_DEVICE_VALUE}"
printf 'train_config FEATURE_LAYERS=%s\n' "${FEATURE_LAYERS_VALUE}"
printf 'train_config LAMBDA_FLOW=%s\n' "${LAMBDA_FLOW_VALUE}"
printf 'train_config LAMBDA_TFD=%s\n' "${LAMBDA_TFD_VALUE}"
printf 'train_config LAMBDA_ANCHOR=%s\n' "${LAMBDA_ANCHOR_VALUE}"
printf 'train_config FEATURE_NOISE=%s\n' "${FEATURE_NOISE_VALUE}"
printf 'train_config POOL_TOKENS=%s\n' "${POOL_TOKENS_VALUE}"
printf 'train_config AUTO_RESUME=%s\n' "${AUTO_RESUME_VALUE}"
printf 'train_config USE_ROPE=%s\n' "1"
printf 'train_config AMP=%s\n' "1"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE_VALUE}" drifting/flux/train.py \
  --exp-id "${EXP_ID_VALUE}" \
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}" \
  --student-init "${STUDENT_INIT_VALUE}" \
  --batch-size "${BATCH_SIZE_VALUE}" \
  --num-workers "${NUM_WORKERS_VALUE}" \
  --iterations "${ITERATIONS_VALUE}" \
  --learning-rate "${LEARNING_RATE_VALUE}" \
  --log-interval "${LOG_INTERVAL_VALUE}" \
  --save-interval "${SAVE_INTERVAL_VALUE}" \
  --eval-interval "${EVAL_INTERVAL_VALUE}" \
  --eval-output-root "${EVAL_OUTPUT_ROOT_VALUE}" \
  --eval-gt-cache "${EVAL_GT_CACHE_VALUE}" \
  --eval-num-steps "${EVAL_NUM_STEPS_VALUE}" \
  --eval-cfg-strength "${EVAL_CFG_STRENGTH_VALUE}" \
  --ema-decay "${EMA_DECAY_VALUE}" \
  --ema-start "${EMA_START_VALUE}" \
  --ema-update-interval "${EMA_UPDATE_INTERVAL_VALUE}" \
  --ema-device "${EMA_DEVICE_VALUE}" \
  --feature-layers "${FEATURE_LAYERS_VALUE}" \
  --lambda-flow "${LAMBDA_FLOW_VALUE}" \
  --lambda-tfd "${LAMBDA_TFD_VALUE}" \
  --lambda-anchor "${LAMBDA_ANCHOR_VALUE}" \
  --feature-noise "${FEATURE_NOISE_VALUE}" \
  --pool-tokens "${POOL_TOKENS_VALUE}" \
  --use-rope \
  --amp \
  "${AUTO_RESUME_ARGS[@]}"
