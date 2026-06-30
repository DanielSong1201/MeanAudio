#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export DRIFTING_CKPT_DIR="${DRIFTING_CKPT_DIR:-drifting/ckpts}"
bash drifting/scripts/prepare_hf_ckpts.sh
export HF_HOME="${DRIFTING_CKPT_DIR}/huggingface"
export HF_HUB_CACHE="${DRIFTING_CKPT_DIR}/huggingface/hub"
export TRANSFORMERS_CACHE="${DRIFTING_CKPT_DIR}/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXP_ID_VALUE="${EXP_ID:-flux_drifting_s_1x4090}"
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
AUTO_RESUME_VALUE="${AUTO_RESUME:-1}"
LOG_PREFIX_VALUE="${LOG_PREFIX:-}"
AUTO_RESUME_ARGS=()
if [[ "${AUTO_RESUME_VALUE}" == "0" ]]; then
  AUTO_RESUME_ARGS=(--no-auto-resume)
fi

print_config() {
  if [[ -n "${LOG_PREFIX_VALUE}" ]]; then
    printf '%s train_config %s=%s\n' "${LOG_PREFIX_VALUE}" "$1" "$2"
  else
    printf 'train_config %s=%s\n' "$1" "$2"
  fi
}

print_config "EXP_ID" "${EXP_ID_VALUE}"
print_config "GPU_COUNT" "1"
print_config "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}"
print_config "TRAIN_SCRIPT" "drifting/scripts/flux/train_flux_1x4090.sh"
print_config "TEACHER_WEIGHTS" "${TEACHER_WEIGHTS_VALUE}"
print_config "STUDENT_INIT" "${STUDENT_INIT_VALUE}"
print_config "BATCH_SIZE_PER_GPU" "${BATCH_SIZE_VALUE}"
print_config "EFFECTIVE_BATCH_SIZE" "${BATCH_SIZE_VALUE}"
print_config "NUM_WORKERS" "${NUM_WORKERS_VALUE}"
print_config "ITERATIONS" "${ITERATIONS_VALUE}"
print_config "LEARNING_RATE" "${LEARNING_RATE_VALUE}"
print_config "LOG_INTERVAL" "${LOG_INTERVAL_VALUE}"
print_config "SAVE_INTERVAL" "${SAVE_INTERVAL_VALUE}"
print_config "EVAL_INTERVAL" "${EVAL_INTERVAL_VALUE}"
print_config "EVAL_OUTPUT_ROOT" "${EVAL_OUTPUT_ROOT_VALUE}"
print_config "EVAL_GT_CACHE" "${EVAL_GT_CACHE_VALUE}"
print_config "EVAL_NUM_STEPS" "${EVAL_NUM_STEPS_VALUE}"
print_config "EVAL_CFG_STRENGTH" "${EVAL_CFG_STRENGTH_VALUE}"
print_config "EMA_DECAY" "${EMA_DECAY_VALUE}"
print_config "EMA_START" "${EMA_START_VALUE}"
print_config "EMA_UPDATE_INTERVAL" "${EMA_UPDATE_INTERVAL_VALUE}"
print_config "EMA_DEVICE" "${EMA_DEVICE_VALUE}"
print_config "FEATURE_LAYERS" "${FEATURE_LAYERS_VALUE}"
print_config "LAMBDA_FLOW" "${LAMBDA_FLOW_VALUE}"
print_config "LAMBDA_TFD" "${LAMBDA_TFD_VALUE}"
print_config "LAMBDA_ANCHOR" "${LAMBDA_ANCHOR_VALUE}"
print_config "FEATURE_NOISE" "${FEATURE_NOISE_VALUE}"
print_config "POOL_TOKENS" "${POOL_TOKENS_VALUE}"
print_config "AUTO_RESUME" "${AUTO_RESUME_VALUE}"
print_config "USE_ROPE" "1"
print_config "AMP" "1"

python drifting/flux/train.py \
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
