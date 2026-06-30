#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-4}"

format_loss_value() {
  local value="$1"
  value="${value,,}"
  value="${value/+e/e}"
  if [[ "${value}" =~ ^([0-9]+)\.0+$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  elif [[ "${value}" == 0.* ]]; then
    printf '0%s' "${value#0.}"
  elif [[ "${value}" == .* ]]; then
    printf '0%s' "${value#.}"
  else
    value="${value//./}"
    value="${value//-/m}"
    value="${value//+/p}"
    printf '%s' "${value}"
  fi
}

format_iterations() {
  local iterations="$1"
  if (( iterations % 1000 == 0 )); then
    printf '%sk' "$((iterations / 1000))"
  else
    printf '%sits' "${iterations}"
  fi
}

TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/fluxaudio_s_full.pth}"
STUDENT_INIT_VALUE="${STUDENT_INIT:-weights/fluxaudio_s_full.pth}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-4}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-4}"
ITERATIONS_VALUE="${ITERATIONS:-200000}"
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
LAMBDA_FLOW_VALUE="${LAMBDA_FLOW:-0.05}"
LAMBDA_TFD_VALUE="${LAMBDA_TFD:-1.0}"
LAMBDA_ANCHOR_VALUE="${LAMBDA_ANCHOR:-1.0}"
FEATURE_NOISE_VALUE="${FEATURE_NOISE:-0.1}"
POOL_TOKENS_VALUE="${POOL_TOKENS:-64}"
AUTO_RESUME_VALUE="${AUTO_RESUME:-1}"
LOG_PREFIX_VALUE="${LOG_PREFIX:-}"
TQDM_POSITION_VALUE="${TQDM_POSITION:-0}"
TQDM_DESC_VALUE="${TQDM_DESC:-[GPU${CUDA_VISIBLE_DEVICES}]}"
QUIET_CONSOLE_AFTER_TQDM_VALUE="${QUIET_CONSOLE_AFTER_TQDM:-1}"
EVAL_TQDM_POSITION_OFFSET_VALUE="${EVAL_TQDM_POSITION_OFFSET:-1}"

FLOW_TAG="$(format_loss_value "${LAMBDA_FLOW_VALUE}")"
TFD_TAG="$(format_loss_value "${LAMBDA_TFD_VALUE}")"
ANCHOR_TAG="$(format_loss_value "${LAMBDA_ANCHOR_VALUE}")"
ITERATIONS_TAG="$(format_iterations "${ITERATIONS_VALUE}")"
EXP_ID_VALUE="${EXP_ID:-flux_tfd${TFD_TAG}_anchor${ANCHOR_TAG}_flow${FLOW_TAG}_${ITERATIONS_TAG}_4gpu}"

EFFECTIVE_BATCH_SIZE_VALUE=$((BATCH_SIZE_VALUE * NPROC_PER_NODE_VALUE))
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
print_config "GPU_COUNT" "${NPROC_PER_NODE_VALUE}"
print_config "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}"
print_config "DDP_TIMEOUT_MINUTES" "${DDP_TIMEOUT_MINUTES}"
print_config "TRAIN_SCRIPT" "drifting/scripts/flux/train_flux_4gpu_custom_loss.sh"
print_config "TEACHER_WEIGHTS" "${TEACHER_WEIGHTS_VALUE}"
print_config "STUDENT_INIT" "${STUDENT_INIT_VALUE}"
print_config "BATCH_SIZE_PER_GPU" "${BATCH_SIZE_VALUE}"
print_config "EFFECTIVE_BATCH_SIZE" "${EFFECTIVE_BATCH_SIZE_VALUE}"
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
print_config "TQDM_POSITION" "${TQDM_POSITION_VALUE}"
print_config "TQDM_DESC" "${TQDM_DESC_VALUE}"
print_config "EVAL_TQDM_POSITION_OFFSET" "${EVAL_TQDM_POSITION_OFFSET_VALUE}"
print_config "USE_ROPE" "1"
print_config "AMP" "1"

TQDM_POSITION="${TQDM_POSITION_VALUE}" \
TQDM_DESC="${TQDM_DESC_VALUE}" \
QUIET_CONSOLE_AFTER_TQDM="${QUIET_CONSOLE_AFTER_TQDM_VALUE}" \
EVAL_TQDM_POSITION_OFFSET="${EVAL_TQDM_POSITION_OFFSET_VALUE}" \
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
