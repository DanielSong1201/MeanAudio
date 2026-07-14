#!/usr/bin/env bash
set -euo pipefail

# NCCL/CUDA fatal errors can terminate a worker with SIGABRT. Do not let the
# kernel write multi-gigabyte core.<pid> files into the repository.
ulimit -c 0

cd "$(dirname "$0")/../../.."

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXP_ID_VALUE="${EXP_ID:-flux_drifting_s_1x4090}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-exps/drifting_flux}"
TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/fluxaudio_s_full.pth}"
STUDENT_INIT_VALUE="${STUDENT_INIT:-weights/fluxaudio_s_full.pth}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-4}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-4}"
ITERATIONS_VALUE="${ITERATIONS:-1000000}"
LEARNING_RATE_VALUE="${LEARNING_RATE:-5e-5}"
LR_WARMUP_STEPS_VALUE="${LR_WARMUP_STEPS:-500}"
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
SAMPLES_PER_CONDITION_VALUE="${SAMPLES_PER_CONDITION:-1}"
TEACHER_POSITIVE_DIR_VALUE="${TEACHER_POSITIVE_DIR:-}"
TEACHER_POSITIVE_COUNT_VALUE="${TEACHER_POSITIVE_COUNT:-3}"
AUTO_RESUME_VALUE="${AUTO_RESUME:-1}"
LOG_PREFIX_VALUE="${LOG_PREFIX:-}"
AUTO_RESUME_ARGS=()
if [[ "${AUTO_RESUME_VALUE}" == "0" ]]; then
  AUTO_RESUME_ARGS=(--no-auto-resume)
fi
TEACHER_POSITIVE_ARGS=()
if [[ -n "${TEACHER_POSITIVE_DIR_VALUE}" ]]; then
  TEACHER_POSITIVE_ARGS=(
    --teacher-positive-dir "${TEACHER_POSITIVE_DIR_VALUE}"
    --teacher-positive-count "${TEACHER_POSITIVE_COUNT_VALUE}"
  )
fi

if ! command -v flock >/dev/null 2>&1; then
  printf 'ERROR: flock is required to prevent duplicate training for one exp_id.\n' >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT_VALUE}/${EXP_ID_VALUE}"
TRAIN_LOCK_PATH="${OUTPUT_ROOT_VALUE}/${EXP_ID_VALUE}/.train.lock"
exec 9>"${TRAIN_LOCK_PATH}"
if ! flock -n 9; then
  printf 'ERROR: exp_id %s is already running (lock: %s).\n' "${EXP_ID_VALUE}" "${TRAIN_LOCK_PATH}" >&2
  exit 1
fi

print_config() {
  if [[ -n "${LOG_PREFIX_VALUE}" ]]; then
    printf '%s train_config %s=%s\n' "${LOG_PREFIX_VALUE}" "$1" "$2"
  else
    printf 'train_config %s=%s\n' "$1" "$2"
  fi
}

print_config "EXP_ID" "${EXP_ID_VALUE}"
print_config "OUTPUT_ROOT" "${OUTPUT_ROOT_VALUE}"
print_config "TRAIN_LOCK" "${TRAIN_LOCK_PATH}"
print_config "GPU_COUNT" "1"
print_config "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}"
print_config "CORE_DUMP_LIMIT" "$(ulimit -c)"
print_config "TRAIN_SCRIPT" "drifting/scripts/flux/train_flux_1x4090.sh"
print_config "TEACHER_WEIGHTS" "${TEACHER_WEIGHTS_VALUE}"
print_config "STUDENT_INIT" "${STUDENT_INIT_VALUE}"
print_config "BATCH_SIZE_PER_GPU" "${BATCH_SIZE_VALUE}"
print_config "EFFECTIVE_BATCH_SIZE" "${BATCH_SIZE_VALUE}"
print_config "NUM_WORKERS" "${NUM_WORKERS_VALUE}"
print_config "ITERATIONS" "${ITERATIONS_VALUE}"
print_config "LEARNING_RATE" "${LEARNING_RATE_VALUE}"
print_config "LR_WARMUP_STEPS" "${LR_WARMUP_STEPS_VALUE}"
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
print_config "SAMPLES_PER_CONDITION" "${SAMPLES_PER_CONDITION_VALUE}"
print_config "MODEL_SAMPLES_PER_GPU" "$((BATCH_SIZE_VALUE * SAMPLES_PER_CONDITION_VALUE))"
print_config "TEACHER_POSITIVE_DIR" "${TEACHER_POSITIVE_DIR_VALUE:-disabled}"
print_config "TEACHER_POSITIVE_COUNT" "${TEACHER_POSITIVE_COUNT_VALUE}"
print_config "AUTO_RESUME" "${AUTO_RESUME_VALUE}"
print_config "USE_ROPE" "1"
print_config "AMP" "1"

python drifting/flux/train.py \
  --exp-id "${EXP_ID_VALUE}" \
  --output-root "${OUTPUT_ROOT_VALUE}" \
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}" \
  --student-init "${STUDENT_INIT_VALUE}" \
  --batch-size "${BATCH_SIZE_VALUE}" \
  --num-workers "${NUM_WORKERS_VALUE}" \
  --iterations "${ITERATIONS_VALUE}" \
  --learning-rate "${LEARNING_RATE_VALUE}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS_VALUE}" \
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
  --samples-per-condition "${SAMPLES_PER_CONDITION_VALUE}" \
  "${TEACHER_POSITIVE_ARGS[@]}" \
  --use-rope \
  --amp \
  "${AUTO_RESUME_ARGS[@]}"
