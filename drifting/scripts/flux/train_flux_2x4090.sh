#!/usr/bin/env bash
set -euo pipefail

# NCCL/CUDA fatal errors can terminate a worker with SIGABRT. Do not let the
# kernel write multi-gigabyte core.<pid> files into the repository.
ulimit -c 0

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-2}"

EXP_ID_VALUE="${EXP_ID:-flux_drifting_s_2x4090}"
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
EFFECTIVE_BATCH_SIZE_VALUE=$((BATCH_SIZE_VALUE * NPROC_PER_NODE_VALUE))
AUTO_RESUME_VALUE="${AUTO_RESUME:-1}"
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

printf 'train_config EXP_ID=%s\n' "${EXP_ID_VALUE}"
printf 'train_config OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT_VALUE}"
printf 'train_config TRAIN_LOCK=%s\n' "${TRAIN_LOCK_PATH}"
printf 'train_config GPU_COUNT=%s\n' "${NPROC_PER_NODE_VALUE}"
printf 'train_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'train_config CORE_DUMP_LIMIT=%s\n' "$(ulimit -c)"
printf 'train_config DDP_TIMEOUT_MINUTES=%s\n' "${DDP_TIMEOUT_MINUTES}"
printf 'train_config OMP_NUM_THREADS=%s\n' "${OMP_NUM_THREADS}"
printf 'train_config TORCH_NCCL_TRACE_BUFFER_SIZE=%s\n' "${TORCH_NCCL_TRACE_BUFFER_SIZE}"
printf 'train_config TORCH_NCCL_HIGH_PRIORITY=%s\n' "${TORCH_NCCL_HIGH_PRIORITY}"
printf 'train_config NCCL_P2P_DISABLE=%s\n' "${NCCL_P2P_DISABLE}"
printf 'train_config TRAIN_SCRIPT=%s\n' "drifting/scripts/flux/train_flux_2x4090.sh"
printf 'train_config TEACHER_WEIGHTS=%s\n' "${TEACHER_WEIGHTS_VALUE}"
printf 'train_config STUDENT_INIT=%s\n' "${STUDENT_INIT_VALUE}"
printf 'train_config BATCH_SIZE_PER_GPU=%s\n' "${BATCH_SIZE_VALUE}"
printf 'train_config EFFECTIVE_BATCH_SIZE=%s\n' "${EFFECTIVE_BATCH_SIZE_VALUE}"
printf 'train_config NUM_WORKERS=%s\n' "${NUM_WORKERS_VALUE}"
printf 'train_config ITERATIONS=%s\n' "${ITERATIONS_VALUE}"
printf 'train_config LEARNING_RATE=%s\n' "${LEARNING_RATE_VALUE}"
printf 'train_config LR_WARMUP_STEPS=%s\n' "${LR_WARMUP_STEPS_VALUE}"
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
printf 'train_config SAMPLES_PER_CONDITION=%s\n' "${SAMPLES_PER_CONDITION_VALUE}"
printf 'train_config MODEL_SAMPLES_PER_GPU=%s\n' "$((BATCH_SIZE_VALUE * SAMPLES_PER_CONDITION_VALUE))"
printf 'train_config TEACHER_POSITIVE_DIR=%s\n' "${TEACHER_POSITIVE_DIR_VALUE:-disabled}"
printf 'train_config TEACHER_POSITIVE_COUNT=%s\n' "${TEACHER_POSITIVE_COUNT_VALUE}"
printf 'train_config AUTO_RESUME=%s\n' "${AUTO_RESUME_VALUE}"
printf 'train_config USE_ROPE=%s\n' "1"
printf 'train_config AMP=%s\n' "1"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE_VALUE}" drifting/flux/train.py \
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
