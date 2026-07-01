#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

export EVAL_CONSOLE_OUTPUT="${EVAL_CONSOLE_OUTPUT:-0}"
export EVAL_FAILURE_FATAL="${EVAL_FAILURE_FATAL:-0}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"

train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}"
tfd1_gpus="${TFD1_GPUS:-0,1}"
tfd100_gpus="${TFD100_GPUS:-2,3}"
nproc_per_task="${NPROC_PER_TASK:-2}"
iterations="${ITERATIONS:-200000}"
learning_rate="${LEARNING_RATE:-1e-6}"
lr_warmup_steps="${LR_WARMUP_STEPS:-1000}"
feature_noise="${FEATURE_NOISE:-0.1}"
eval_interval="${EVAL_INTERVAL:-10000}"
ema_decay="${EMA_DECAY:-0.9999}"
auto_resume="${AUTO_RESUME:-1}"
live_tqdm="${LIVE_TQDM:-1}"
log_root="${SWEEP_LAUNCH_LOG_ROOT:-exps/drifting_flux/sweep_launch_logs}"
teacher_positive_dir="${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6}"
teacher_positive_count="${TEACHER_POSITIVE_COUNT:-3}"

tfd1_exp_id="${TFD1_EXP_ID:-flux_lr1e6_tfd1_anchor1_flow005_hybridpos4_warmup1000_200k_2gpu}"
tfd1_lambda_flow="${TFD1_LAMBDA_FLOW:-0.05}"
tfd1_lambda_tfd="${TFD1_LAMBDA_TFD:-1.0}"
tfd1_lambda_anchor="${TFD1_LAMBDA_ANCHOR:-1.0}"

tfd100_exp_id="${TFD100_EXP_ID:-flux_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k_2gpu}"
tfd100_lambda_flow="${TFD100_LAMBDA_FLOW:-0.1}"
tfd100_lambda_tfd="${TFD100_LAMBDA_TFD:-100.0}"
tfd100_lambda_anchor="${TFD100_LAMBDA_ANCHOR:-1.0}"

mkdir -p "${log_root}"

if [[ "${PREPARE_TEACHER_POSITIVES:-1}" == "1" \
  && ! -f "${teacher_positive_dir}/complete.json" ]]; then
  CUDA_VISIBLE_DEVICES="${POSITIVE_BANK_GPUS:-0,1,2,3}" \
  NPROC_PER_NODE="${POSITIVE_BANK_NPROC:-4}" \
  TEACHER_POSITIVE_DIR="${teacher_positive_dir}" \
  TEACHER_POSITIVE_COUNT="${teacher_positive_count}" \
  TEACHER_POSITIVE_NUM_STEPS="${TEACHER_POSITIVE_NUM_STEPS:-25}" \
  TEACHER_POSITIVE_CFG_STRENGTH="${TEACHER_POSITIVE_CFG_STRENGTH:-6.0}" \
  bash drifting/scripts/flux/build_teacher_positive_bank_4gpu.sh
elif [[ ! -f "${teacher_positive_dir}/complete.json" ]]; then
  printf 'Missing completed teacher-positive bank: %s/complete.json\n' \
    "${teacher_positive_dir}" >&2
  exit 1
else
  printf 'Using completed teacher-positive bank: %s\n' \
    "${teacher_positive_dir}"
fi

run_one() {
  local label="$1"
  local exp_id="$2"
  local lambda_flow="$3"
  local lambda_tfd="$4"
  local lambda_anchor="$5"
  local gpus="$6"
  local tqdm_position="$7"
  local log_path="${log_root}/${exp_id}_gpus_${gpus//,/_}.log"

  echo "================================================================"
  echo "Running one FluxAudio experiment on two GPUs"
  echo "LABEL=${label}"
  echo "GPUS=${gpus}"
  echo "NPROC_PER_TASK=${nproc_per_task}"
  echo "EXP_ID=${exp_id}"
  echo "LAMBDA_FLOW=${lambda_flow}"
  echo "LAMBDA_TFD=${lambda_tfd}"
  echo "LAMBDA_ANCHOR=${lambda_anchor}"
  echo "ITERATIONS=${iterations}"
  echo "LEARNING_RATE=${learning_rate}"
  echo "LR_WARMUP_STEPS=${lr_warmup_steps}"
  echo "FEATURE_NOISE=${feature_noise}"
  echo "EVAL_INTERVAL=${eval_interval}"
  echo "EMA_DECAY=${ema_decay}"
  echo "AUTO_RESUME=${auto_resume}"
  echo "DDP_TIMEOUT_MINUTES=${DDP_TIMEOUT_MINUTES}"
  echo "TRAIN_SCRIPT=${train_script}"
  echo "LOG=${log_path}"
  echo "LIVE_TQDM=${live_tqdm}"
  echo "================================================================"

  if [[ "${live_tqdm}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpus}" \
    NPROC_PER_NODE="${nproc_per_task}" \
    TQDM_POSITION="${tqdm_position}" \
    TQDM_DESC="[GPU${gpus}]" \
    LOG_PREFIX="[GPU${gpus}]" \
    QUIET_CONSOLE_AFTER_TQDM=1 \
    EVAL_TQDM_POSITION_OFFSET=2 \
    EXP_ID="${exp_id}" \
    BATCH_SIZE="${BATCH_SIZE:-1}" \
    SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-4}" \
    TEACHER_POSITIVE_DIR="${teacher_positive_dir}" \
    TEACHER_POSITIVE_COUNT="${teacher_positive_count}" \
    ITERATIONS="${iterations}" \
    LEARNING_RATE="${learning_rate}" \
    LR_WARMUP_STEPS="${lr_warmup_steps}" \
    LAMBDA_FLOW="${lambda_flow}" \
    LAMBDA_TFD="${lambda_tfd}" \
    LAMBDA_ANCHOR="${lambda_anchor}" \
    FEATURE_NOISE="${feature_noise}" \
    EVAL_INTERVAL="${eval_interval}" \
    EMA_DECAY="${ema_decay}" \
    AUTO_RESUME="${auto_resume}" \
    bash "${train_script}"
  else
    CUDA_VISIBLE_DEVICES="${gpus}" \
    NPROC_PER_NODE="${nproc_per_task}" \
    EXP_ID="${exp_id}" \
    BATCH_SIZE="${BATCH_SIZE:-1}" \
    SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-4}" \
    TEACHER_POSITIVE_DIR="${teacher_positive_dir}" \
    TEACHER_POSITIVE_COUNT="${teacher_positive_count}" \
    ITERATIONS="${iterations}" \
    LEARNING_RATE="${learning_rate}" \
    LR_WARMUP_STEPS="${lr_warmup_steps}" \
    LAMBDA_FLOW="${lambda_flow}" \
    LAMBDA_TFD="${lambda_tfd}" \
    LAMBDA_ANCHOR="${lambda_anchor}" \
    FEATURE_NOISE="${feature_noise}" \
    EVAL_INTERVAL="${eval_interval}" \
    EMA_DECAY="${ema_decay}" \
    AUTO_RESUME="${auto_resume}" \
    bash "${train_script}" 2>&1 | awk -v prefix="[GPU${gpus}] " '{ print prefix $0; fflush() }' | tee "${log_path}"
  fi
}

failed=()

run_one "tfd1_flow005" \
  "${tfd1_exp_id}" \
  "${tfd1_lambda_flow}" \
  "${tfd1_lambda_tfd}" \
  "${tfd1_lambda_anchor}" \
  "${tfd1_gpus}" \
  "0" &
pid_tfd1=$!

run_one "tfd100_flow01" \
  "${tfd100_exp_id}" \
  "${tfd100_lambda_flow}" \
  "${tfd100_lambda_tfd}" \
  "${tfd100_lambda_anchor}" \
  "${tfd100_gpus}" \
  "1" &
pid_tfd100=$!

if ! wait "${pid_tfd1}"; then
  failed+=("${tfd1_exp_id}")
  echo "Sweep failed: ${tfd1_exp_id}" >&2
fi

if ! wait "${pid_tfd100}"; then
  failed+=("${tfd100_exp_id}")
  echo "Sweep failed: ${tfd100_exp_id}" >&2
fi

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed sweeps:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi

echo "Both parallel two-GPU FluxAudio experiments completed."
