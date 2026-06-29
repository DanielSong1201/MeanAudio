#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}"
flow01_gpus="${FLOW01_GPUS:-0,1}"
flow00_gpus="${FLOW00_GPUS:-2,3}"
nproc_per_task="${NPROC_PER_TASK:-2}"
iterations="${ITERATIONS:-200000}"
learning_rate="${LEARNING_RATE:-5e-5}"
lambda_tfd="${LAMBDA_TFD:-1.0}"
lambda_anchor="${LAMBDA_ANCHOR:-1.0}"
feature_noise="${FEATURE_NOISE:-0.1}"
eval_interval="${EVAL_INTERVAL:-10000}"
ema_decay="${EMA_DECAY:-0.9999}"
live_tqdm="${LIVE_TQDM:-1}"
log_root="${SWEEP_LAUNCH_LOG_ROOT:-exps/drifting_flux/sweep_launch_logs}"

mkdir -p "${log_root}"

run_one() {
  local exp_id="$1"
  local lambda_flow="$2"
  local gpus="$3"
  local tqdm_position="$4"
  local log_path="${log_root}/${exp_id}_gpus_${gpus//,/_}.log"

  echo "================================================================"
  echo "Running FluxAudio 2-GPU flow ablation"
  echo "GPUS=${gpus}"
  echo "NPROC_PER_TASK=${nproc_per_task}"
  echo "EXP_ID=${exp_id}"
  echo "LAMBDA_FLOW=${lambda_flow}"
  echo "LAMBDA_TFD=${lambda_tfd}"
  echo "LAMBDA_ANCHOR=${lambda_anchor}"
  echo "ITERATIONS=${iterations}"
  echo "LEARNING_RATE=${learning_rate}"
  echo "EVAL_INTERVAL=${eval_interval}"
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
    ITERATIONS="${iterations}" \
    LEARNING_RATE="${learning_rate}" \
    LAMBDA_FLOW="${lambda_flow}" \
    LAMBDA_TFD="${lambda_tfd}" \
    LAMBDA_ANCHOR="${lambda_anchor}" \
    FEATURE_NOISE="${feature_noise}" \
    EVAL_INTERVAL="${eval_interval}" \
    EMA_DECAY="${ema_decay}" \
    bash "${train_script}"
  else
    CUDA_VISIBLE_DEVICES="${gpus}" \
    NPROC_PER_NODE="${nproc_per_task}" \
    EXP_ID="${exp_id}" \
    ITERATIONS="${iterations}" \
    LEARNING_RATE="${learning_rate}" \
    LAMBDA_FLOW="${lambda_flow}" \
    LAMBDA_TFD="${lambda_tfd}" \
    LAMBDA_ANCHOR="${lambda_anchor}" \
    FEATURE_NOISE="${feature_noise}" \
    EVAL_INTERVAL="${eval_interval}" \
    EMA_DECAY="${ema_decay}" \
    bash "${train_script}" 2>&1 | awk -v prefix="[GPU${gpus}] " '{ print prefix $0; fflush() }' | tee "${log_path}"
  fi
}

run_one "flux_sweep_tfd1_anchor1_flow01_200k" "0.1" "${flow01_gpus}" "0" &
pid_flow01=$!
run_one "flux_sweep_tfd1_anchor1_flow00_200k" "0.0" "${flow00_gpus}" "1" &
pid_flow00=$!

failed=0
if ! wait "${pid_flow01}"; then
  echo "Flow ablation failed on GPUs ${flow01_gpus}: flow=0.1"
  failed=1
fi

if ! wait "${pid_flow00}"; then
  echo "Flow ablation failed on GPUs ${flow00_gpus}: flow=0.0"
  failed=1
fi

if [[ "${failed}" != "0" ]]; then
  exit 1
fi

echo "FluxAudio flow=0.1/0.0 2-GPU ablation sweeps completed."
