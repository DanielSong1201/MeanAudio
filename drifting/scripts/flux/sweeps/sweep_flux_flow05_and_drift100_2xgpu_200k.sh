#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}"
flow05_gpus="${FLOW05_GPUS:-0,1}"
drift_gpus="${DRIFT_GPUS:-2,3}"
nproc_per_task="${NPROC_PER_TASK:-2}"
iterations="${ITERATIONS:-200000}"
learning_rate="${LEARNING_RATE:-5e-5}"
feature_noise="${FEATURE_NOISE:-0.1}"
eval_interval="${EVAL_INTERVAL:-10000}"
ema_decay="${EMA_DECAY:-0.9999}"
live_tqdm="${LIVE_TQDM:-1}"
log_root="${SWEEP_LAUNCH_LOG_ROOT:-exps/drifting_flux/sweep_launch_logs}"

flow05_exp_id="${FLOW05_EXP_ID:-flux_tfd1_anchor1_flow005_200k}"
flow05_lambda_flow="${FLOW05_LAMBDA_FLOW:-0.05}"
flow05_lambda_tfd="${FLOW05_LAMBDA_TFD:-1.0}"
flow05_lambda_anchor="${FLOW05_LAMBDA_ANCHOR:-1.0}"

drift_exp_id="${DRIFT_EXP_ID:-flux_sweep_tfd100_anchor1_flow01_200k}"
drift_lambda_flow="${DRIFT_LAMBDA_FLOW:-0.1}"
drift_lambda_tfd="${DRIFT_LAMBDA_TFD:-100.0}"
drift_lambda_anchor="${DRIFT_LAMBDA_ANCHOR:-1.0}"

mkdir -p "${log_root}"

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
  echo "Running FluxAudio parallel 2-GPU sweep"
  echo "LABEL=${label}"
  echo "GPUS=${gpus}"
  echo "NPROC_PER_TASK=${nproc_per_task}"
  echo "EXP_ID=${exp_id}"
  echo "LAMBDA_FLOW=${lambda_flow}"
  echo "LAMBDA_TFD=${lambda_tfd}"
  echo "LAMBDA_ANCHOR=${lambda_anchor}"
  echo "ITERATIONS=${iterations}"
  echo "LEARNING_RATE=${learning_rate}"
  echo "FEATURE_NOISE=${feature_noise}"
  echo "EVAL_INTERVAL=${eval_interval}"
  echo "EMA_DECAY=${ema_decay}"
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

failed=()

run_one "flow05" "${flow05_exp_id}" "${flow05_lambda_flow}" "${flow05_lambda_tfd}" "${flow05_lambda_anchor}" "${flow05_gpus}" "0" &
pid_flow05=$!

run_one "enhanced_drift" "${drift_exp_id}" "${drift_lambda_flow}" "${drift_lambda_tfd}" "${drift_lambda_anchor}" "${drift_gpus}" "1" &
pid_drift=$!

if ! wait "${pid_flow05}"; then
  failed+=("${flow05_exp_id}")
  echo "Sweep failed: ${flow05_exp_id}" >&2
fi

if ! wait "${pid_drift}"; then
  failed+=("${drift_exp_id}")
  echo "Sweep failed: ${drift_exp_id}" >&2
fi

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed sweeps:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi

echo "FluxAudio flow=0.05 and enhanced-drift parallel 2-GPU sweeps completed."
