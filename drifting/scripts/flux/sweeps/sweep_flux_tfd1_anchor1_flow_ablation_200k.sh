#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

export EVAL_CONSOLE_OUTPUT="${EVAL_CONSOLE_OUTPUT:-0}"
export EVAL_FAILURE_FATAL="${EVAL_FAILURE_FATAL:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_1x4090.sh}"
gpu0="${GPU0:-0}"
gpu1="${GPU1:-1}"
gpu2="${GPU2:-2}"
iterations="${ITERATIONS:-200000}"
learning_rate="${LEARNING_RATE:-5e-5}"
lambda_tfd="${LAMBDA_TFD:-1.0}"
lambda_anchor="${LAMBDA_ANCHOR:-1.0}"
feature_noise="${FEATURE_NOISE:-0.1}"
eval_interval="${EVAL_INTERVAL:-10000}"
ema_decay="${EMA_DECAY:-0.9999}"
live_tqdm="${LIVE_TQDM:-1}"

run_one() {
  local exp_id="$1"
  local lambda_flow="$2"
  local gpu="$3"
  local tqdm_position="$4"

  echo "================================================================"
  echo "Running FluxAudio flow ablation"
  echo "GPU=${gpu}"
  echo "EXP_ID=${exp_id}"
  echo "LAMBDA_FLOW=${lambda_flow}"
  echo "LAMBDA_TFD=${lambda_tfd}"
  echo "LAMBDA_ANCHOR=${lambda_anchor}"
  echo "ITERATIONS=${iterations}"
  echo "LEARNING_RATE=${learning_rate}"
  echo "TRAIN_SCRIPT=${train_script}"
  echo "LIVE_TQDM=${live_tqdm}"
  echo "================================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  TQDM_POSITION="${tqdm_position}" \
  TQDM_DESC="[GPU${gpu}]" \
  LOG_PREFIX="[GPU${gpu}]" \
  QUIET_CONSOLE_AFTER_TQDM="${live_tqdm}" \
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
}

run_one "flux_sweep_tfd1_anchor1_flow03_200k" "0.3" "${gpu0}" "0" &
pid0=$!
run_one "flux_sweep_tfd1_anchor1_flow01_200k" "0.1" "${gpu1}" "1" &
pid1=$!
run_one "flux_sweep_tfd1_anchor1_flow00_200k" "0.0" "${gpu2}" "2" &
pid2=$!

failed=0
if ! wait "${pid0}"; then
  echo "Flow ablation failed on GPU ${gpu0}: flow=0.3"
  failed=1
fi

if ! wait "${pid1}"; then
  echo "Flow ablation failed on GPU ${gpu1}: flow=0.1"
  failed=1
fi

if ! wait "${pid2}"; then
  echo "Flow ablation failed on GPU ${gpu2}: flow=0.0"
  failed=1
fi

if [[ "${failed}" != "0" ]]; then
  exit 1
fi

echo "FluxAudio flow ablation sweeps completed."
