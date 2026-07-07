#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# User-tunable launch settings. Override them from the shell, for example:
#   BATCH_SIZE=2 GPU0=2 GPU1=3 bash drifting/scripts/resonate/parallel_tfd100_feature_layer_ablation_30k_2x1gpu.sh
GPU0_VALUE="${GPU0:-0}"
GPU1_VALUE="${GPU1:-1}"
SAVE_ITS_VALUE="${SAVE_ITS:-${SAVE_INTERVAL:-2000}}"
EMA_DEVICE_VALUE="${EMA_DEVICE:-cuda}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-1}"
QUIET_CONSOLE_AFTER_TQDM_VALUE="${QUIET_CONSOLE_AFTER_TQDM:-1}"
EVAL_TQDM_POSITION_OFFSET_VALUE="${EVAL_TQDM_POSITION_OFFSET:-2}"

export SAVE_ITS="${SAVE_ITS_VALUE}"
export EMA_DEVICE="${EMA_DEVICE_VALUE}"
export BATCH_SIZE="${BATCH_SIZE_VALUE}"
export NPROC_PER_NODE=1
export QUIET_CONSOLE_AFTER_TQDM="${QUIET_CONSOLE_AFTER_TQDM_VALUE}"
export EVAL_TQDM_POSITION_OFFSET="${EVAL_TQDM_POSITION_OFFSET_VALUE}"

echo "parallel_ablation_config GPU0=${GPU0_VALUE}"
echo "parallel_ablation_config GPU1=${GPU1_VALUE}"
echo "parallel_ablation_config SAVE_ITS=${SAVE_ITS_VALUE}"
echo "parallel_ablation_config EMA_DEVICE=${EMA_DEVICE_VALUE}"
echo "parallel_ablation_config BATCH_SIZE=${BATCH_SIZE_VALUE}"
echo "parallel_ablation_config NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "parallel_ablation_config QUIET_CONSOLE_AFTER_TQDM=${QUIET_CONSOLE_AFTER_TQDM_VALUE}"
echo "parallel_ablation_config EVAL_TQDM_POSITION_OFFSET=${EVAL_TQDM_POSITION_OFFSET_VALUE}"

run_one() {
  local label="$1"
  local gpu="$2"
  local tqdm_position="$3"
  local script="$4"
  echo "${label}_start gpu=${gpu} tqdm_position=${tqdm_position} script=${script}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  BATCH_SIZE="${BATCH_SIZE_VALUE}" \
  EMA_DEVICE="${EMA_DEVICE_VALUE}" \
  SAVE_ITS="${SAVE_ITS_VALUE}" \
  TQDM_POSITION="${tqdm_position}" \
  TQDM_DESC="[GPU${gpu}] ${label}" \
  LOG_PREFIX="[GPU${gpu}]" \
  QUIET_CONSOLE_AFTER_TQDM="${QUIET_CONSOLE_AFTER_TQDM_VALUE}" \
  EVAL_TQDM_POSITION_OFFSET="${EVAL_TQDM_POSITION_OFFSET_VALUE}" \
  bash "${script}"
  echo "${label}_done gpu=${gpu} script=${script}"
}

run_sequence() {
  local worker="$1"
  local gpu="$2"
  local tqdm_position="$3"
  shift 3
  local index=1
  for script in "$@"; do
    run_one "${worker}_task${index}" "${gpu}" "${tqdm_position}" "${script}"
    index=$((index + 1))
  done
}

gpu0_scripts=(
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_current_30k_4gpu.sh"
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_j5_j15_f8_30k_4gpu.sh"
)
gpu1_scripts=(
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_j5_j15_f17_30k_4gpu.sh"
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_j3_j5_j15_f8_f17_30k_4gpu.sh"
)

run_sequence "gpu0" "${GPU0_VALUE}" 0 "${gpu0_scripts[@]}" &
pid_gpu0=$!
run_sequence "gpu1" "${GPU1_VALUE}" 1 "${gpu1_scripts[@]}" &
pid_gpu1=$!

failed=()
if ! wait "${pid_gpu0}"; then
  failed+=("gpu0")
fi
if ! wait "${pid_gpu1}"; then
  failed+=("gpu1")
fi

if ((${#failed[@]})); then
  echo "Failed workers:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "All two-GPU single-process feature-layer ablation runs completed."
