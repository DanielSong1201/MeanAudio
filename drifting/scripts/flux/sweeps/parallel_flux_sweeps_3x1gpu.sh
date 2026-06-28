#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

gpu0="${GPU0:-0}"
gpu1="${GPU1:-1}"
gpu2="${GPU2:-2}"
train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_1x4090.sh}"
log_root="${SWEEP_LAUNCH_LOG_ROOT:-exps/drifting_flux/sweep_launch_logs}"

sweep0="drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor05_200k.sh"
sweep1="drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor1_flow03_200k.sh"
sweep2="drifting/scripts/flux/sweeps/sweep_flux_lr1e5_tfd1_anchor1_flow05_200k.sh"

mkdir -p "${log_root}"

log0="${log_root}/gpu${gpu0}_$(basename "${sweep0}" .sh).log"
log1="${log_root}/gpu${gpu1}_$(basename "${sweep1}" .sh).log"
log2="${log_root}/gpu${gpu2}_$(basename "${sweep2}" .sh).log"

echo "================================================================"
echo "Running FluxAudio sweeps as three independent 1-GPU jobs"
echo "GPU ${gpu0}: ${sweep0}"
echo "GPU ${gpu1}: ${sweep1}"
echo "GPU ${gpu2}: ${sweep2}"
echo "TRAIN_SCRIPT=${train_script}"
echo "LOG ${gpu0}: ${log0}"
echo "LOG ${gpu1}: ${log1}"
echo "LOG ${gpu2}: ${log2}"
echo "================================================================"

(
  CUDA_VISIBLE_DEVICES="${gpu0}" \
  TRAIN_SCRIPT="${train_script}" \
  EXP_ID= \
  bash "${sweep0}"
) >"${log0}" 2>&1 &
pid0=$!

(
  CUDA_VISIBLE_DEVICES="${gpu1}" \
  TRAIN_SCRIPT="${train_script}" \
  EXP_ID= \
  bash "${sweep1}"
) >"${log1}" 2>&1 &
pid1=$!

(
  CUDA_VISIBLE_DEVICES="${gpu2}" \
  TRAIN_SCRIPT="${train_script}" \
  EXP_ID= \
  bash "${sweep2}"
) >"${log2}" 2>&1 &
pid2=$!

failed=0
if ! wait "${pid0}"; then
  echo "Sweep failed on GPU ${gpu0}: ${sweep0}; see ${log0}"
  failed=1
fi

if ! wait "${pid1}"; then
  echo "Sweep failed on GPU ${gpu1}: ${sweep1}; see ${log1}"
  failed=1
fi

if ! wait "${pid2}"; then
  echo "Sweep failed on GPU ${gpu2}: ${sweep2}; see ${log2}"
  failed=1
fi

if [[ "${failed}" != "0" ]]; then
  exit 1
fi

echo "All three 1-GPU FluxAudio sweeps completed."
