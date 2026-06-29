#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

gpu0="${GPU0:-0}"
gpu1="${GPU1:-1}"
gpu2="${GPU2:-2}"
train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_1x4090.sh}"
log_root="${SWEEP_LAUNCH_LOG_ROOT:-exps/drifting_flux/sweep_launch_logs}"
live_tqdm="${LIVE_TQDM:-1}"

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
echo "LIVE_TQDM=${live_tqdm}"
if [[ "${live_tqdm}" == "1" ]]; then
  echo "Child output streams directly to this terminal so three tqdm bars can use fixed positions."
  echo "Set LIVE_TQDM=0 to prefix child logs with [GPUx] and tee them to launch logs."
else
  echo "Child logs will be prefixed with [GPUx] and tee'd to launch logs."
fi
echo "================================================================"

if [[ "${live_tqdm}" == "1" ]]; then
  (
    CUDA_VISIBLE_DEVICES="${gpu0}" \
    TRAIN_SCRIPT="${train_script}" \
    TQDM_POSITION=0 \
    TQDM_DESC="[GPU${gpu0}]" \
    LOG_PREFIX="[GPU${gpu0}]" \
    QUIET_CONSOLE_AFTER_TQDM=1 \
    EVAL_TQDM_POSITION_OFFSET=3 \
    EXP_ID= \
    bash "${sweep0}"
  ) &
  pid0=$!

  (
    CUDA_VISIBLE_DEVICES="${gpu1}" \
    TRAIN_SCRIPT="${train_script}" \
    TQDM_POSITION=1 \
    TQDM_DESC="[GPU${gpu1}]" \
    LOG_PREFIX="[GPU${gpu1}]" \
    QUIET_CONSOLE_AFTER_TQDM=1 \
    EVAL_TQDM_POSITION_OFFSET=3 \
    EXP_ID= \
    bash "${sweep1}"
  ) &
  pid1=$!

  (
    CUDA_VISIBLE_DEVICES="${gpu2}" \
    TRAIN_SCRIPT="${train_script}" \
    TQDM_POSITION=2 \
    TQDM_DESC="[GPU${gpu2}]" \
    LOG_PREFIX="[GPU${gpu2}]" \
    QUIET_CONSOLE_AFTER_TQDM=1 \
    EVAL_TQDM_POSITION_OFFSET=3 \
    EXP_ID= \
    bash "${sweep2}"
  ) &
  pid2=$!
else
  (
    CUDA_VISIBLE_DEVICES="${gpu0}" \
    TRAIN_SCRIPT="${train_script}" \
    EXP_ID= \
    bash "${sweep0}"
  ) > >(awk -v prefix="[GPU${gpu0}] " '{ print prefix $0; fflush() }' | tee "${log0}") 2>&1 &
  pid0=$!

  (
    CUDA_VISIBLE_DEVICES="${gpu1}" \
    TRAIN_SCRIPT="${train_script}" \
    EXP_ID= \
    bash "${sweep1}"
  ) > >(awk -v prefix="[GPU${gpu1}] " '{ print prefix $0; fflush() }' | tee "${log1}") 2>&1 &
  pid1=$!

  (
    CUDA_VISIBLE_DEVICES="${gpu2}" \
    TRAIN_SCRIPT="${train_script}" \
    EXP_ID= \
    bash "${sweep2}"
  ) > >(awk -v prefix="[GPU${gpu2}] " '{ print prefix $0; fflush() }' | tee "${log2}") 2>&1 &
  pid2=$!
fi

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
