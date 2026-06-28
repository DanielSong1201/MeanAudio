#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

sweeps=(
  "drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor05_200k.sh"
  "drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor1_flow03_200k.sh"
  "drifting/scripts/flux/sweeps/sweep_flux_lr1e5_tfd1_anchor1_flow05_200k.sh"
)

failed=()
for sweep in "${sweeps[@]}"; do
  echo "================================================================"
  echo "Running ${sweep} with two-GPU FluxAudio training"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}"
  echo "================================================================"
  if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    if ! TRAIN_SCRIPT="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}" EXP_ID= bash "${sweep}"; then
      failed+=("${sweep}")
    fi
  else
    TRAIN_SCRIPT="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}" EXP_ID= bash "${sweep}"
  fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed sweeps:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi

echo "All FluxAudio 200K sweeps completed."
