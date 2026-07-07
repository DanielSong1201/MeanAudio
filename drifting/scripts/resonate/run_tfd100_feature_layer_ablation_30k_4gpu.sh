#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

scripts=(
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_current_30k_4gpu.sh"
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_j5_j15_f17_30k_4gpu.sh"
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_j5_j15_f8_30k_4gpu.sh"
  "drifting/scripts/resonate/train_tfd100_anchor1_flow01_layers_j3_j5_j15_f8_f17_30k_4gpu.sh"
)

failed=()
for script in "${scripts[@]}"; do
  echo "ablation_start ${script}"
  if ! bash "${script}"; then
    echo "ablation_failed ${script}" >&2
    failed+=("${script}")
  else
    echo "ablation_done ${script}"
  fi
done

if ((${#failed[@]})); then
  echo "Failed ablation runs:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "All feature-layer ablation runs completed."
