#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

info() {
  printf '%s | INFO | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cmd=("${PYTHON:-python}" "${TEST_ENTRYPOINT:-drifting/flux/test.py}" \
  --mode eval \
  --model-path "${MODEL_PATH:-exps/drifting_flux/flux_drifting_s_1x4090/flux_drifting_s_1x4090_last.pth}" \
  --output "${OUTPUT_PATH:-exps/drifting_flux_eval/flux_drifting_s_1x4090}" \
  --gt-cache "${GT_CACHE:-data/audiocaps/test-features}" \
  --num-steps "${NUM_STEPS:-1}" \
  --cfg-strength "${CFG_STRENGTH:-4.5}")

if [[ "${USE_ROPE:-1}" == "1" ]]; then
  cmd+=(--use-rope)
fi

printf 'eval_config TEST_ENTRYPOINT=%s\n' "${TEST_ENTRYPOINT:-drifting/flux/test.py}"
printf 'eval_config MODEL_PATH=%s\n' "${MODEL_PATH:-exps/drifting_flux/flux_drifting_s_1x4090/flux_drifting_s_1x4090_last.pth}"
printf 'eval_config OUTPUT_PATH=%s\n' "${OUTPUT_PATH:-exps/drifting_flux_eval/flux_drifting_s_1x4090}"
printf 'eval_config GT_CACHE=%s\n' "${GT_CACHE:-data/audiocaps/test-features}"
printf 'eval_config NUM_STEPS=%s\n' "${NUM_STEPS:-1}"
printf 'eval_config CFG_STRENGTH=%s\n' "${CFG_STRENGTH:-4.5}"
printf 'eval_config USE_ROPE=%s\n' "${USE_ROPE:-1}"
printf 'eval_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"

info "Starting checkpoint evaluation"
info "Step 1/2: generate audio and save files under ${OUTPUT_PATH:-exps/drifting_flux_eval/flux_drifting_s_1x4090}/audio"
"${cmd[@]}"
info "Checkpoint evaluation finished"
