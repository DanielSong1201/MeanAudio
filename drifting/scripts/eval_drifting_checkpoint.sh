#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cmd=("${PYTHON:-python}" "${TEST_ENTRYPOINT:-drifting/test.py}" \
  --mode eval \
  --model-path "${MODEL_PATH:-exps/drifting/drifting_fluxaudio_s_1x4090/drifting_fluxaudio_s_1x4090_last.pth}" \
  --output "${OUTPUT_PATH:-exps/drifting_eval/drifting_fluxaudio_s_1x4090}" \
  --gt-cache "${GT_CACHE:-data/audiocaps/test-features}" \
  --num-steps "${NUM_STEPS:-1}" \
  --cfg-strength "${CFG_STRENGTH:-0.9}")

if [[ "${USE_ROPE:-1}" == "1" ]]; then
  cmd+=(--use-rope)
fi

printf 'eval_config TEST_ENTRYPOINT=%s\n' "${TEST_ENTRYPOINT:-drifting/test.py}"
printf 'eval_config MODEL_PATH=%s\n' "${MODEL_PATH:-exps/drifting/drifting_fluxaudio_s_1x4090/drifting_fluxaudio_s_1x4090_last.pth}"
printf 'eval_config OUTPUT_PATH=%s\n' "${OUTPUT_PATH:-exps/drifting_eval/drifting_fluxaudio_s_1x4090}"
printf 'eval_config GT_CACHE=%s\n' "${GT_CACHE:-data/audiocaps/test-features}"
printf 'eval_config NUM_STEPS=%s\n' "${NUM_STEPS:-1}"
printf 'eval_config CFG_STRENGTH=%s\n' "${CFG_STRENGTH:-0.9}"
printf 'eval_config USE_ROPE=%s\n' "${USE_ROPE:-1}"
printf 'eval_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"

"${cmd[@]}"
