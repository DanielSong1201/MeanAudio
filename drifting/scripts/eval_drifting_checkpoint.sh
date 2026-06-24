#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python drifting/test.py \
  --mode eval \
  --model-path "${MODEL_PATH:-exps/drifting/drifting_fluxaudio_s_1x4090/drifting_fluxaudio_s_1x4090_last.pth}" \
  --output "${OUTPUT_PATH:-exps/drifting_eval/drifting_fluxaudio_s_1x4090}" \
  --gt-cache "${GT_CACHE:-data/audiocaps/test-features}" \
  --num-steps "${NUM_STEPS:-1}" \
  --cfg-strength "${CFG_STRENGTH:-0.9}" \
  --use-rope
