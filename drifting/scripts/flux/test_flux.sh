#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python drifting/flux/test.py \
  --mode "${MODE:-loss}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-float32}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --model-path "${MODEL_PATH:-exps/drifting_flux/flux_drifting_s_1x4090/flux_drifting_s_1x4090_last.pth}" \
  --output "${OUTPUT_PATH:-exps/drifting_flux_eval/manual}" \
  --gt-cache "${GT_CACHE:-data/audiocaps/test-features}" \
  --num-steps "${NUM_STEPS:-1}" \
  --cfg-strength "${CFG_STRENGTH:-4.5}" \
  --use-rope
