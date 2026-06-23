#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_STEPS="${NUM_STEPS:-25}"
WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"
CKPT_PATH="${TEACHER_WEIGHTS:-${WEIGHTS_DIR}/fluxaudio_s_full.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-./exps/phase0_fluxaudio_s_full/test_${NUM_STEPS}nfe_fp32}"
GT_CACHE="${GT_CACHE:-./data/audiocaps/test-features}"
CFG_STRENGTH="${CFG_STRENGTH:-4.5}"

python eval.py \
  --variant "fluxaudio_s" \
  --model_path "$CKPT_PATH" \
  --output "$OUTPUT_PATH/audio" \
  --cfg_strength "$CFG_STRENGTH" \
  --encoder_name t5_clap \
  --duration 10 \
  --use_rope \
  --text_c_dim 512 \
  --num_steps "$NUM_STEPS" \
  --full_precision

python av-benchmark/evaluate.py \
  --gt_audio gt_audio \
  --gt_cache "$GT_CACHE" \
  --pred_audio "$OUTPUT_PATH/audio" \
  --pred_cache "$OUTPUT_PATH/cache" \
  --audio_length=10 \
  --recompute_pred_cache \
  --skip_video_related \
  --output_metrics_dir="$OUTPUT_PATH"
