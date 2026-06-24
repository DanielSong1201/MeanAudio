#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-${WEIGHTS_DIR}/fluxaudio_s_full.pth}"
SAMPLE_NPZ="${SAMPLE_NPZ:-data/audiocaps/test-npz-t5-clap/0.npz}"
LAYERS="${LAYERS:-joint_3,fused_3,fused_7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

python tst/phase1_check_teacher_features.py \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --sample-npz "$SAMPLE_NPZ" \
  --layers "$LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --dtype "$DTYPE" \
  --use-rope \
  --text-c-dim 512
