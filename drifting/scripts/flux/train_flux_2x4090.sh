#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" drifting/flux/train.py \
  --exp-id "${EXP_ID:-flux_drifting_s_2x4090}" \
  --teacher-weights "${TEACHER_WEIGHTS:-weights/fluxaudio_s_full.pth}" \
  --student-init "${STUDENT_INIT:-weights/fluxaudio_s_full.pth}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --iterations "${ITERATIONS:-1000000}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --eval-interval "${EVAL_INTERVAL:-10000}" \
  --eval-output-root "${EVAL_OUTPUT_ROOT:-exps/drifting_flux_eval}" \
  --eval-gt-cache "${EVAL_GT_CACHE:-data/audiocaps/test-features}" \
  --eval-num-steps "${EVAL_NUM_STEPS:-1}" \
  --eval-cfg-strength "${EVAL_CFG_STRENGTH:-4.5}" \
  --ema-decay "${EMA_DECAY:-0.9999}" \
  --ema-start "${EMA_START:-0}" \
  --ema-update-interval "${EMA_UPDATE_INTERVAL:-1}" \
  --ema-device "${EMA_DEVICE:-cpu}" \
  --feature-layers "${FEATURE_LAYERS:-joint_3,fused_3,fused_7}" \
  --lambda-flow "${LAMBDA_FLOW:-1.0}" \
  --lambda-tfd "${LAMBDA_TFD:-0.2}" \
  --lambda-anchor "${LAMBDA_ANCHOR:-0.05}" \
  --feature-noise "${FEATURE_NOISE:-0.1}" \
  --pool-tokens "${POOL_TOKENS:-64}" \
  --use-rope \
  --amp
