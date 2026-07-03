#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH_VALUE="${MODEL_PATH:-weights/Resonate_GRPO.pth}"
OUTPUT_PATH_VALUE="${OUTPUT_PATH:-exps/drifting_resonate_eval/manual}"
GT_CACHE_VALUE="${GT_CACHE:-data/audiocaps/test-features}"
EVAL_TSV_VALUE="${EVAL_TSV:-data/audiocaps_resonate/test.tsv}"
EVAL_NPZ_DIR_VALUE="${EVAL_NPZ_DIR:-data/audiocaps_resonate/test-npz-flant5-44k}"
VAE_WEIGHTS_VALUE="${VAE_WEIGHTS:-weights/v1-44.pth}"
VOCODER_WEIGHTS_VALUE="${VOCODER_WEIGHTS:-weights/bigvgan_v2_44khz_128band_512x}"
DURATION_VALUE="${DURATION:-10}"
SEED_VALUE="${SEED:-42}"
NUM_STEPS_VALUE="${NUM_STEPS:-1}"
CFG_STRENGTH_VALUE="${CFG_STRENGTH:-4.5}"
USE_ROPE_VALUE="${USE_ROPE:-1}"
EVAL_LIMIT_VALUE="${EVAL_LIMIT:-}"
EVAL_SKIP_AV_BENCHMARK_VALUE="${EVAL_SKIP_AV_BENCHMARK:-0}"

printf 'eval_config MODEL_PATH=%s\n' "${MODEL_PATH_VALUE}"
printf 'eval_config OUTPUT_PATH=%s\n' "${OUTPUT_PATH_VALUE}"
printf 'eval_config EVAL_TSV=%s\n' "${EVAL_TSV_VALUE}"
printf 'eval_config EVAL_NPZ_DIR=%s\n' "${EVAL_NPZ_DIR_VALUE}"
printf 'eval_config NUM_STEPS=%s\n' "${NUM_STEPS_VALUE}"
printf 'eval_config CFG_STRENGTH=%s\n' "${CFG_STRENGTH_VALUE}"
printf 'eval_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"

TEST_ENTRYPOINT=drifting/Resonate/test.py \
MODEL_PATH="${MODEL_PATH_VALUE}" \
OUTPUT_PATH="${OUTPUT_PATH_VALUE}" \
GT_CACHE="${GT_CACHE_VALUE}" \
EVAL_TSV="${EVAL_TSV_VALUE}" \
EVAL_NPZ_DIR="${EVAL_NPZ_DIR_VALUE}" \
VAE_WEIGHTS="${VAE_WEIGHTS_VALUE}" \
VOCODER_WEIGHTS="${VOCODER_WEIGHTS_VALUE}" \
DURATION="${DURATION_VALUE}" \
SEED="${SEED_VALUE}" \
NUM_STEPS="${NUM_STEPS_VALUE}" \
CFG_STRENGTH="${CFG_STRENGTH_VALUE}" \
USE_ROPE="${USE_ROPE_VALUE}" \
EVAL_LIMIT="${EVAL_LIMIT_VALUE}" \
EVAL_SKIP_AV_BENCHMARK="${EVAL_SKIP_AV_BENCHMARK_VALUE}" \
bash drifting/scripts/eval_drifting_checkpoint.sh
