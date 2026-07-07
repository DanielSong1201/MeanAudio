#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_VALUE="${PYTHON:-python}"
EXP_ID_VALUE="${EXP_ID:-resonate_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k}"
MODEL_PATH_VALUE="${MODEL_PATH:-weights/Resonate_GRPO.pth}"
TRAIN_OUTPUT_ROOT_VALUE="${TRAIN_OUTPUT_ROOT:-exps/drifting_resonate}"
EVAL_OUTPUT_ROOT_VALUE="${EVAL_OUTPUT_ROOT:-exps/drifting_resonate_eval}"
GT_CACHE_VALUE="${GT_CACHE:-data/audiocaps/test-features}"
GT_AUDIO_VALUE="${GT_AUDIO:-gt_audio}"
EVAL_TSV_VALUE="${EVAL_TSV:-data/audiocaps_resonate/test.tsv}"
EVAL_NPZ_DIR_VALUE="${EVAL_NPZ_DIR:-data/audiocaps_resonate/test-npz-flant5-44k}"
VAE_WEIGHTS_VALUE="${VAE_WEIGHTS:-weights/v1-44.pth}"
VOCODER_WEIGHTS_VALUE="${VOCODER_WEIGHTS:-weights/bigvgan_v2_44khz_128band_512x}"
FORCE_VALUE="${FORCE:-0}"
asset_download_gpu="${ASSET_DOWNLOAD_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"

PYTHON="${PYTHON_VALUE}" \
ASSET_DOWNLOAD_GPU="${asset_download_gpu}" \
TEACHER_WEIGHTS="${MODEL_PATH_VALUE}" \
STUDENT_INIT="${MODEL_PATH_VALUE}" \
EVAL_VAE_WEIGHTS="${VAE_WEIGHTS_VALUE}" \
EVAL_VOCODER_DIR="${VOCODER_WEIGHTS_VALUE}" \
WITH_EVAL=1 \
WITH_AV_BENCHMARK=1 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh

args=(
  drifting/Resonate/eval_baseline_it0.py
  --exp-id "${EXP_ID_VALUE}"
  --model-path "${MODEL_PATH_VALUE}"
  --train-output-root "${TRAIN_OUTPUT_ROOT_VALUE}"
  --eval-output-root "${EVAL_OUTPUT_ROOT_VALUE}"
  --gt-cache "${GT_CACHE_VALUE}"
  --gt-audio "${GT_AUDIO_VALUE}"
  --eval-tsv "${EVAL_TSV_VALUE}"
  --eval-npz-dir "${EVAL_NPZ_DIR_VALUE}"
  --vae-weights "${VAE_WEIGHTS_VALUE}"
  --vocoder-weights "${VOCODER_WEIGHTS_VALUE}"
  --num-steps 1
  --cfg-strength 4.5
)
if [[ "${FORCE_VALUE}" == "1" ]]; then
  args+=(--force)
fi

printf 'eval_it0 EXP_ID=%s\n' "${EXP_ID_VALUE}"
printf 'eval_it0 MODEL_PATH=%s\n' "${MODEL_PATH_VALUE}"
printf 'eval_it0 NUM_STEPS=1\n'
printf 'eval_it0 CFG_STRENGTH=4.5\n'
printf 'eval_it0 OUTPUT=%s/%s/it_00000000\n' \
  "${EVAL_OUTPUT_ROOT_VALUE}" "${EXP_ID_VALUE}"
printf 'eval_it0 METRICS_CSV=%s/%s/eval_metrics.csv\n' \
  "${TRAIN_OUTPUT_ROOT_VALUE}" "${EXP_ID_VALUE}"

exec "${PYTHON_VALUE}" "${args[@]}"
