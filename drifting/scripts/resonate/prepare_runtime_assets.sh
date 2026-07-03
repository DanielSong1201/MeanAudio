#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_VALUE="${PYTHON:-python}"
ASSET_DOWNLOAD_GPU_VALUE="${ASSET_DOWNLOAD_GPU:-0}"
TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/Resonate_GRPO.pth}"
STUDENT_INIT_VALUE="${STUDENT_INIT:-weights/Resonate_GRPO.pth}"
LATENT_MEAN_VALUE="${LATENT_MEAN:-sets/latent_mean_44k.pt}"
LATENT_STD_VALUE="${LATENT_STD:-sets/latent_std_44k.pt}"
EVAL_VAE_WEIGHTS_VALUE="${EVAL_VAE_WEIGHTS:-weights/v1-44.pth}"
EVAL_VOCODER_DIR_VALUE="${EVAL_VOCODER_DIR:-weights/bigvgan_v2_44khz_128band_512x}"
AV_BENCHMARK_DIR_VALUE="${AV_BENCHMARK_DIR:-av-benchmark}"
ASSET_STATE_DIR_VALUE="${ASSET_STATE_DIR:-weights/.resonate-runtime-assets}"
WITH_EVAL_VALUE="${WITH_EVAL:-1}"
WITH_AV_BENCHMARK_VALUE="${WITH_AV_BENCHMARK:-1}"

if [[ ! "${ASSET_DOWNLOAD_GPU_VALUE}" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: ASSET_DOWNLOAD_GPU must be one GPU id, got %q\n' \
    "${ASSET_DOWNLOAD_GPU_VALUE}" >&2
  exit 1
fi

args=(
  drifting/Resonate/prepare_runtime_assets.py
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}"
  --student-init "${STUDENT_INIT_VALUE}"
  --latent-mean "${LATENT_MEAN_VALUE}"
  --latent-std "${LATENT_STD_VALUE}"
  --vae-weights "${EVAL_VAE_WEIGHTS_VALUE}"
  --vocoder-dir "${EVAL_VOCODER_DIR_VALUE}"
  --av-benchmark-dir "${AV_BENCHMARK_DIR_VALUE}"
  --state-dir "${ASSET_STATE_DIR_VALUE}"
)
if [[ "${WITH_EVAL_VALUE}" == "1" ]]; then
  args+=(--with-eval)
fi
if [[ "${WITH_AV_BENCHMARK_VALUE}" == "1" ]]; then
  args+=(--with-av-benchmark)
fi

printf '%s | INFO | Asset bootstrap is standalone and exposes only GPU %s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${ASSET_DOWNLOAD_GPU_VALUE}"
CUDA_VISIBLE_DEVICES="${ASSET_DOWNLOAD_GPU_VALUE}" \
HF_HUB_DISABLE_PROGRESS_BARS=0 \
"${PYTHON_VALUE}" "${args[@]}"
