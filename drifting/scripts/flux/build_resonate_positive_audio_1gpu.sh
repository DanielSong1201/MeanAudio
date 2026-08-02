#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. On the server, Resonate is expected beside MeanAudio:
#
#   workspace/
#     MeanAudio/
#     Resonate/

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  printf 'ERROR: this is a single-GPU launcher; CUDA_VISIBLE_DEVICES must contain one GPU id.\n' >&2
  exit 1
fi
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: invalid CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_VALUE="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_VALUE="python"
else
  PYTHON_VALUE="python3"
fi
RESONATE_ROOT_VALUE="${RESONATE_ROOT:-../Resonate}"
MANIFEST_VALUE="${MANIFEST:-data/audiocaps/train-memmap.tsv}"
OUTPUT_DIR_VALUE="${OUTPUT_DIR:-data/audiocaps/train-teacher-audio-resonate-grpo-25step-cfg4.5}"
CHECKPOINT_VALUE="${CHECKPOINT:-${RESONATE_ROOT_VALUE}/weights/Resonate_GRPO.pth}"
CONFIG_NAME_VALUE="${CONFIG_NAME:-GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic}"

POSITIVES_PER_CONDITION_VALUE="${POSITIVES_PER_CONDITION:-3}"
PROMPT_BATCH_SIZE_VALUE="${PROMPT_BATCH_SIZE:-1}"
NUM_STEPS_VALUE="${NUM_STEPS:-25}"
CFG_STRENGTH_VALUE="${CFG_STRENGTH:-4.5}"
DURATION_VALUE="${DURATION:-10}"
NEGATIVE_PROMPT_VALUE="${NEGATIVE_PROMPT:-}"
BASE_SEED_VALUE="${BASE_SEED:-20260802}"
START_INDEX_VALUE="${START_INDEX:-0}"
END_INDEX_VALUE="${END_INDEX:-}"
LIMIT_VALUE="${LIMIT:-}"
AUTO_DOWNLOAD_VALUE="${AUTO_DOWNLOAD:-1}"
OVERWRITE_VALUE="${OVERWRITE:-0}"
FULL_PRECISION_VALUE="${FULL_PRECISION:-0}"
DRY_RUN_VALUE="${DRY_RUN:-0}"

extra_args=()
if [[ -n "${END_INDEX_VALUE}" ]]; then
  extra_args+=(--end-index "${END_INDEX_VALUE}")
fi
if [[ -n "${LIMIT_VALUE}" ]]; then
  extra_args+=(--limit "${LIMIT_VALUE}")
fi
if [[ "${AUTO_DOWNLOAD_VALUE}" == "1" ]]; then
  extra_args+=(--download-if-missing)
elif [[ "${AUTO_DOWNLOAD_VALUE}" != "0" ]]; then
  printf 'ERROR: AUTO_DOWNLOAD must be 0 or 1.\n' >&2
  exit 1
fi
if [[ "${OVERWRITE_VALUE}" == "1" ]]; then
  extra_args+=(--overwrite)
elif [[ "${OVERWRITE_VALUE}" != "0" ]]; then
  printf 'ERROR: OVERWRITE must be 0 or 1.\n' >&2
  exit 1
fi
if [[ "${FULL_PRECISION_VALUE}" == "1" ]]; then
  extra_args+=(--full-precision)
elif [[ "${FULL_PRECISION_VALUE}" != "0" ]]; then
  printf 'ERROR: FULL_PRECISION must be 0 or 1.\n' >&2
  exit 1
fi
if [[ "${DRY_RUN_VALUE}" == "1" ]]; then
  extra_args+=(--dry-run)
elif [[ "${DRY_RUN_VALUE}" != "0" ]]; then
  printf 'ERROR: DRY_RUN must be 0 or 1.\n' >&2
  exit 1
fi

printf 'positive_audio_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'positive_audio_config RESONATE_ROOT=%s\n' "${RESONATE_ROOT_VALUE}"
printf 'positive_audio_config MANIFEST=%s\n' "${MANIFEST_VALUE}"
printf 'positive_audio_config OUTPUT_DIR=%s\n' "${OUTPUT_DIR_VALUE}"
printf 'positive_audio_config CHECKPOINT=%s\n' "${CHECKPOINT_VALUE}"
printf 'positive_audio_config POSITIVES_PER_CONDITION=%s\n' "${POSITIVES_PER_CONDITION_VALUE}"
printf 'positive_audio_config PROMPT_BATCH_SIZE=%s\n' "${PROMPT_BATCH_SIZE_VALUE}"
printf 'positive_audio_config NUM_STEPS=%s\n' "${NUM_STEPS_VALUE}"
printf 'positive_audio_config CFG_STRENGTH=%s\n' "${CFG_STRENGTH_VALUE}"
printf 'positive_audio_config DURATION=%s\n' "${DURATION_VALUE}"
printf 'positive_audio_config BASE_SEED=%s\n' "${BASE_SEED_VALUE}"
printf 'positive_audio_config START_INDEX=%s\n' "${START_INDEX_VALUE}"
printf 'positive_audio_config END_INDEX=%s\n' "${END_INDEX_VALUE:-all}"
printf 'positive_audio_config LIMIT=%s\n' "${LIMIT_VALUE:-all}"
printf 'positive_audio_config AUTO_DOWNLOAD=%s\n' "${AUTO_DOWNLOAD_VALUE}"
printf 'positive_audio_config OVERWRITE=%s\n' "${OVERWRITE_VALUE}"

exec "${PYTHON_VALUE}" drifting/flux/build_resonate_positive_audio.py \
  --resonate-root "${RESONATE_ROOT_VALUE}" \
  --manifest "${MANIFEST_VALUE}" \
  --output-dir "${OUTPUT_DIR_VALUE}" \
  --checkpoint "${CHECKPOINT_VALUE}" \
  --config-name "${CONFIG_NAME_VALUE}" \
  --positives-per-condition "${POSITIVES_PER_CONDITION_VALUE}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE_VALUE}" \
  --num-steps "${NUM_STEPS_VALUE}" \
  --cfg-strength "${CFG_STRENGTH_VALUE}" \
  --duration "${DURATION_VALUE}" \
  --negative-prompt "${NEGATIVE_PROMPT_VALUE}" \
  --base-seed "${BASE_SEED_VALUE}" \
  --start-index "${START_INDEX_VALUE}" \
  "${extra_args[@]}"
