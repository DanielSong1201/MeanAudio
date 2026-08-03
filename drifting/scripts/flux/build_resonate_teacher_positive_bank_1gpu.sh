#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. On the server, Resonate must be beside MeanAudio:
#
#   workspace/
#     MeanAudio/
#     Resonate/

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: single GPU required; invalid CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}" >&2
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
OUTPUT_DIR_VALUE="${OUTPUT_DIR:-${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-resonate-grpo-25step-cfg4.5}}"
CHECKPOINT_VALUE="${CHECKPOINT:-${RESONATE_ROOT_VALUE}/weights/Resonate_GRPO.pth}"
CONFIG_NAME_VALUE="${CONFIG_NAME:-GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic}"
TARGET_VAE_WEIGHTS_VALUE="${TARGET_VAE_WEIGHTS:-weights/v1-16.pth}"
TARGET_LATENT_MEAN_VALUE="${TARGET_LATENT_MEAN:-sets/latent_mean.pt}"
TARGET_LATENT_STD_VALUE="${TARGET_LATENT_STD:-sets/latent_std.pt}"

POSITIVES_PER_CONDITION_VALUE="${POSITIVES_PER_CONDITION:-3}"
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
for pair in \
  "AUTO_DOWNLOAD:${AUTO_DOWNLOAD_VALUE}:--download-if-missing" \
  "OVERWRITE:${OVERWRITE_VALUE}:--overwrite" \
  "FULL_PRECISION:${FULL_PRECISION_VALUE}:--full-precision" \
  "DRY_RUN:${DRY_RUN_VALUE}:--dry-run"; do
  name="${pair%%:*}"
  rest="${pair#*:}"
  value="${rest%%:*}"
  flag="${rest#*:}"
  if [[ "${value}" == "1" ]]; then
    extra_args+=("${flag}")
  elif [[ "${value}" != "0" ]]; then
    printf 'ERROR: %s must be 0 or 1.\n' "${name}" >&2
    exit 1
  fi
done

printf 'resonate_teacher_bank CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'resonate_teacher_bank RESONATE_ROOT=%s\n' "${RESONATE_ROOT_VALUE}"
printf 'resonate_teacher_bank MANIFEST=%s\n' "${MANIFEST_VALUE}"
printf 'resonate_teacher_bank OUTPUT_DIR=%s\n' "${OUTPUT_DIR_VALUE}"
printf 'resonate_teacher_bank POSITIVES_PER_CONDITION=%s\n' "${POSITIVES_PER_CONDITION_VALUE}"
printf 'resonate_teacher_bank NUM_STEPS=%s\n' "${NUM_STEPS_VALUE}"
printf 'resonate_teacher_bank CFG_STRENGTH=%s\n' "${CFG_STRENGTH_VALUE}"
printf 'resonate_teacher_bank START_INDEX=%s END_INDEX=%s LIMIT=%s\n' \
  "${START_INDEX_VALUE}" "${END_INDEX_VALUE:-all}" "${LIMIT_VALUE:-all}"
printf 'resonate_teacher_bank AUTO_DOWNLOAD=%s OVERWRITE=%s\n' \
  "${AUTO_DOWNLOAD_VALUE}" "${OVERWRITE_VALUE}"

exec "${PYTHON_VALUE}" -m drifting.flux.build_resonate_teacher_positive_bank \
  --resonate-root "${RESONATE_ROOT_VALUE}" \
  --manifest "${MANIFEST_VALUE}" \
  --output-dir "${OUTPUT_DIR_VALUE}" \
  --checkpoint "${CHECKPOINT_VALUE}" \
  --config-name "${CONFIG_NAME_VALUE}" \
  --target-vae-weights "${TARGET_VAE_WEIGHTS_VALUE}" \
  --target-latent-mean "${TARGET_LATENT_MEAN_VALUE}" \
  --target-latent-std "${TARGET_LATENT_STD_VALUE}" \
  --positives-per-condition "${POSITIVES_PER_CONDITION_VALUE}" \
  --num-steps "${NUM_STEPS_VALUE}" \
  --cfg-strength "${CFG_STRENGTH_VALUE}" \
  --duration "${DURATION_VALUE}" \
  --negative-prompt "${NEGATIVE_PROMPT_VALUE}" \
  --base-seed "${BASE_SEED_VALUE}" \
  --start-index "${START_INDEX_VALUE}" \
  "${extra_args[@]}"
