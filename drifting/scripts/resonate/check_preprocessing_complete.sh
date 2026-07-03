#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_VALUE="${PYTHON:-python}"
DATA_CONFIG_VALUE="${DATA_CONFIG:-config/data/resonate_flant5_44k.yaml}"
TEACHER_POSITIVE_DIR_VALUE="${TEACHER_POSITIVE_DIR:-data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5}"
TEACHER_POSITIVE_COUNT_VALUE="${TEACHER_POSITIVE_COUNT:-3}"
SAMPLE_COUNT_VALUE="${SAMPLE_COUNT:-16}"
DEEP_VALUE="${DEEP:-0}"

command=(
  "${PYTHON_VALUE}"
  drifting/Resonate/validate_preprocessing.py
  --data-config "${DATA_CONFIG_VALUE}"
  --teacher-positive-dir "${TEACHER_POSITIVE_DIR_VALUE}"
  --teacher-positive-count "${TEACHER_POSITIVE_COUNT_VALUE}"
  --sample-count "${SAMPLE_COUNT_VALUE}"
)
if [[ "${DEEP_VALUE}" == "1" ]]; then
  command+=(--deep)
fi

printf 'preprocess_check DATA_CONFIG=%s\n' "${DATA_CONFIG_VALUE}"
printf 'preprocess_check TEACHER_POSITIVE_DIR=%s\n' "${TEACHER_POSITIVE_DIR_VALUE}"
printf 'preprocess_check TEACHER_POSITIVE_COUNT=%s\n' "${TEACHER_POSITIVE_COUNT_VALUE}"
printf 'preprocess_check SAMPLE_COUNT=%s\n' "${SAMPLE_COUNT_VALUE}"
printf 'preprocess_check DEEP=%s\n' "${DEEP_VALUE}"

exec "${command[@]}"
