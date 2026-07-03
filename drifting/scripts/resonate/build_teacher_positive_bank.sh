#!/usr/bin/env bash
set -euo pipefail

# A fatal CUDA worker should not write a multi-gigabyte core.<pid> file.
ulimit -c 0

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export POSITIVE_RUN_ID="${POSITIVE_RUN_ID:-$(date '+%Y%m%d_%H%M%S')_$$}"

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
detected_gpu_count="${#visible_devices[@]}"
if (( detected_gpu_count < 1 )); then
  printf 'ERROR: CUDA_VISIBLE_DEVICES must contain at least one GPU id.\n' >&2
  exit 1
fi
for device_id in "${visible_devices[@]}"; do
  if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: invalid CUDA device id %q in CUDA_VISIBLE_DEVICES=%q.\n' \
      "${device_id}" "${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
done

NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-${detected_gpu_count}}"
if [[ ! "${NPROC_PER_NODE_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: NPROC_PER_NODE must be a positive integer, got %q.\n' \
    "${NPROC_PER_NODE_VALUE}" >&2
  exit 1
fi
if (( NPROC_PER_NODE_VALUE != detected_gpu_count )); then
  printf 'ERROR: NPROC_PER_NODE=%s but CUDA_VISIBLE_DEVICES exposes %s GPU(s): %s\n' \
    "${NPROC_PER_NODE_VALUE}" "${detected_gpu_count}" "${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

PYTHON_VALUE="${PYTHON:-python}"
TSV_VALUE="${TSV:-data/audiocaps_resonate/train.tsv}"
NPZ_DIR_VALUE="${NPZ_DIR:-data/audiocaps_resonate/train-npz-flant5-44k}"
OUTPUT_DIR_VALUE="${OUTPUT_DIR:-data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5}"
TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/Resonate_GRPO.pth}"
POSITIVES_PER_CONDITION_VALUE="${POSITIVES_PER_CONDITION:-3}"
NUM_STEPS_VALUE="${NUM_STEPS:-25}"
CFG_STRENGTH_VALUE="${CFG_STRENGTH:-4.5}"
SEED_VALUE="${SEED:-20260702}"
STORAGE_DTYPE_VALUE="${STORAGE_DTYPE:-float16}"
LOG_LEVEL_VALUE="${LOG_LEVEL:-INFO}"
AMP_VALUE="${AMP:-1}"
OVERWRITE_VALUE="${OVERWRITE:-0}"
ALLOW_INCOMPLETE_DATA_VALUE="${ALLOW_INCOMPLETE_DATA:-0}"
LIMIT_VALUE="${LIMIT:-}"
DRY_RUN_VALUE="${DRY_RUN:-0}"
COMPLETION_POLL_SECONDS_VALUE="${COMPLETION_POLL_SECONDS:-30}"
COMPLETION_TIMEOUT_MINUTES_VALUE="${COMPLETION_TIMEOUT_MINUTES:-0}"
SHOW_ALL_GPU_PROGRESS_VALUE="${SHOW_ALL_GPU_PROGRESS:-1}"
LOCK_PATH="${OUTPUT_DIR_VALUE}/.positive-bank.lock"

positive_args=(
  drifting/Resonate/build_teacher_positive_bank.py
  --tsv "${TSV_VALUE}"
  --npz-dir "${NPZ_DIR_VALUE}"
  --output-dir "${OUTPUT_DIR_VALUE}"
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}"
  --positives-per-condition "${POSITIVES_PER_CONDITION_VALUE}"
  --num-steps "${NUM_STEPS_VALUE}"
  --cfg-strength "${CFG_STRENGTH_VALUE}"
  --seed "${SEED_VALUE}"
  --storage-dtype "${STORAGE_DTYPE_VALUE}"
  --log-level "${LOG_LEVEL_VALUE}"
  --completion-poll-seconds "${COMPLETION_POLL_SECONDS_VALUE}"
  --completion-timeout-minutes "${COMPLETION_TIMEOUT_MINUTES_VALUE}"
)
if [[ "${AMP_VALUE}" == "0" ]]; then
  positive_args+=(--no-amp)
fi
if [[ "${OVERWRITE_VALUE}" == "1" ]]; then
  positive_args+=(--overwrite)
fi
if [[ "${ALLOW_INCOMPLETE_DATA_VALUE}" == "1" ]]; then
  positive_args+=(--allow-incomplete-data)
fi
if [[ -n "${LIMIT_VALUE}" ]]; then
  positive_args+=(--limit "${LIMIT_VALUE}")
fi
if [[ "${SHOW_ALL_GPU_PROGRESS_VALUE}" == "0" ]]; then
  positive_args+=(--no-progress-all-ranks)
fi

printf 'positive_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'positive_config GPU_COUNT=%s\n' "${NPROC_PER_NODE_VALUE}"
printf 'positive_config PROCESS_GROUP=disabled\n'
printf 'positive_config WORKER_SYNC=file-markers\n'
printf 'positive_config POSITIVE_RUN_ID=%s\n' "${POSITIVE_RUN_ID}"
printf 'positive_config CORE_DUMP_LIMIT=%s\n' "$(ulimit -c)"
printf 'positive_config TSV=%s\n' "${TSV_VALUE}"
printf 'positive_config NPZ_DIR=%s\n' "${NPZ_DIR_VALUE}"
printf 'positive_config OUTPUT_DIR=%s\n' "${OUTPUT_DIR_VALUE}"
printf 'positive_config TEACHER_WEIGHTS=%s\n' "${TEACHER_WEIGHTS_VALUE}"
printf 'positive_config POSITIVES_PER_CONDITION=%s\n' "${POSITIVES_PER_CONDITION_VALUE}"
printf 'positive_config NUM_STEPS=%s\n' "${NUM_STEPS_VALUE}"
printf 'positive_config CFG_STRENGTH=%s\n' "${CFG_STRENGTH_VALUE}"
printf 'positive_config SEED=%s\n' "${SEED_VALUE}"
printf 'positive_config AMP=%s\n' "${AMP_VALUE}"
printf 'positive_config OVERWRITE=%s\n' "${OVERWRITE_VALUE}"
printf 'positive_config LIMIT=%s\n' "${LIMIT_VALUE:-disabled}"
printf 'positive_config COMPLETION_POLL_SECONDS=%s\n' "${COMPLETION_POLL_SECONDS_VALUE}"
printf 'positive_config COMPLETION_TIMEOUT_MINUTES=%s\n' "${COMPLETION_TIMEOUT_MINUTES_VALUE}"
printf 'positive_config SHOW_ALL_GPU_PROGRESS=%s\n' "${SHOW_ALL_GPU_PROGRESS_VALUE}"
printf 'positive_config LOCK=%s\n' "${LOCK_PATH}"

if (( NPROC_PER_NODE_VALUE == 1 )); then
  command=("${PYTHON_VALUE}" "${positive_args[@]}")
else
  command=(
    torchrun
    --standalone
    --nproc_per_node="${NPROC_PER_NODE_VALUE}"
    "${positive_args[@]}"
  )
fi

if [[ "${DRY_RUN_VALUE}" == "1" ]]; then
  printf 'positive_command'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
  printf 'ERROR: flock is required to prevent duplicate positive-bank jobs.\n' >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR_VALUE}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  printf 'ERROR: another positive-bank job owns lock %s.\n' "${LOCK_PATH}" >&2
  exit 1
fi

exec "${command[@]}"
