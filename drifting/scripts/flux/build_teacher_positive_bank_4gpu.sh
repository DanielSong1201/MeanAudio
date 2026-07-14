#!/usr/bin/env bash
set -euo pipefail

# Prevent a fatal CUDA/NCCL worker from writing a large core.<pid> file.
ulimit -c 0

cd "$(dirname "$0")/../../.."

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-4}"
OUTPUT_DIR_VALUE="${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6}"
TEACHER_WEIGHTS_VALUE="${TEACHER_POSITIVE_WEIGHTS:-weights/meanaudio_l_full.pth}"
TEACHER_VARIANT_VALUE="${TEACHER_POSITIVE_VARIANT:-meanaudio_l}"
POSITIVES_PER_CONDITION_VALUE="${TEACHER_POSITIVE_COUNT:-3}"
NUM_STEPS_VALUE="${TEACHER_POSITIVE_NUM_STEPS:-25}"
CFG_STRENGTH_VALUE="${TEACHER_POSITIVE_CFG_STRENGTH:-6.0}"
SEED_VALUE="${TEACHER_POSITIVE_SEED:-20260507}"
OVERWRITE_VALUE="${TEACHER_POSITIVE_OVERWRITE:-0}"

overwrite_args=()
if [[ "${OVERWRITE_VALUE}" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

printf 'teacher_positive_config OUTPUT_DIR=%s\n' "${OUTPUT_DIR_VALUE}"
printf 'teacher_positive_config TEACHER_WEIGHTS=%s\n' "${TEACHER_WEIGHTS_VALUE}"
printf 'teacher_positive_config TEACHER_VARIANT=%s\n' "${TEACHER_VARIANT_VALUE}"
printf 'teacher_positive_config POSITIVES_PER_CONDITION=%s\n' "${POSITIVES_PER_CONDITION_VALUE}"
printf 'teacher_positive_config NUM_STEPS=%s\n' "${NUM_STEPS_VALUE}"
printf 'teacher_positive_config CFG_STRENGTH=%s\n' "${CFG_STRENGTH_VALUE}"
printf 'teacher_positive_config SEED=%s\n' "${SEED_VALUE}"
printf 'teacher_positive_config GPU_COUNT=%s\n' "${NPROC_PER_NODE_VALUE}"
printf 'teacher_positive_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'teacher_positive_config CORE_DUMP_LIMIT=%s\n' "$(ulimit -c)"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE_VALUE}" \
  drifting/flux/build_teacher_positive_bank.py \
  --output-dir "${OUTPUT_DIR_VALUE}" \
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}" \
  --teacher-variant "${TEACHER_VARIANT_VALUE}" \
  --positives-per-condition "${POSITIVES_PER_CONDITION_VALUE}" \
  --num-steps "${NUM_STEPS_VALUE}" \
  --cfg-strength "${CFG_STRENGTH_VALUE}" \
  --seed "${SEED_VALUE}" \
  --use-rope \
  --amp \
  "${overwrite_args[@]}"
