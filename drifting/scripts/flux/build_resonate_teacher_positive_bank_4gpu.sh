#!/usr/bin/env bash
set -euo pipefail

# Independent GPU workers share one bank directory. They do not initialize
# torch.distributed/NCCL; rank 0 aggregates progress through small shared files.
# The historical "4gpu" filename is retained for backward compatibility; the
# worker count is derived from CUDA_VISIBLE_DEVICES and is no longer fixed.

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#visible_gpus[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  printf 'ERROR: CUDA_VISIBLE_DEVICES must contain at least one GPU id.\n' >&2
  exit 1
fi
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-${GPU_COUNT}}"
if [[ ! "${NPROC_PER_NODE_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: NPROC_PER_NODE must be a positive integer; got %q.\n' \
    "${NPROC_PER_NODE_VALUE}" >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE_VALUE}" -ne "${GPU_COUNT}" ]]; then
  printf 'ERROR: NPROC_PER_NODE=%s must match the %s ids in CUDA_VISIBLE_DEVICES=%q.\n' \
    "${NPROC_PER_NODE_VALUE}" "${GPU_COUNT}" "${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
  gpu="${visible_gpus[gpu_index]}"
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: invalid GPU id %q in CUDA_VISIBLE_DEVICES.\n' "${gpu}" >&2
    exit 1
  fi
  for ((previous_index = 0; previous_index < gpu_index; previous_index++)); do
    if [[ "${gpu}" == "${visible_gpus[previous_index]}" ]]; then
      printf 'ERROR: duplicate GPU id %q in CUDA_VISIBLE_DEVICES.\n' "${gpu}" >&2
      exit 1
    fi
  done
done

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
PROGRESS_POLL_INTERVAL_VALUE="${PROGRESS_POLL_INTERVAL:-1}"
COORDINATION_TIMEOUT_MINUTES_VALUE="${COORDINATION_TIMEOUT_MINUTES:-720}"
MASTER_ADDR_VALUE="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_VALUE="${MASTER_PORT:-$((20000 + ($$ % 20000)))}"

# A unique shared-files namespace prevents stale markers from a previous job
# from being counted by this run. torchrun propagates it to every worker.
export BANK_RUN_ID="${BANK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

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

printf 'resonate_teacher_bank_multigpu CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'resonate_teacher_bank_multigpu NPROC_PER_NODE=%s\n' "${NPROC_PER_NODE_VALUE}"
for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
  printf 'resonate_teacher_bank_multigpu rank=%s logical_cuda=%s physical_gpu=%s\n' \
    "${gpu_index}" "${gpu_index}" "${visible_gpus[gpu_index]}"
done
printf 'resonate_teacher_bank_multigpu BANK_RUN_ID=%s\n' "${BANK_RUN_ID}"
printf 'resonate_teacher_bank_multigpu NCCL_COLLECTIVES=disabled\n'
printf 'resonate_teacher_bank_multigpu TORCHRUN_RENDEZVOUS=%s:%s\n' \
  "${MASTER_ADDR_VALUE}" "${MASTER_PORT_VALUE}"
printf 'resonate_teacher_bank_multigpu RESONATE_ROOT=%s\n' "${RESONATE_ROOT_VALUE}"
printf 'resonate_teacher_bank_multigpu MANIFEST=%s\n' "${MANIFEST_VALUE}"
printf 'resonate_teacher_bank_multigpu OUTPUT_DIR=%s\n' "${OUTPUT_DIR_VALUE}"
printf 'resonate_teacher_bank_multigpu POSITIVES_PER_CONDITION=%s\n' "${POSITIVES_PER_CONDITION_VALUE}"
printf 'resonate_teacher_bank_multigpu NUM_STEPS=%s CFG_STRENGTH=%s\n' \
  "${NUM_STEPS_VALUE}" "${CFG_STRENGTH_VALUE}"
printf 'resonate_teacher_bank_multigpu START_INDEX=%s END_INDEX=%s LIMIT=%s\n' \
  "${START_INDEX_VALUE}" "${END_INDEX_VALUE:-all}" "${LIMIT_VALUE:-all}"
printf 'resonate_teacher_bank_multigpu AUTO_DOWNLOAD=%s OVERWRITE=%s\n' \
  "${AUTO_DOWNLOAD_VALUE}" "${OVERWRITE_VALUE}"
printf 'resonate_teacher_bank_multigpu COORDINATION_TIMEOUT_MINUTES=%s\n' \
  "${COORDINATION_TIMEOUT_MINUTES_VALUE}"

exec "${PYTHON_VALUE}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE_VALUE}" \
  --master-addr="${MASTER_ADDR_VALUE}" \
  --master-port="${MASTER_PORT_VALUE}" \
  --max-restarts=0 \
  -m drifting.flux.build_resonate_teacher_positive_bank \
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
  --progress-poll-interval "${PROGRESS_POLL_INTERVAL_VALUE}" \
  --coordination-timeout-minutes "${COORDINATION_TIMEOUT_MINUTES_VALUE}" \
  "${extra_args[@]}"
