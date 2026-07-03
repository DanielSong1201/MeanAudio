#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

IFS=',' read -r -a visible_devices <<<"${CUDA_VISIBLE_DEVICES}"
gpu_count="${#visible_devices[@]}"
for device_id in "${visible_devices[@]}"; do
  if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: invalid CUDA device id %q in CUDA_VISIBLE_DEVICES=%q\n' \
      "${device_id}" "${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
done
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-${gpu_count}}"
if [[ ! "${NPROC_PER_NODE_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ERROR: NPROC_PER_NODE must be a positive integer\n' >&2
  exit 1
fi
if ((NPROC_PER_NODE_VALUE != gpu_count)); then
  printf 'ERROR: NPROC_PER_NODE=%s but CUDA_VISIBLE_DEVICES exposes %s GPUs\n' \
    "${NPROC_PER_NODE_VALUE}" "${gpu_count}" >&2
  exit 1
fi

PYTHON_VALUE="${PYTHON:-python}"
EXP_ID_VALUE="${EXP_ID:-resonate_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-exps/drifting_resonate}"
DATA_CONFIG_VALUE="${DATA_CONFIG:-config/data/resonate_flant5_44k.yaml}"
TRAIN_SPLIT_VALUE="${TRAIN_SPLIT:-AudioCaps_npz}"
TEACHER_POSITIVE_DIR_VALUE="${TEACHER_POSITIVE_DIR:-data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5}"
TEACHER_POSITIVE_COUNT_VALUE="${TEACHER_POSITIVE_COUNT:-3}"
TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/Resonate_GRPO.pth}"
STUDENT_INIT_VALUE="${STUDENT_INIT:-weights/Resonate_GRPO.pth}"
LATENT_MEAN_VALUE="${LATENT_MEAN:-sets/latent_mean_44k.pt}"
LATENT_STD_VALUE="${LATENT_STD:-sets/latent_std_44k.pt}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-1}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-4}"
ITERATIONS_VALUE="${ITERATIONS:-200000}"
LEARNING_RATE_VALUE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS_VALUE="${LR_WARMUP_STEPS:-1000}"
WEIGHT_DECAY_VALUE="${WEIGHT_DECAY:-1e-6}"
FEATURE_LAYERS_VALUE="${FEATURE_LAYERS:-joint_15,fused_17,fused_35}"
FEATURE_NOISE_VALUE="${FEATURE_NOISE:-0.1}"
POOL_TOKENS_VALUE="${POOL_TOKENS:-64}"
SAMPLES_PER_CONDITION_VALUE="${SAMPLES_PER_CONDITION:-4}"
RADII_VALUE="${RADII:-0.02,0.05,0.1,0.2}"
LAMBDA_FLOW_VALUE="${LAMBDA_FLOW:-0.1}"
LAMBDA_TFD_VALUE="${LAMBDA_TFD:-100.0}"
LAMBDA_ANCHOR_VALUE="${LAMBDA_ANCHOR:-1.0}"
ANCHOR_MARGIN_ALPHA_VALUE="${ANCHOR_MARGIN_ALPHA:-0.5}"
LOG_INTERVAL_VALUE="${LOG_INTERVAL:-20}"
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL_VALUE="${EVAL_INTERVAL:-10000}"
EVAL_OUTPUT_ROOT_VALUE="${EVAL_OUTPUT_ROOT:-exps/drifting_resonate_eval}"
EVAL_GT_CACHE_VALUE="${EVAL_GT_CACHE:-data/audiocaps/test-features}"
EVAL_GT_AUDIO_VALUE="${EVAL_GT_AUDIO:-gt_audio}"
EVAL_TSV_VALUE="${EVAL_TSV:-data/audiocaps_resonate/test.tsv}"
EVAL_NPZ_DIR_VALUE="${EVAL_NPZ_DIR:-data/audiocaps_resonate/test-npz-flant5-44k}"
EVAL_VAE_WEIGHTS_VALUE="${EVAL_VAE_WEIGHTS:-weights/v1-44.pth}"
EVAL_VOCODER_DIR_VALUE="${EVAL_VOCODER_DIR:-weights/bigvgan_v2_44khz_128band_512x}"
EVAL_DURATION_VALUE="${EVAL_DURATION:-10}"
EVAL_NUM_STEPS_VALUE="${EVAL_NUM_STEPS:-1}"
EVAL_CFG_STRENGTH_VALUE="${EVAL_CFG_STRENGTH:-4.5}"
EVAL_LIMIT_VALUE="${EVAL_LIMIT:-}"
EVAL_SKIP_AV_BENCHMARK_VALUE="${EVAL_SKIP_AV_BENCHMARK:-0}"
EMA_DECAY_VALUE="${EMA_DECAY:-0.9999}"
EMA_START_VALUE="${EMA_START:-0}"
EMA_UPDATE_INTERVAL_VALUE="${EMA_UPDATE_INTERVAL:-1}"
EMA_DEVICE_VALUE="${EMA_DEVICE:-cpu}"
CLIP_GRAD_NORM_VALUE="${CLIP_GRAD_NORM:-1.0}"
AUTO_RESUME_VALUE="${AUTO_RESUME:-1}"
DRY_RUN_VALUE="${DRY_RUN:-0}"
VALIDATE_PREPROCESSING_VALUE="${VALIDATE_PREPROCESSING:-1}"
VALIDATION_DEEP_VALUE="${VALIDATION_DEEP:-0}"
VALIDATION_SAMPLE_COUNT_VALUE="${VALIDATION_SAMPLE_COUNT:-16}"
PREPARE_ASSETS_VALUE="${PREPARE_ASSETS:-1}"
ASSETS_PREPARED_VALUE="${ASSETS_PREPARED:-0}"
ASSET_DOWNLOAD_GPU_VALUE="${ASSET_DOWNLOAD_GPU:-${visible_devices[0]}}"

train_args=(
  drifting/Resonate/train.py
  --exp-id "${EXP_ID_VALUE}"
  --output-root "${OUTPUT_ROOT_VALUE}"
  --data-config "${DATA_CONFIG_VALUE}"
  --train-split "${TRAIN_SPLIT_VALUE}"
  --teacher-positive-dir "${TEACHER_POSITIVE_DIR_VALUE}"
  --teacher-positive-count "${TEACHER_POSITIVE_COUNT_VALUE}"
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}"
  --student-init "${STUDENT_INIT_VALUE}"
  --latent-mean "${LATENT_MEAN_VALUE}"
  --latent-std "${LATENT_STD_VALUE}"
  --batch-size "${BATCH_SIZE_VALUE}"
  --num-workers "${NUM_WORKERS_VALUE}"
  --iterations "${ITERATIONS_VALUE}"
  --learning-rate "${LEARNING_RATE_VALUE}"
  --lr-warmup-steps "${LR_WARMUP_STEPS_VALUE}"
  --weight-decay "${WEIGHT_DECAY_VALUE}"
  --feature-layers "${FEATURE_LAYERS_VALUE}"
  --feature-noise "${FEATURE_NOISE_VALUE}"
  --pool-tokens "${POOL_TOKENS_VALUE}"
  --samples-per-condition "${SAMPLES_PER_CONDITION_VALUE}"
  --radii "${RADII_VALUE}"
  --lambda-flow "${LAMBDA_FLOW_VALUE}"
  --lambda-tfd "${LAMBDA_TFD_VALUE}"
  --lambda-anchor "${LAMBDA_ANCHOR_VALUE}"
  --anchor-margin-alpha "${ANCHOR_MARGIN_ALPHA_VALUE}"
  --log-interval "${LOG_INTERVAL_VALUE}"
  --save-interval "${SAVE_INTERVAL_VALUE}"
  --eval-interval "${EVAL_INTERVAL_VALUE}"
  --eval-output-root "${EVAL_OUTPUT_ROOT_VALUE}"
  --eval-gt-cache "${EVAL_GT_CACHE_VALUE}"
  --eval-gt-audio "${EVAL_GT_AUDIO_VALUE}"
  --eval-tsv "${EVAL_TSV_VALUE}"
  --eval-npz-dir "${EVAL_NPZ_DIR_VALUE}"
  --eval-vae-weights "${EVAL_VAE_WEIGHTS_VALUE}"
  --eval-vocoder-dir "${EVAL_VOCODER_DIR_VALUE}"
  --eval-duration "${EVAL_DURATION_VALUE}"
  --eval-num-steps "${EVAL_NUM_STEPS_VALUE}"
  --eval-cfg-strength "${EVAL_CFG_STRENGTH_VALUE}"
  --ema-decay "${EMA_DECAY_VALUE}"
  --ema-start "${EMA_START_VALUE}"
  --ema-update-interval "${EMA_UPDATE_INTERVAL_VALUE}"
  --ema-device "${EMA_DEVICE_VALUE}"
  --clip-grad-norm "${CLIP_GRAD_NORM_VALUE}"
  --use-rope
  --amp
)
if [[ -n "${EVAL_LIMIT_VALUE}" ]]; then
  train_args+=(--eval-limit "${EVAL_LIMIT_VALUE}")
fi
if [[ "${EVAL_SKIP_AV_BENCHMARK_VALUE}" == "1" ]]; then
  train_args+=(--eval-skip-av-benchmark)
fi
if [[ "${AUTO_RESUME_VALUE}" == "0" ]]; then
  train_args+=(--no-auto-resume)
fi

printf 'train_config EXP_ID=%s\n' "${EXP_ID_VALUE}"
printf 'train_config GPU_COUNT=%s\n' "${NPROC_PER_NODE_VALUE}"
printf 'train_config CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'train_config CORE_DUMP_LIMIT=%s\n' "$(ulimit -c)"
printf 'train_config DDP_TIMEOUT_MINUTES=%s\n' "${DDP_TIMEOUT_MINUTES}"
printf 'train_config TEACHER_WEIGHTS=%s\n' "${TEACHER_WEIGHTS_VALUE}"
printf 'train_config STUDENT_INIT=%s\n' "${STUDENT_INIT_VALUE}"
printf 'train_config TEACHER_POSITIVE_DIR=%s\n' "${TEACHER_POSITIVE_DIR_VALUE}"
printf 'train_config BATCH_SIZE_PER_GPU=%s\n' "${BATCH_SIZE_VALUE}"
printf 'train_config ITERATIONS=%s\n' "${ITERATIONS_VALUE}"
printf 'train_config LEARNING_RATE=%s\n' "${LEARNING_RATE_VALUE}"
printf 'train_config LR_WARMUP_STEPS=%s\n' "${LR_WARMUP_STEPS_VALUE}"
printf 'train_config LAMBDA_FLOW=%s\n' "${LAMBDA_FLOW_VALUE}"
printf 'train_config LAMBDA_TFD=%s\n' "${LAMBDA_TFD_VALUE}"
printf 'train_config LAMBDA_ANCHOR=%s\n' "${LAMBDA_ANCHOR_VALUE}"
printf 'train_config EVAL_INTERVAL=%s\n' "${EVAL_INTERVAL_VALUE}"
printf 'train_config EVAL_NUM_STEPS=%s\n' "${EVAL_NUM_STEPS_VALUE}"
printf 'train_config EVAL_CFG_STRENGTH=%s\n' "${EVAL_CFG_STRENGTH_VALUE}"
printf 'train_config VALIDATE_PREPROCESSING=%s\n' "${VALIDATE_PREPROCESSING_VALUE}"
printf 'train_config VALIDATION_DEEP=%s\n' "${VALIDATION_DEEP_VALUE}"
printf 'train_config PREPARE_ASSETS=%s\n' "${PREPARE_ASSETS_VALUE}"

if ((NPROC_PER_NODE_VALUE == 1)); then
  command=("${PYTHON_VALUE}" "${train_args[@]}")
else
  command=(
    torchrun
    --standalone
    --nproc_per_node="${NPROC_PER_NODE_VALUE}"
    "${train_args[@]}"
  )
fi

if [[ "${DRY_RUN_VALUE}" == "1" ]]; then
  printf 'train_command'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ "${PREPARE_ASSETS_VALUE}" == "1" && "${ASSETS_PREPARED_VALUE}" != "1" ]]; then
  if [[ ! "${EVAL_INTERVAL_VALUE}" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: EVAL_INTERVAL must be a non-negative integer\n' >&2
    exit 1
  fi
  with_eval=0
  with_av_benchmark=0
  if ((EVAL_INTERVAL_VALUE > 0)); then
    with_eval=1
    if [[ "${EVAL_SKIP_AV_BENCHMARK_VALUE}" != "1" ]]; then
      with_av_benchmark=1
    fi
  fi
  printf '%s | INFO | Preparing model assets before distributed training\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')"
  PYTHON="${PYTHON_VALUE}" \
  ASSET_DOWNLOAD_GPU="${ASSET_DOWNLOAD_GPU_VALUE}" \
  TEACHER_WEIGHTS="${TEACHER_WEIGHTS_VALUE}" \
  STUDENT_INIT="${STUDENT_INIT_VALUE}" \
  LATENT_MEAN="${LATENT_MEAN_VALUE}" \
  LATENT_STD="${LATENT_STD_VALUE}" \
  EVAL_VAE_WEIGHTS="${EVAL_VAE_WEIGHTS_VALUE}" \
  EVAL_VOCODER_DIR="${EVAL_VOCODER_DIR_VALUE}" \
  WITH_EVAL="${with_eval}" \
  WITH_AV_BENCHMARK="${with_av_benchmark}" \
  bash drifting/scripts/resonate/prepare_runtime_assets.sh
fi

if [[ "${VALIDATE_PREPROCESSING_VALUE}" == "1" ]]; then
  PYTHON="${PYTHON_VALUE}" \
  DATA_CONFIG="${DATA_CONFIG_VALUE}" \
  TEACHER_POSITIVE_DIR="${TEACHER_POSITIVE_DIR_VALUE}" \
  TEACHER_POSITIVE_COUNT="${TEACHER_POSITIVE_COUNT_VALUE}" \
  SAMPLE_COUNT="${VALIDATION_SAMPLE_COUNT_VALUE}" \
  DEEP="${VALIDATION_DEEP_VALUE}" \
  bash drifting/scripts/resonate/check_preprocessing_complete.sh
fi

if ! command -v flock >/dev/null 2>&1; then
  printf 'ERROR: flock is required to prevent duplicate training jobs\n' >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT_VALUE}/${EXP_ID_VALUE}"
lock_path="${OUTPUT_ROOT_VALUE}/${EXP_ID_VALUE}/.train.lock"
exec 9>"${lock_path}"
if ! flock -n 9; then
  printf 'ERROR: exp_id %s is already running (lock: %s)\n' \
    "${EXP_ID_VALUE}" "${lock_path}" >&2
  exit 1
fi

exec "${command[@]}"
