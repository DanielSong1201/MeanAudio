#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_VALUE="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE_VALUE="${DEVICE:-cuda}"
DTYPE_VALUE="${DTYPE:-bfloat16}"

DATA_CONFIG_VALUE="${DATA_CONFIG:-config/data/resonate_flant5_44k.yaml}"
TRAIN_SPLIT_VALUE="${TRAIN_SPLIT:-AudioCaps_npz}"
TEACHER_POSITIVE_DIR_VALUE="${TEACHER_POSITIVE_DIR:-data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5}"
TEACHER_POSITIVE_COUNT_VALUE="${TEACHER_POSITIVE_COUNT:-3}"
TEACHER_WEIGHTS_VALUE="${TEACHER_WEIGHTS:-weights/Resonate_GRPO.pth}"
STUDENT_INIT_VALUE="${STUDENT_INIT:-weights/Resonate_GRPO.pth}"
LATENT_MEAN_VALUE="${LATENT_MEAN:-sets/latent_mean_44k.pt}"
LATENT_STD_VALUE="${LATENT_STD:-sets/latent_std_44k.pt}"

LAYERS_VALUE="${LAYERS:-audio_proj,joint_3,joint_5,joint_7,joint_11,joint_15,fused_0,fused_8,fused_17,fused_26,fused_35}"
FEATURE_NOISES_VALUE="${FEATURE_NOISES:-0.0,0.05,0.1,0.2}"
MAIN_FEATURE_NOISE_VALUE="${MAIN_FEATURE_NOISE:-0.1}"
POOL_TOKENS_VALUE="${POOL_TOKENS:-64}"
SAMPLE_COUNT_VALUE="${SAMPLE_COUNT:-64}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-1}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-0}"
SAMPLE_STRATEGY_VALUE="${SAMPLE_STRATEGY:-stride}"
SEED_VALUE="${SEED:-14159265}"
SAMPLES_PER_CONDITION_VALUE="${SAMPLES_PER_CONDITION:-4}"
GRAD_SAMPLE_COUNT_VALUE="${GRAD_SAMPLE_COUNT:-16}"
OUTPUT_CSV_VALUE="${OUTPUT_CSV:-exps/drifting_resonate_layer_probe/feature_layer_metrics.csv}"
PREPARE_ASSETS_VALUE="${PREPARE_ASSETS:-1}"
ASSET_DOWNLOAD_GPU_VALUE="${ASSET_DOWNLOAD_GPU:-${CUDA_VISIBLE_DEVICES_VALUE%%,*}}"

echo "probe_config CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "probe_config DEVICE=${DEVICE_VALUE}"
echo "probe_config DTYPE=${DTYPE_VALUE}"
echo "probe_config DATA_CONFIG=${DATA_CONFIG_VALUE}"
echo "probe_config TEACHER_POSITIVE_DIR=${TEACHER_POSITIVE_DIR_VALUE}"
echo "probe_config TEACHER_WEIGHTS=${TEACHER_WEIGHTS_VALUE}"
echo "probe_config STUDENT_INIT=${STUDENT_INIT_VALUE}"
echo "probe_config LAYERS=${LAYERS_VALUE}"
echo "probe_config FEATURE_NOISES=${FEATURE_NOISES_VALUE}"
echo "probe_config SAMPLE_COUNT=${SAMPLE_COUNT_VALUE}"
echo "probe_config OUTPUT_CSV=${OUTPUT_CSV_VALUE}"

if [[ "${PREPARE_ASSETS_VALUE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${ASSET_DOWNLOAD_GPU_VALUE}" \
  PYTHON="${PYTHON_VALUE}" \
  TEACHER_WEIGHTS="${TEACHER_WEIGHTS_VALUE}" \
  STUDENT_INIT="${STUDENT_INIT_VALUE}" \
  WITH_EVAL=0 \
  WITH_AV_BENCHMARK=0 \
  bash drifting/scripts/resonate/prepare_runtime_assets.sh
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
"${PYTHON_VALUE}" drifting/Resonate/probe_feature_layers.py \
  --data-config "${DATA_CONFIG_VALUE}" \
  --train-split "${TRAIN_SPLIT_VALUE}" \
  --teacher-positive-dir "${TEACHER_POSITIVE_DIR_VALUE}" \
  --teacher-positive-count "${TEACHER_POSITIVE_COUNT_VALUE}" \
  --teacher-weights "${TEACHER_WEIGHTS_VALUE}" \
  --student-init "${STUDENT_INIT_VALUE}" \
  --latent-mean "${LATENT_MEAN_VALUE}" \
  --latent-std "${LATENT_STD_VALUE}" \
  --layers "${LAYERS_VALUE}" \
  --feature-noises "${FEATURE_NOISES_VALUE}" \
  --main-feature-noise "${MAIN_FEATURE_NOISE_VALUE}" \
  --pool-tokens "${POOL_TOKENS_VALUE}" \
  --sample-count "${SAMPLE_COUNT_VALUE}" \
  --batch-size "${BATCH_SIZE_VALUE}" \
  --num-workers "${NUM_WORKERS_VALUE}" \
  --sample-strategy "${SAMPLE_STRATEGY_VALUE}" \
  --seed "${SEED_VALUE}" \
  --device "${DEVICE_VALUE}" \
  --dtype "${DTYPE_VALUE}" \
  --samples-per-condition "${SAMPLES_PER_CONDITION_VALUE}" \
  --grad-sample-count "${GRAD_SAMPLE_COUNT_VALUE}" \
  --output-csv "${OUTPUT_CSV_VALUE}"
