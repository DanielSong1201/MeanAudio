#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
cd "$(dirname "$0")/../../.."

# Evaluate the released Resonate checkpoint on AV-Benchmark.
#
# Defaults run both the one-step baseline and the 25-step teacher upper bound:
#   bash drifting/scripts/resonate/test_resonate_avbench.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0 NUM_STEPS_LIST="1 4 8 25" bash drifting/scripts/resonate/test_resonate_avbench.sh
#   NUM_STEPS=25 CFG_STRENGTH=4.5 bash drifting/scripts/resonate/test_resonate_avbench.sh
#   EVAL_LIMIT=16 NUM_STEPS=1 bash drifting/scripts/resonate/test_resonate_avbench.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_VALUE="${PYTHON:-python}"
MODEL_PATH_VALUE="${MODEL_PATH:-weights/Resonate_GRPO.pth}"
RUN_ID_VALUE="${RUN_ID:-resonate_grpo_avbench}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-exps/drifting_resonate_eval}"
GT_CACHE_VALUE="${GT_CACHE:-data/audiocaps/test-features}"
GT_AUDIO_VALUE="${GT_AUDIO:-gt_audio}"
EVAL_TSV_VALUE="${EVAL_TSV:-data/audiocaps_resonate/test.tsv}"
EVAL_NPZ_DIR_VALUE="${EVAL_NPZ_DIR:-data/audiocaps_resonate/test-npz-flant5-44k}"
VAE_WEIGHTS_VALUE="${VAE_WEIGHTS:-weights/v1-44.pth}"
VOCODER_WEIGHTS_VALUE="${VOCODER_WEIGHTS:-weights/bigvgan_v2_44khz_128band_512x}"
DURATION_VALUE="${DURATION:-10}"
SEED_VALUE="${SEED:-42}"
CFG_STRENGTH_VALUE="${CFG_STRENGTH:-4.5}"
USE_ROPE_VALUE="${USE_ROPE:-1}"
EVAL_LIMIT_VALUE="${EVAL_LIMIT:-}"
PREPARE_ASSETS_VALUE="${PREPARE_ASSETS:-1}"
NUM_STEPS_LIST_VALUE="${NUM_STEPS_LIST:-${NUM_STEPS:-1 25}}"
asset_download_gpu="${ASSET_DOWNLOAD_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"

cfg_label="${CFG_STRENGTH_VALUE//./p}"
limit_label="full"
if [[ -n "${EVAL_LIMIT_VALUE}" ]]; then
  limit_label="limit_${EVAL_LIMIT_VALUE}"
fi

printf 'resonate_avbench MODEL_PATH=%s\n' "${MODEL_PATH_VALUE}"
printf 'resonate_avbench RUN_ID=%s\n' "${RUN_ID_VALUE}"
printf 'resonate_avbench CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'resonate_avbench NUM_STEPS_LIST=%s\n' "${NUM_STEPS_LIST_VALUE}"
printf 'resonate_avbench CFG_STRENGTH=%s\n' "${CFG_STRENGTH_VALUE}"
printf 'resonate_avbench EVAL_LIMIT=%s\n' "${EVAL_LIMIT_VALUE:-<full>}"
printf 'resonate_avbench OUTPUT_ROOT=%s/%s\n' "${OUTPUT_ROOT_VALUE}" "${RUN_ID_VALUE}"

if [[ "${PREPARE_ASSETS_VALUE}" == "1" ]]; then
  PYTHON="${PYTHON_VALUE}" \
  ASSET_DOWNLOAD_GPU="${asset_download_gpu}" \
  TEACHER_WEIGHTS="${MODEL_PATH_VALUE}" \
  STUDENT_INIT="${MODEL_PATH_VALUE}" \
  EVAL_VAE_WEIGHTS="${VAE_WEIGHTS_VALUE}" \
  EVAL_VOCODER_DIR="${VOCODER_WEIGHTS_VALUE}" \
  WITH_EVAL=1 \
  WITH_AV_BENCHMARK=1 \
  bash drifting/scripts/resonate/prepare_runtime_assets.sh
fi

summary_dir="${OUTPUT_ROOT_VALUE}/${RUN_ID_VALUE}"
summary_csv="${summary_dir}/avbench_summary.csv"
mkdir -p "${summary_dir}"
if [[ ! -s "${summary_csv}" ]]; then
  printf 'model_path,num_steps,cfg_strength,eval_limit,output_path,metrics_json\n' \
    > "${summary_csv}"
fi

for num_steps in ${NUM_STEPS_LIST_VALUE}; do
  output_path="${summary_dir}/steps_${num_steps}_cfg_${cfg_label}_${limit_label}"
  printf '\n'
  printf 'resonate_avbench_start NUM_STEPS=%s OUTPUT_PATH=%s\n' \
    "${num_steps}" "${output_path}"

  PREPARE_ASSETS=0 \
  MODEL_PATH="${MODEL_PATH_VALUE}" \
  OUTPUT_PATH="${output_path}" \
  GT_CACHE="${GT_CACHE_VALUE}" \
  GT_AUDIO="${GT_AUDIO_VALUE}" \
  EVAL_TSV="${EVAL_TSV_VALUE}" \
  EVAL_NPZ_DIR="${EVAL_NPZ_DIR_VALUE}" \
  VAE_WEIGHTS="${VAE_WEIGHTS_VALUE}" \
  VOCODER_WEIGHTS="${VOCODER_WEIGHTS_VALUE}" \
  DURATION="${DURATION_VALUE}" \
  SEED="${SEED_VALUE}" \
  NUM_STEPS="${num_steps}" \
  CFG_STRENGTH="${CFG_STRENGTH_VALUE}" \
  USE_ROPE="${USE_ROPE_VALUE}" \
  EVAL_LIMIT="${EVAL_LIMIT_VALUE}" \
  EVAL_SKIP_AV_BENCHMARK=0 \
  bash drifting/scripts/resonate/eval_checkpoint.sh

  metrics_path="${output_path}/cache/output_metrics.json"
  if [[ ! -s "${metrics_path}" ]]; then
    printf 'ERROR: missing AV-Benchmark metrics: %s\n' "${metrics_path}" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s,%s\n' \
    "${MODEL_PATH_VALUE}" \
    "${num_steps}" \
    "${CFG_STRENGTH_VALUE}" \
    "${EVAL_LIMIT_VALUE}" \
    "${output_path}" \
    "${metrics_path}" \
    >> "${summary_csv}"

  "${PYTHON_VALUE}" - "${metrics_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
print(f"resonate_avbench_metrics {path}")
for key in sorted(metrics):
    print(f"  {key}: {metrics[key]}")
PY

  printf 'resonate_avbench_done NUM_STEPS=%s METRICS=%s\n' \
    "${num_steps}" "${metrics_path}"
done

printf '\nresonate_avbench_summary %s\n' "${summary_csv}"
