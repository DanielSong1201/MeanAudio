#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
cd "$(dirname "$0")/../../.."

timestamp="$(date '+%Y%m%d_%H%M%S')"
smoke_id="${EXP_ID:-resonate_train_eval_smoke_${timestamp}}"
output_root="${OUTPUT_ROOT:-exps/drifting_resonate_smoke}"
eval_root="${EVAL_OUTPUT_ROOT:-exps/drifting_resonate_eval_smoke}"
train_dir="${output_root}/${smoke_id}"
eval_dir="${eval_root}/${smoke_id}"
python_value="${PYTHON:-python}"
asset_download_gpu="${ASSET_DOWNLOAD_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
asset_download_gpu="${asset_download_gpu%%,*}"

if [[ -e "${train_dir}" || -e "${eval_dir}" ]]; then
  printf 'ERROR: refusing to overwrite smoke-test output:\n  %s\n  %s\n' \
    "${train_dir}" "${eval_dir}" >&2
  exit 1
fi

with_av_benchmark=1
if [[ "${EVAL_SKIP_AV_BENCHMARK:-0}" == "1" ]]; then
  with_av_benchmark=0
fi
printf '%s | INFO | Preparing smoke-test model assets before training\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')"
PYTHON="${python_value}" \
ASSET_DOWNLOAD_GPU="${asset_download_gpu}" \
WITH_EVAL=1 \
WITH_AV_BENCHMARK="${with_av_benchmark}" \
bash drifting/scripts/resonate/prepare_runtime_assets.sh

required=(
  weights/Resonate_GRPO.pth
  weights/v1-44.pth
  weights/bigvgan_v2_44khz_128band_512x/config.json
  weights/bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt
  sets/latent_mean_44k.pt
  sets/latent_std_44k.pt
  data/audiocaps_resonate/train.tsv
  data/audiocaps_resonate/train-npz-flant5-44k/complete.json
  data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5/complete.json
  data/audiocaps_resonate/test.tsv
  data/audiocaps_resonate/test-npz-flant5-44k/complete.json
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'ERROR: missing smoke-test asset: %s\n' "${path}" >&2
    exit 1
  fi
done

if [[ "${EVAL_SKIP_AV_BENCHMARK:-0}" != "1" ]]; then
  if [[ ! -f av-benchmark/evaluate.py ]]; then
    printf 'ERROR: AV-Benchmark smoke requires av-benchmark/evaluate.py\n' >&2
    printf 'Set EVAL_SKIP_AV_BENCHMARK=1 only to test generation without metrics.\n' >&2
    exit 1
  fi
  gt_cache="${EVAL_GT_CACHE:-data/audiocaps/test-features}"
  gt_audio="${EVAL_GT_AUDIO:-gt_audio}"
  gt_cache_files=(
    pann_features.pth
    vggish_features.pth
    passt_features_embed.pth
    passt_logits.pth
  )
  missing_gt_cache=0
  for filename in "${gt_cache_files[@]}"; do
    if [[ ! -s "${gt_cache}/${filename}" ]]; then
      missing_gt_cache=1
      break
    fi
  done
  if ((missing_gt_cache == 1)) && [[ ! -d "${gt_audio}" ]]; then
    printf 'ERROR: AV-Benchmark GT cache is incomplete and GT audio is missing: %s\n' \
      "${gt_audio}" >&2
    exit 1
  fi
fi

printf 'smoke_config EXP_ID=%s\n' "${smoke_id}"
printf 'smoke_config FLOW=it1 train -> it2 save/eval -> it3 resume train\n'
printf 'smoke_config OUTPUT=%s\n' "${train_dir}"
printf 'smoke_config EVAL_OUTPUT=%s\n' "${eval_dir}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-}" \
EXP_ID="${smoke_id}" \
OUTPUT_ROOT="${output_root}" \
EVAL_OUTPUT_ROOT="${eval_root}" \
AUTO_RESUME=0 \
ITERATIONS=3 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
LEARNING_RATE=1e-6 \
LR_WARMUP_STEPS=2 \
LAMBDA_TFD="${LAMBDA_TFD:-1.0}" \
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}" \
LAMBDA_FLOW="${LAMBDA_FLOW:-0.05}" \
LOG_INTERVAL=1 \
SAVE_INTERVAL=2 \
EVAL_INTERVAL=2 \
EVAL_LIMIT="${EVAL_LIMIT:-16}" \
EVAL_SKIP_AV_BENCHMARK="${EVAL_SKIP_AV_BENCHMARK:-0}" \
EVAL_FAILURE_FATAL=1 \
ASSETS_PREPARED=1 \
bash drifting/scripts/resonate/train.sh

metrics="${train_dir}/metrics.csv"
train_log="${train_dir}/train.log"
eval_log="${eval_dir}/it_00000002/evaluate.log"
eval_status="${eval_dir}/it_00000002/distributed_eval_status.json"
checkpoint="${train_dir}/${smoke_id}_2.pth"

for path in \
  "${metrics}" \
  "${train_log}" \
  "${eval_log}" \
  "${eval_status}" \
  "${checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: smoke output missing: %s\n' "${path}" >&2
    exit 1
  fi
done
if ! grep -q '"success": true' "${eval_status}"; then
  printf 'ERROR: eval status is not successful: %s\n' "${eval_status}" >&2
  exit 1
fi
if [[ "${EVAL_SKIP_AV_BENCHMARK:-0}" != "1" ]] \
  && [[ ! -s "${eval_dir}/it_00000002/cache/output_metrics.json" ]]; then
  printf 'ERROR: smoke AV-Benchmark metrics are missing\n' >&2
  exit 1
fi
if ! awk -F, 'NR > 1 && $1 == 3 {found=1} END {exit(found ? 0 : 1)}' "${metrics}"; then
  printf 'ERROR: training did not continue through iteration 3 after eval\n' >&2
  exit 1
fi
if ! grep -q 'TRAIN_RESUME_AFTER_EVAL iteration=2 ' "${train_log}"; then
  printf 'ERROR: eval-resume marker missing from %s\n' "${train_log}" >&2
  exit 1
fi

printf 'PASS: Resonate train/save/eval/resume smoke test completed\n'
printf '  train: %s\n  eval:  %s\n' "${train_dir}" "${eval_dir}"
