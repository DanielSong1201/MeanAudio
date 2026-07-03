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

if [[ -e "${train_dir}" || -e "${eval_dir}" ]]; then
  printf 'ERROR: refusing to overwrite smoke-test output:\n  %s\n  %s\n' \
    "${train_dir}" "${eval_dir}" >&2
  exit 1
fi

required=(
  weights/Resonate_GRPO.pth
  weights/v1-44.pth
  weights/bigvgan_v2_44khz_128band_512x
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
  if [[ ! -f av-benchmark/evaluate.py || ! -d gt_audio ]]; then
    printf 'ERROR: AV-Benchmark smoke requires av-benchmark/evaluate.py and gt_audio/\n' >&2
    printf 'Set EVAL_SKIP_AV_BENCHMARK=1 only to test generation without metrics.\n' >&2
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
bash drifting/scripts/resonate/train.sh

metrics="${train_dir}/metrics.csv"
train_log="${train_dir}/train.log"
eval_log="${eval_dir}/it_00000002/evaluate.log"
checkpoint="${train_dir}/${smoke_id}_2.pth"

for path in "${metrics}" "${train_log}" "${eval_log}" "${checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: smoke output missing: %s\n' "${path}" >&2
    exit 1
  fi
done
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
