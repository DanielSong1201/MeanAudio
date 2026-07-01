#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

variant="${TEST_VARIANT:-tfd1}"
source_iteration="${SOURCE_ITERATION:-10000}"
fake_resume_iteration="${FAKE_RESUME_ITERATION:-19999}"
eval_iteration="${TEST_EVAL_ITERATION:-20000}"
target_iteration="${TEST_TARGET_ITERATION:-20100}"
source_output_root="${SOURCE_OUTPUT_ROOT:-exps/drifting_flux}"
test_output_root="${TEST_OUTPUT_ROOT:-exps/drifting_flux_resume_smoke}"
test_eval_output_root="${TEST_EVAL_OUTPUT_ROOT:-exps/drifting_flux_eval_resume_smoke}"
teacher_positive_dir="${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6}"
train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}"
timestamp="$(date '+%Y%m%d_%H%M%S')"

case "${variant}" in
  tfd1)
    source_exp_id="${SOURCE_EXP_ID:-flux_lr1e6_tfd1_anchor1_flow005_hybridpos4_warmup1000_200k_2gpu}"
    lambda_flow="${LAMBDA_FLOW:-0.05}"
    lambda_tfd="${LAMBDA_TFD:-1.0}"
    lambda_anchor="${LAMBDA_ANCHOR:-1.0}"
    ;;
  tfd100)
    source_exp_id="${SOURCE_EXP_ID:-flux_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k_2gpu}"
    lambda_flow="${LAMBDA_FLOW:-0.1}"
    lambda_tfd="${LAMBDA_TFD:-100.0}"
    lambda_anchor="${LAMBDA_ANCHOR:-1.0}"
    ;;
  *)
    printf 'ERROR: TEST_VARIANT must be tfd1 or tfd100, got %s\n' "${variant}" >&2
    exit 1
    ;;
esac

test_exp_id="${TEST_EXP_ID:-eval_resume_smoke_${variant}_src${source_iteration}_as${fake_resume_iteration}_${timestamp}}"
source_dir="${source_output_root}/${source_exp_id}"
source_raw="${source_dir}/${source_exp_id}_${source_iteration}.pth"
source_ema="${source_dir}/${source_exp_id}_${source_iteration}_ema.pth"
test_dir="${test_output_root}/${test_exp_id}"
test_eval_dir="${test_eval_output_root}/${test_exp_id}"
fake_raw="${test_dir}/${test_exp_id}_${fake_resume_iteration}.pth"
fake_ema="${test_dir}/${test_exp_id}_${fake_resume_iteration}_ema.pth"

if [[ "${fake_resume_iteration}" -ne $((eval_iteration - 1)) ]]; then
  printf 'ERROR: FAKE_RESUME_ITERATION must equal TEST_EVAL_ITERATION - 1.\n' >&2
  exit 1
fi
if [[ "${target_iteration}" -le "${eval_iteration}" ]]; then
  printf 'ERROR: TEST_TARGET_ITERATION must be greater than TEST_EVAL_ITERATION.\n' >&2
  exit 1
fi
if [[ ! -f "${source_raw}" ]]; then
  printf 'ERROR: source checkpoint does not exist: %s\n' "${source_raw}" >&2
  printf 'Set TEST_VARIANT, SOURCE_EXP_ID, SOURCE_OUTPUT_ROOT, or SOURCE_ITERATION to an existing checkpoint.\n' >&2
  exit 1
fi
if [[ ! -f "${teacher_positive_dir}/complete.json" ]]; then
  printf 'ERROR: teacher-positive bank is incomplete: %s/complete.json\n' \
    "${teacher_positive_dir}" >&2
  exit 1
fi
if [[ -e "${test_dir}" || -e "${test_eval_dir}" ]]; then
  printf 'ERROR: refusing to overwrite an existing smoke-test path:\n  %s\n  %s\n' \
    "${test_dir}" "${test_eval_dir}" >&2
  exit 1
fi

mkdir -p "${test_dir}"
cp -- "${source_raw}" "${fake_raw}"
if [[ -f "${source_ema}" ]]; then
  cp -- "${source_ema}" "${fake_ema}"
else
  printf 'WARNING: source EMA checkpoint is missing; test EMA will initialize from the copied student.\n'
fi

printf '%s\n' "================================================================"
printf 'Eval-resume smoke test (isolated; source files are read-only)\n'
printf 'TEST_VARIANT=%s\n' "${variant}"
printf 'SOURCE_CHECKPOINT=%s\n' "${source_raw}"
printf 'COPIED_AS=%s\n' "${fake_raw}"
printf 'TEST_EXP_ID=%s\n' "${test_exp_id}"
printf 'TEST_OUTPUT_ROOT=%s\n' "${test_output_root}"
printf 'TEST_EVAL_OUTPUT_ROOT=%s\n' "${test_eval_output_root}"
printf 'TRAIN_SCRIPT=%s\n' "${train_script}"
printf 'EXPECTED_FLOW=%s: resume %s -> train/eval %s -> continue through %s\n' \
  "${source_iteration}" "${fake_resume_iteration}" "${eval_iteration}" "${target_iteration}"
printf '%s\n' "================================================================"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
NPROC_PER_NODE=2 \
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}" \
EVAL_CONSOLE_OUTPUT=0 \
EVAL_FAILURE_FATAL=1 \
QUIET_CONSOLE_AFTER_TQDM=1 \
TQDM_POSITION=0 \
TQDM_DESC="[eval-resume-smoke]" \
EXP_ID="${test_exp_id}" \
OUTPUT_ROOT="${test_output_root}" \
AUTO_RESUME=1 \
BATCH_SIZE=1 \
SAMPLES_PER_CONDITION=4 \
TEACHER_POSITIVE_DIR="${teacher_positive_dir}" \
TEACHER_POSITIVE_COUNT=3 \
ITERATIONS="${target_iteration}" \
LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}" \
LAMBDA_FLOW="${lambda_flow}" \
LAMBDA_TFD="${lambda_tfd}" \
LAMBDA_ANCHOR="${lambda_anchor}" \
FEATURE_NOISE="${FEATURE_NOISE:-0.1}" \
LOG_INTERVAL=1 \
SAVE_INTERVAL="${eval_iteration}" \
EVAL_INTERVAL="${eval_iteration}" \
EVAL_OUTPUT_ROOT="${test_eval_output_root}" \
EMA_DECAY="${EMA_DECAY:-0.9999}" \
bash "${train_script}"

metrics_path="${test_dir}/metrics.csv"
train_log="${test_dir}/train.log"
eval_driver_log="${test_eval_dir}/it_$(printf '%08d' "${eval_iteration}")/eval_driver.log"
evaluate_log="${test_eval_dir}/it_$(printf '%08d' "${eval_iteration}")/evaluate.log"

for expected_iteration in $(seq "${eval_iteration}" "${target_iteration}"); do
  if ! awk -F, -v expected="${expected_iteration}" \
    'NR > 1 && $1 == expected { found=1 } END { exit(found ? 0 : 1) }' \
    "${metrics_path}"; then
    printf 'ERROR: metrics.csv is missing iteration %s: %s\n' \
      "${expected_iteration}" "${metrics_path}" >&2
    exit 1
  fi
done

if ! grep -q "TRAIN_RESUME_AFTER_EVAL iteration=${eval_iteration} " "${train_log}"; then
  printf 'ERROR: missing eval-resume marker in %s\n' "${train_log}" >&2
  exit 1
fi
if [[ ! -f "${eval_driver_log}" || ! -f "${evaluate_log}" ]]; then
  printf 'ERROR: eval logs are incomplete:\n  %s\n  %s\n' \
    "${eval_driver_log}" "${evaluate_log}" >&2
  exit 1
fi

printf '%s\n' "================================================================"
printf 'PASS: eval-resume smoke test completed without touching the source experiment.\n'
printf 'Verified iterations: %s..%s\n' "${eval_iteration}" "${target_iteration}"
printf 'Resume marker: %s\n' "${train_log}"
printf 'Eval driver log: %s\n' "${eval_driver_log}"
printf 'Metrics log: %s\n' "${metrics_path}"
printf '%s\n' "================================================================"
