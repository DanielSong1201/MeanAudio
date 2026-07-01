#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

source_iteration="${SOURCE_ITERATION:-10000}"
fake_resume_iteration="${FAKE_RESUME_ITERATION:-19500}"
eval_iteration="${TEST_EVAL_ITERATION:-20000}"
target_iteration="${TEST_TARGET_ITERATION:-20100}"
source_output_root="${SOURCE_OUTPUT_ROOT:-exps/drifting_flux}"
test_output_root="${TEST_OUTPUT_ROOT:-exps/drifting_flux_resume_smoke}"
test_eval_output_root="${TEST_EVAL_OUTPUT_ROOT:-exps/drifting_flux_eval_resume_smoke}"
teacher_positive_dir="${TEACHER_POSITIVE_DIR:-data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6}"
train_script="${TRAIN_SCRIPT:-drifting/scripts/flux/train_flux_2x4090.sh}"
tfd1_gpus="${TFD1_GPUS:-0,1}"
tfd100_gpus="${TFD100_GPUS:-2,3}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
suite_id="${TEST_SUITE_ID:-eval_resume_parallel_src${source_iteration}_as${fake_resume_iteration}_${timestamp}}"

tfd1_source_exp_id="${TFD1_SOURCE_EXP_ID:-flux_lr1e6_tfd1_anchor1_flow005_hybridpos4_warmup1000_200k_2gpu}"
tfd100_source_exp_id="${TFD100_SOURCE_EXP_ID:-flux_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k_2gpu}"
tfd1_test_exp_id="${TFD1_TEST_EXP_ID:-${suite_id}_tfd1}"
tfd100_test_exp_id="${TFD100_TEST_EXP_ID:-${suite_id}_tfd100}"

if [[ "${fake_resume_iteration}" -ge "${eval_iteration}" ]]; then
  printf 'ERROR: FAKE_RESUME_ITERATION must be less than TEST_EVAL_ITERATION.\n' >&2
  exit 1
fi
if [[ "${target_iteration}" -le "${eval_iteration}" ]]; then
  printf 'ERROR: TEST_TARGET_ITERATION must be greater than TEST_EVAL_ITERATION.\n' >&2
  exit 1
fi
if [[ ! -f "${teacher_positive_dir}/complete.json" ]]; then
  printf 'ERROR: teacher-positive bank is incomplete: %s/complete.json\n' \
    "${teacher_positive_dir}" >&2
  exit 1
fi

source_checkpoint_path() {
  local source_exp_id="$1"
  printf '%s/%s/%s_%s.pth' \
    "${source_output_root}" \
    "${source_exp_id}" \
    "${source_exp_id}" \
    "${source_iteration}"
}

source_ema_path() {
  local source_exp_id="$1"
  printf '%s/%s/%s_%s_ema.pth' \
    "${source_output_root}" \
    "${source_exp_id}" \
    "${source_exp_id}" \
    "${source_iteration}"
}

test_train_dir() {
  printf '%s/%s' "${test_output_root}" "$1"
}

test_eval_dir() {
  printf '%s/%s' "${test_eval_output_root}" "$1"
}

validate_source_and_destination() {
  local label="$1"
  local source_exp_id="$2"
  local test_exp_id="$3"
  local source_raw
  local train_dir
  local eval_dir
  source_raw="$(source_checkpoint_path "${source_exp_id}")"
  train_dir="$(test_train_dir "${test_exp_id}")"
  eval_dir="$(test_eval_dir "${test_exp_id}")"

  if [[ ! -f "${source_raw}" ]]; then
    printf 'ERROR: %s source checkpoint does not exist: %s\n' \
      "${label}" "${source_raw}" >&2
    exit 1
  fi
  if [[ -e "${train_dir}" || -e "${eval_dir}" ]]; then
    printf 'ERROR: refusing to overwrite existing %s smoke-test paths:\n  %s\n  %s\n' \
      "${label}" "${train_dir}" "${eval_dir}" >&2
    exit 1
  fi
}

prepare_fake_checkpoint() {
  local label="$1"
  local source_exp_id="$2"
  local test_exp_id="$3"
  local source_raw
  local source_ema
  local train_dir
  local fake_raw
  local fake_ema
  source_raw="$(source_checkpoint_path "${source_exp_id}")"
  source_ema="$(source_ema_path "${source_exp_id}")"
  train_dir="$(test_train_dir "${test_exp_id}")"
  fake_raw="${train_dir}/${test_exp_id}_${fake_resume_iteration}.pth"
  fake_ema="${train_dir}/${test_exp_id}_${fake_resume_iteration}_ema.pth"

  mkdir -p "${train_dir}"
  cp -- "${source_raw}" "${fake_raw}"
  if [[ -f "${source_ema}" ]]; then
    cp -- "${source_ema}" "${fake_ema}"
  else
    printf 'WARNING: %s source EMA is missing; test EMA will initialize from student weights.\n' \
      "${label}"
  fi
  printf '%s copied %s -> %s\n' "${label}" "${source_raw}" "${fake_raw}"
}

run_one() {
  local label="$1"
  local test_exp_id="$2"
  local gpus="$3"
  local tqdm_position="$4"
  local lambda_flow="$5"
  local lambda_tfd="$6"
  local lambda_anchor="$7"

  CUDA_VISIBLE_DEVICES="${gpus}" \
  NPROC_PER_NODE=2 \
  DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}" \
  EVAL_CONSOLE_OUTPUT=0 \
  EVAL_FAILURE_FATAL=1 \
  QUIET_CONSOLE_AFTER_TQDM=1 \
  TQDM_POSITION="${tqdm_position}" \
  TQDM_DESC="[smoke-${label}-GPU${gpus}]" \
  LOG_PREFIX="[smoke-${label}-GPU${gpus}]" \
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
}

validate_result() {
  local label="$1"
  local test_exp_id="$2"
  local train_dir
  local eval_dir
  local metrics_path
  local train_log
  local eval_driver_log
  local evaluate_log
  local eval_tag
  train_dir="$(test_train_dir "${test_exp_id}")"
  eval_dir="$(test_eval_dir "${test_exp_id}")"
  metrics_path="${train_dir}/metrics.csv"
  train_log="${train_dir}/train.log"
  eval_tag="$(printf '%08d' "${eval_iteration}")"
  eval_driver_log="${eval_dir}/it_${eval_tag}/eval_driver.log"
  evaluate_log="${eval_dir}/it_${eval_tag}/evaluate.log"

  for expected_iteration in $(seq "$((fake_resume_iteration + 1))" "${target_iteration}"); do
    if ! awk -F, -v expected="${expected_iteration}" \
      'NR > 1 && $1 == expected { found=1 } END { exit(found ? 0 : 1) }' \
      "${metrics_path}"; then
      printf 'ERROR: %s metrics.csv is missing iteration %s: %s\n' \
        "${label}" "${expected_iteration}" "${metrics_path}" >&2
      return 1
    fi
  done
  if ! grep -q "TRAIN_RESUME_AFTER_EVAL iteration=${eval_iteration} " "${train_log}"; then
    printf 'ERROR: %s is missing its eval-resume marker: %s\n' \
      "${label}" "${train_log}" >&2
    return 1
  fi
  if [[ ! -f "${eval_driver_log}" || ! -f "${evaluate_log}" ]]; then
    printf 'ERROR: %s eval logs are incomplete:\n  %s\n  %s\n' \
      "${label}" "${eval_driver_log}" "${evaluate_log}" >&2
    return 1
  fi

  printf 'PASS %s: iterations %s..%s, resume marker and eval logs verified.\n' \
    "${label}" "$((fake_resume_iteration + 1))" "${target_iteration}"
  printf '  train: %s\n  eval:  %s\n' "${train_dir}" "${eval_dir}"
}

validate_source_and_destination "tfd1" "${tfd1_source_exp_id}" "${tfd1_test_exp_id}"
validate_source_and_destination "tfd100" "${tfd100_source_exp_id}" "${tfd100_test_exp_id}"
prepare_fake_checkpoint "tfd1" "${tfd1_source_exp_id}" "${tfd1_test_exp_id}"
prepare_fake_checkpoint "tfd100" "${tfd100_source_exp_id}" "${tfd100_test_exp_id}"

printf '%s\n' "================================================================"
printf 'Parallel eval-resume smoke test; source checkpoints remain untouched.\n'
printf 'TFD1_GPUS=%s TEST_EXP_ID=%s\n' "${tfd1_gpus}" "${tfd1_test_exp_id}"
printf 'TFD100_GPUS=%s TEST_EXP_ID=%s\n' "${tfd100_gpus}" "${tfd100_test_exp_id}"
printf 'FLOW: source %s -> fake resume %s -> eval %s -> continue through %s\n' \
  "${source_iteration}" "${fake_resume_iteration}" "${eval_iteration}" "${target_iteration}"
printf 'TRAIN_ROOT=%s\nEVAL_ROOT=%s\nTRAIN_SCRIPT=%s\n' \
  "${test_output_root}" "${test_eval_output_root}" "${train_script}"
printf '%s\n' "================================================================"

run_one "tfd1" "${tfd1_test_exp_id}" "${tfd1_gpus}" "0" "0.05" "1.0" "1.0" &
pid_tfd1=$!
run_one "tfd100" "${tfd100_test_exp_id}" "${tfd100_gpus}" "1" "0.1" "100.0" "1.0" &
pid_tfd100=$!

failed=0
if ! wait "${pid_tfd1}"; then
  printf 'ERROR: tfd1 smoke process failed.\n' >&2
  failed=1
fi
if ! wait "${pid_tfd100}"; then
  printf 'ERROR: tfd100 smoke process failed.\n' >&2
  failed=1
fi
if [[ "${failed}" != "0" ]]; then
  exit 1
fi

validate_result "tfd1" "${tfd1_test_exp_id}"
validate_result "tfd100" "${tfd100_test_exp_id}"

printf '%s\n' "================================================================"
printf 'PASS: both parallel two-GPU eval-resume smoke processes completed.\n'
printf '%s\n' "================================================================"
