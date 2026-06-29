#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

train_root="${TRAIN_ROOT:-exps/drifting_flux}"
eval_root="${EVAL_ROOT:-exps/drifting_flux_eval}"
gt_cache="${GT_CACHE:-data/audiocaps/test-features}"
num_steps="${NUM_STEPS:-1}"
cfg_strength="${CFG_STRENGTH:-4.5}"
use_rope="${USE_ROPE:-1}"
use_ema="${EVAL_USE_EMA:-1}"
test_entrypoint="${TEST_ENTRYPOINT:-drifting/flux/test.py}"
eval_script="${EVAL_SCRIPT:-drifting/scripts/eval_drifting_checkpoint.sh}"
summary_csv="${SUMMARY_CSV:-${eval_root}/latest_eval_summary.csv}"

if [[ ! -d "${train_root}" ]]; then
  echo "Missing training root: ${train_root}" >&2
  exit 1
fi

mkdir -p "${eval_root}"
if [[ ! -f "${summary_csv}" ]]; then
  printf 'exp_id,iteration,checkpoint,output,status\n' >"${summary_csv}"
fi

find_latest_raw_checkpoint() {
  local exp_dir="$1"
  local exp_id="$2"
  local latest_iteration="-1"
  local latest_path=""
  local path base prefix rest iteration

  prefix="${exp_id}_"
  for path in "${exp_dir}"/"${exp_id}"_*.pth; do
    [[ -f "${path}" ]] || continue
    base="$(basename "${path}")"
    rest="${base#"${prefix}"}"
    rest="${rest%.pth}"
    [[ "${rest}" =~ ^[0-9]+$ ]] || continue
    iteration=$((10#${rest}))
    if (( iteration > latest_iteration )); then
      latest_iteration="${iteration}"
      latest_path="${path}"
    fi
  done

  if [[ -n "${latest_path}" ]]; then
    printf '%s\t%s\n' "${latest_iteration}" "${latest_path}"
  fi
  return 0
}

found=0
failed=0

for exp_dir in "${train_root}"/*; do
  [[ -d "${exp_dir}" ]] || continue
  exp_id="$(basename "${exp_dir}")"
  latest="$(find_latest_raw_checkpoint "${exp_dir}" "${exp_id}")"
  if [[ -z "${latest}" ]]; then
    echo "Skipping ${exp_id}: no numeric checkpoint found in ${exp_dir}"
    continue
  fi

  found=1
  iteration="${latest%%$'\t'*}"
  raw_checkpoint="${latest#*$'\t'}"
  checkpoint="${raw_checkpoint}"
  ema_checkpoint="${exp_dir}/${exp_id}_${iteration}_ema.pth"
  if [[ "${use_ema}" == "1" && -f "${ema_checkpoint}" ]]; then
    checkpoint="${ema_checkpoint}"
  fi

  output_dir="${eval_root}/${exp_id}/it_$(printf '%08d' "${iteration}")"
  mkdir -p "${output_dir}"
  driver_log="${output_dir}/eval_driver.log"

  echo "================================================================"
  echo "Evaluating latest Flux checkpoint"
  echo "EXP_ID=${exp_id}"
  echo "ITERATION=${iteration}"
  echo "CHECKPOINT=${checkpoint}"
  echo "OUTPUT_PATH=${output_dir}"
  echo "GT_CACHE=${gt_cache}"
  echo "NUM_STEPS=${num_steps}"
  echo "CFG_STRENGTH=${cfg_strength}"
  echo "USE_ROPE=${use_rope}"
  echo "EVAL_USE_EMA=${use_ema}"
  echo "EVAL_SCRIPT=${eval_script}"
  echo "DRIVER_LOG=${driver_log}"
  echo "================================================================"

  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TEST_ENTRYPOINT="${test_entrypoint}" \
    MODEL_PATH="${checkpoint}" \
    OUTPUT_PATH="${output_dir}" \
    GT_CACHE="${gt_cache}" \
    NUM_STEPS="${num_steps}" \
    CFG_STRENGTH="${cfg_strength}" \
    USE_ROPE="${use_rope}" \
    bash "${eval_script}" >"${driver_log}" 2>&1; then
    printf '%s,%s,%s,%s,ok\n' "${exp_id}" "${iteration}" "${checkpoint}" "${output_dir}" >>"${summary_csv}"
    echo "Eval completed: ${exp_id} it=${iteration}"
  else
    printf '%s,%s,%s,%s,failed\n' "${exp_id}" "${iteration}" "${checkpoint}" "${output_dir}" >>"${summary_csv}"
    echo "Eval failed: ${exp_id} it=${iteration}; see ${driver_log}" >&2
    failed=1
  fi
done

if [[ "${found}" == "0" ]]; then
  echo "No exp_id directories with numeric checkpoints found under ${train_root}." >&2
  exit 1
fi

if [[ "${failed}" != "0" ]]; then
  exit 1
fi

echo "All latest Flux checkpoint evaluations completed."
echo "Summary: ${summary_csv}"
