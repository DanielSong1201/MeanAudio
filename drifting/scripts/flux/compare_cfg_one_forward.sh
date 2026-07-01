#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

exp_id="${EXP_ID:-flux_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k_4gpu}"
train_root="${TRAIN_ROOT:-exps/drifting_flux}"
eval_root="${EVAL_ROOT:-exps/drifting_flux_eval_cfg_comparison}"
model_path="${MODEL_PATH:-${train_root}/${exp_id}/${exp_id}_ema_last.pth}"
gt_cache="${GT_CACHE:-data/audiocaps/test-features}"
num_steps="${NUM_STEPS:-1}"
cfg_off="${CFG_OFF_STRENGTH:-0.0}"
cfg_on="${CFG_ON_STRENGTH:-4.5}"
use_rope="${USE_ROPE:-1}"
eval_script="${EVAL_SCRIPT:-drifting/scripts/eval_drifting_checkpoint.sh}"

if [[ ! -f "${model_path}" ]]; then
  raw_model_path="${train_root}/${exp_id}/${exp_id}_last.pth"
  if [[ -f "${raw_model_path}" ]]; then
    model_path="${raw_model_path}"
  else
    printf 'Missing EMA and raw final checkpoints for %s under %s\n' "${exp_id}" "${train_root}" >&2
    exit 1
  fi
fi

format_cfg_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

run_eval() {
  local label="$1"
  local cfg_strength="$2"
  local output_path="${eval_root}/${exp_id}/${label}"

  printf 'cfg_comparison label=%s cfg_strength=%s model_path=%s output=%s\n' \
    "${label}" "${cfg_strength}" "${model_path}" "${output_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  TEST_ENTRYPOINT="${TEST_ENTRYPOINT:-drifting/flux/test.py}" \
  MODEL_PATH="${model_path}" \
  OUTPUT_PATH="${output_path}" \
  GT_CACHE="${gt_cache}" \
  NUM_STEPS="${num_steps}" \
  CFG_STRENGTH="${cfg_strength}" \
  USE_ROPE="${use_rope}" \
  SEED="${SEED:-42}" \
  bash "${eval_script}"
}

mkdir -p "${eval_root}/${exp_id}"
summary_path="${eval_root}/${exp_id}/comparison.csv"
printf 'label,cfg_strength,model_forwards_per_step,checkpoint,output,evaluate_log\n' >"${summary_path}"

off_label="cfg_$(format_cfg_tag "${cfg_off}")_one_forward"
on_label="cfg_$(format_cfg_tag "${cfg_on}")_two_forwards"

run_eval "${off_label}" "${cfg_off}"
printf '%s,%s,1,%s,%s,%s\n' \
  "${off_label}" "${cfg_off}" "${model_path}" \
  "${eval_root}/${exp_id}/${off_label}" \
  "${eval_root}/${exp_id}/${off_label}/evaluate.log" >>"${summary_path}"

run_eval "${on_label}" "${cfg_on}"
printf '%s,%s,2,%s,%s,%s\n' \
  "${on_label}" "${cfg_on}" "${model_path}" \
  "${eval_root}/${exp_id}/${on_label}" \
  "${eval_root}/${exp_id}/${on_label}/evaluate.log" >>"${summary_path}"

printf 'CFG comparison complete: %s\n' "${summary_path}"
