#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES=0

exp_id="${EXP_ID:-fluxaudio_s_full_pre_distill_1step}"
iteration="${ITERATION:-0}"
output_root="${OUTPUT_ROOT:-exps/drifting_flux_eval}"
output_path="${OUTPUT_PATH:-${output_root}/${exp_id}/it_$(printf '%08d' "${iteration}")}"
driver_log="${output_path}/eval_driver.log"

model_path="${MODEL_PATH:-weights/fluxaudio_s_full.pth}"
gt_cache="${GT_CACHE:-data/audiocaps/test-features}"
eval_tsv="${EVAL_TSV:-sets/test-audiocaps.tsv}"
eval_npz_dir="${EVAL_NPZ_DIR:-data/audiocaps/test-npz-t5-clap}"
vae_weights="${VAE_WEIGHTS:-weights/v1-16.pth}"
vocoder_weights="${VOCODER_WEIGHTS:-weights/best_netG.pt}"
duration="${DURATION:-10}"
seed="${SEED:-42}"
num_steps="${NUM_STEPS:-1}"
cfg_strength="${CFG_STRENGTH:-4.5}"
use_rope="${USE_ROPE:-1}"

mkdir -p "${output_path}"

{
  echo "================================================================"
  echo "Evaluating pre-distillation FluxAudio-S-Full with one-step sampling"
  echo "EXP_ID=${exp_id}"
  echo "ITERATION=${iteration}"
  echo "MODEL_PATH=${model_path}"
  echo "OUTPUT_PATH=${output_path}"
  echo "GT_CACHE=${gt_cache}"
  echo "EVAL_TSV=${eval_tsv}"
  echo "EVAL_NPZ_DIR=${eval_npz_dir}"
  echo "VAE_WEIGHTS=${vae_weights}"
  echo "VOCODER_WEIGHTS=${vocoder_weights}"
  echo "DURATION=${duration}"
  echo "SEED=${seed}"
  echo "NUM_STEPS=${num_steps}"
  echo "CFG_STRENGTH=${cfg_strength}"
  echo "USE_ROPE=${use_rope}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "DRIVER_LOG=${driver_log}"
  echo "================================================================"
} | tee "${driver_log}"

TEST_ENTRYPOINT="${TEST_ENTRYPOINT:-drifting/flux/test.py}" \
MODEL_PATH="${model_path}" \
OUTPUT_PATH="${output_path}" \
GT_CACHE="${gt_cache}" \
EVAL_TSV="${eval_tsv}" \
EVAL_NPZ_DIR="${eval_npz_dir}" \
VAE_WEIGHTS="${vae_weights}" \
VOCODER_WEIGHTS="${vocoder_weights}" \
DURATION="${duration}" \
SEED="${seed}" \
NUM_STEPS="${num_steps}" \
CFG_STRENGTH="${cfg_strength}" \
USE_ROPE="${use_rope}" \
bash drifting/scripts/eval_drifting_checkpoint.sh 2>&1 | tee -a "${driver_log}"
