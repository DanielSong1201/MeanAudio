#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ckpt_root="${DRIFTING_CKPT_DIR:-drifting/ckpts}"
python_bin="${PYTHON:-python}"
prepare_ckpts="${PREPARE_CKPTS:-1}"

if [[ "${prepare_ckpts}" == "1" ]]; then
  printf '%s | INFO | Downloading any missing test assets before offline validation\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  DRIFTING_CKPT_DIR="${ckpt_root}" PYTHON="${python_bin}" \
    bash drifting/scripts/prepare_hf_ckpts.sh
fi

export HF_HOME="${ckpt_root}/huggingface"
export HF_HUB_CACHE="${ckpt_root}/huggingface/hub"
export TRANSFORMERS_CACHE="${ckpt_root}/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

printf '%s | INFO | Testing local Hugging Face assets in offline mode\n' "$(date '+%Y-%m-%d %H:%M:%S')"
"${python_bin}" drifting/scripts/test_local_hf_ckpts.py \
  --repo-root . \
  --ckpt-root "${ckpt_root}"
