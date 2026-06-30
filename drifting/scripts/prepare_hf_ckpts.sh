#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ckpt_root="${DRIFTING_CKPT_DIR:-drifting/ckpts}"
python_bin="${PYTHON:-python}"

printf '%s | INFO | Preparing Hugging Face assets in %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${ckpt_root}"

HF_HOME="${ckpt_root}/huggingface" \
HF_HUB_CACHE="${ckpt_root}/huggingface/hub" \
TRANSFORMERS_CACHE="${ckpt_root}/huggingface/transformers" \
HF_HUB_OFFLINE=0 \
TRANSFORMERS_OFFLINE=0 \
"${python_bin}" drifting/scripts/prepare_hf_ckpts.py \
  --repo-root . \
  --ckpt-root "${ckpt_root}"

printf '%s | INFO | Hugging Face assets are ready in %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${ckpt_root}"
