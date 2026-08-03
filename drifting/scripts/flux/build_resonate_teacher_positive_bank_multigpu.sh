#!/usr/bin/env bash
set -euo pipefail

# Generic entry point. The implementation remains in the historical 4gpu-named
# launcher so existing server commands continue to work unchanged.
script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "${script_dir}/build_resonate_teacher_positive_bank_4gpu.sh" "$@"
