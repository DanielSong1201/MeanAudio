#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash tst/phase0_00_download_assets.sh
bash tst/phase0_00_check_assets.sh
bash tst/phase0_01_eval_fluxaudio_teacher.sh
bash tst/phase0_02_eval_meanaudio_baseline.sh

echo "[done] Phase-0 checks and baseline evaluations completed."
