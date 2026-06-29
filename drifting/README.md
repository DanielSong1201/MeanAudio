# DriftingAudio Setup and Usage

This document describes a minimal, hardware-agnostic setup for running the
DriftingAudio scripts in this repository. It covers environment setup, assets,
sanity checks, training, sweeps, checkpoint resume, and evaluation.

Run all commands from the MeanAudio repository root unless noted otherwise:

```bash
cd /path/to/MeanAudio
```

## 1. Get the Code

Fresh clone:

```bash
git clone git@github.com:DanielSong1201/MeanAudio.git
cd MeanAudio
git switch drifting
```

Existing checkout:

```bash
cd /path/to/MeanAudio
git fetch origin
git switch drifting
git pull --ff-only origin drifting
```

If the server checkout has local edits:

```bash
git stash push -u -m "server local changes before drifting update"
git pull --ff-only origin drifting
```

## 2. Create the Python Environment

Use Python 3.11. Install the PyTorch build matching your CUDA driver and cluster
policy. The command below follows the upstream README CUDA 11.8 example:

```bash
conda create -n meanaudio python=3.11 -y
conda activate meanaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -e .
```

Check the environment:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(idx, torch.cuda.get_device_name(idx))
PY
```

Full evaluation uses:

```text
av-benchmark/evaluate.py
```

Place/install `av-benchmark` under the repository root so this path exists:

```text
MeanAudio/
  av-benchmark/
    evaluate.py
```

Training can start without this path, but full evaluation will fail.

## 3. Prepare Weights and Data

### 3.1 Download Required Weights

The Phase-0 helper downloads the required public weights from Hugging Face:

```bash
bash tst/phase0_00_download_assets.sh
```

If authentication or higher rate limits are needed:

```bash
HF_TOKEN=your_token bash tst/phase0_00_download_assets.sh
```

Required default assets:

```text
weights/fluxaudio_s_full.pth
weights/meanaudio_s_full.pth
weights/v1-16.pth
weights/best_netG.pt
sets/latent_mean.pt
sets/latent_std.pt
```

Optional weights and empty-string condition tensors:

```bash
INCLUDE_OPTIONAL_WEIGHTS=1 bash tst/phase0_00_download_assets.sh
```

### 3.2 Prepare AudioCaps Features

Default training/evaluation paths are defined in:

```text
config/data/t5_clap.yaml
```

The default AudioCaps layout is:

```text
data/audiocaps/train-memmap.tsv
data/audiocaps/train-npz-t5-clap/
data/audiocaps/val-memmap.tsv
data/audiocaps/val-npz-t5-clap/
data/audiocaps/test-memmap.tsv
data/audiocaps/test-npz-t5-clap/
data/audiocaps/test-features/
```

If your data lives elsewhere, update `config/data/t5_clap.yaml`.

## 4. Sanity Checks

Check assets, data files, and latent statistics:

```bash
bash tst/phase0_00_check_assets.sh
```

Run the teacher and baseline evaluations:

```bash
CUDA_VISIBLE_DEVICES=0 bash tst/phase0_01_eval_fluxaudio_teacher.sh
CUDA_VISIBLE_DEVICES=0 bash tst/phase0_02_eval_meanaudio_baseline.sh
```

Or run all Phase-0 checks:

```bash
CUDA_VISIBLE_DEVICES=0 bash tst/phase0_run_all.sh
```

Check teacher feature extraction and gradient flow:

```bash
CUDA_VISIBLE_DEVICES=0 bash tst/phase1_00_check_teacher_features.sh
```

Check the synthetic TFD loss:

```bash
bash tst/phase2_00_check_tfd_loss.sh
```

CUDA variant:

```bash
DEVICE=cuda DTYPE=bfloat16 bash tst/phase2_00_check_tfd_loss.sh
```

## 5. Experiment Routes

### Flux Route

Location:

```text
drifting/flux/
```

Model relation:

```text
FluxAudio-S-Full teacher -> FluxAudio-S one-step student
```

Loss:

```text
total_loss = lambda_flow * flow_loss
           + lambda_tfd * drifting_loss
           + lambda_anchor * anchor_loss
```

Default teacher feature layers:

```text
joint_3,fused_3,fused_7
```

### Mean Route

Location:

```text
drifting/mean/
```

Model relation:

```text
FluxAudio-S-Full teacher -> MeanAudio-S one-step student
```

This route keeps the MeanAudio/MeanFlow student while adding TFD and anchor
regularization.

## 6. Drifting Tests

Flux route:

```bash
MODE=assets bash drifting/scripts/flux/test_flux.sh
MODE=loss bash drifting/scripts/flux/test_flux.sh
CUDA_VISIBLE_DEVICES=0 MODE=train-step BATCH_SIZE=2 bash drifting/scripts/flux/test_flux.sh
```

Mean route:

```bash
MODE=loss bash drifting/scripts/mean/test_mean.sh
MODE=teacher bash drifting/scripts/mean/test_mean.sh
```

Compatibility route:

```bash
bash drifting/scripts/test_drifting.sh
```

## 7. Flux Training

Single visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash drifting/scripts/flux/train_flux_1x4090.sh
```

The script name contains `4090` for historical reasons only. It is not limited
to NVIDIA 4090 GPUs.

Debug run:

```bash
CUDA_VISIBLE_DEVICES=0 \
EXP_ID=flux_debug \
BATCH_SIZE=2 \
ITERATIONS=100 \
EVAL_INTERVAL=0 \
LOG_INTERVAL=10 \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

Two-process DDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash drifting/scripts/flux/train_flux_2x4090.sh
```

Override process count if needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash drifting/scripts/flux/train_flux_2x4090.sh
```

`BATCH_SIZE` is per process/GPU.

Common overrides:

```bash
EXP_ID=your_exp
ITERATIONS=100000
BATCH_SIZE=4
LEARNING_RATE=5e-5
SAVE_INTERVAL=1000
EVAL_INTERVAL=10000
EVAL_NUM_STEPS=1
EVAL_CFG_STRENGTH=4.5
LAMBDA_FLOW=0.3
LAMBDA_TFD=1.0
LAMBDA_ANCHOR=1.0
EMA_DECAY=0.9999
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
EXP_ID=flux_flow03_tfd1_anchor1 \
ITERATIONS=200000 \
LAMBDA_FLOW=0.3 \
LAMBDA_TFD=1.0 \
LAMBDA_ANCHOR=1.0 \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

## 8. Mean Training

Single visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash drifting/scripts/mean/train_mean_1x4090.sh
```

Debug run:

```bash
CUDA_VISIBLE_DEVICES=0 \
EXP_ID=mean_debug \
BATCH_SIZE=2 \
ITERATIONS=100 \
EVAL_INTERVAL=0 \
bash drifting/scripts/mean/train_mean_1x4090.sh
```

Compatibility entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash drifting/scripts/train_drifting_1x4090.sh
```

## 9. Auto Resume

Training auto-resumes by default.

Flux route checks:

```text
exps/drifting_flux/<exp_id>/
```

Mean route checks:

```text
exps/drifting/<exp_id>/
```

If numeric checkpoints exist:

```text
<exp_id>_<iteration>.pth
```

the trainer loads the largest iteration and continues from `iteration + 1`.
If the matching EMA checkpoint exists, it is loaded too:

```text
<exp_id>_<iteration>_ema.pth
```

These checkpoints are weights-only. Optimizer state is initialized fresh after
resume.

Force a fresh run:

```bash
AUTO_RESUME=0 bash drifting/scripts/flux/train_flux_1x4090.sh
```

`ITERATIONS` is the final target iteration. If `60000.pth` exists and
`ITERATIONS=100000`, training runs from `60001` to `100000`.

## 10. Logs and Outputs

Flux route:

```text
exps/drifting_flux/<exp_id>/train.log
exps/drifting_flux/<exp_id>/metrics.csv
exps/drifting_flux/<exp_id>/eval_metrics.csv
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>.pth
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>_ema.pth
exps/drifting_flux/<exp_id>/<exp_id>_last.pth
exps/drifting_flux/<exp_id>/<exp_id>_ema_last.pth
```

Mean route:

```text
exps/drifting/<exp_id>/train.log
exps/drifting/<exp_id>/metrics.csv
exps/drifting/<exp_id>/eval_metrics.csv
```

Training metrics are shown in the tqdm postfix. They are also written to
`metrics.csv`.

## 11. Training-Time Evaluation

Default:

```text
EVAL_INTERVAL=10000
```

Flux eval output:

```text
exps/drifting_flux_eval/<exp_id>/it_00010000/
  audio/
  cache/
  evaluate.log
  eval_driver.log
```

Mean eval output:

```text
exps/drifting_eval/<exp_id>/it_00010000/
  audio/
  cache/
  evaluate.log
  eval_driver.log
```

Disable training-time evaluation:

```bash
EVAL_INTERVAL=0 bash drifting/scripts/flux/train_flux_1x4090.sh
```

By default, training-time eval uses EMA checkpoints. To evaluate raw weights
during training, call the Python entrypoint with `--eval-raw`.

## 12. Manual Evaluation

Flux route:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODE=eval \
MODEL_PATH=exps/drifting_flux/flux_debug/flux_debug_ema_last.pth \
OUTPUT_PATH=exps/drifting_flux_eval/manual_flux_debug \
bash drifting/scripts/flux/test_flux.sh
```

Generic full eval wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEST_ENTRYPOINT=drifting/flux/test.py \
MODEL_PATH=exps/drifting_flux/flux_debug/flux_debug_ema_last.pth \
OUTPUT_PATH=exps/drifting_flux_eval/manual_flux_debug \
GT_CACHE=data/audiocaps/test-features \
NUM_STEPS=1 \
CFG_STRENGTH=4.5 \
USE_ROPE=1 \
bash drifting/scripts/eval_drifting_checkpoint.sh
```

## 13. Evaluate Latest Flux Checkpoints

Scan all experiment directories under `exps/drifting_flux/`, select the largest
numeric checkpoint in each directory, and run the same full eval path used by
training-time evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash drifting/scripts/flux/eval_latest_flux_checkpoints.sh
```

Default output:

```text
exps/drifting_flux_eval/<exp_id>/it_000xxxxx/
exps/drifting_flux_eval/latest_eval_summary.csv
```

Use raw checkpoints instead of EMA:

```bash
EVAL_USE_EMA=0 bash drifting/scripts/flux/eval_latest_flux_checkpoints.sh
```

This script uses:

```text
drifting/scripts/eval_drifting_checkpoint.sh
  -> drifting/flux/test.py
  -> eval.py --variant fluxaudio_s
  -> av-benchmark/evaluate.py
```

## 14. Sweeps

### Original Three 200K Sweeps

Run all sequentially:

```bash
bash drifting/scripts/flux/sweeps/run_all_flux_sweeps_200k.sh
```

Run individually:

```bash
bash drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor05_200k.sh
bash drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor1_flow03_200k.sh
bash drifting/scripts/flux/sweeps/sweep_flux_lr1e5_tfd1_anchor1_flow05_200k.sh
```

Configurations:

```text
flux_sweep_tfd1_anchor05_200k:
  LR=5e-5, flow=1.0, tfd=1.0, anchor=0.5

flux_sweep_tfd1_anchor1_flow03_200k:
  LR=5e-5, flow=0.3, tfd=1.0, anchor=1.0

flux_sweep_lr1e5_tfd1_anchor1_flow05_200k:
  LR=1e-5, flow=0.5, tfd=1.0, anchor=1.0
```

### Parallel Three-Sweep Launcher

Use three independent visible GPUs/processes:

```bash
bash drifting/scripts/flux/sweeps/parallel_flux_sweeps_3x1gpu.sh
```

Defaults:

```text
GPU0 -> tfd1_anchor05
GPU1 -> tfd1_anchor1_flow03
GPU2 -> lr1e5_tfd1_anchor1_flow05
```

Override device IDs:

```bash
GPU0=0 GPU1=2 GPU2=3 bash drifting/scripts/flux/sweeps/parallel_flux_sweeps_3x1gpu.sh
```

Live tqdm mode is enabled by default. To use prefixed line logs instead:

```bash
LIVE_TQDM=0 bash drifting/scripts/flux/sweeps/parallel_flux_sweeps_3x1gpu.sh
```

### Flow Ablation

Run `flow=0.3/0.1/0.0` with `tfd=1.0` and `anchor=1.0`:

```bash
bash drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor1_flow_ablation_200k.sh
```

Experiment IDs:

```text
flow=0.3 -> flux_sweep_tfd1_anchor1_flow03_200k
flow=0.1 -> flux_sweep_tfd1_anchor1_flow01_200k
flow=0.0 -> flux_sweep_tfd1_anchor1_flow00_200k
```

The `flow=0.3` experiment ID intentionally stays unchanged.

## 15. Troubleshooting

### `python: command not found`

```bash
conda activate meanaudio
which python
```

### Missing weights

```bash
bash tst/phase0_00_download_assets.sh
bash tst/phase0_00_check_assets.sh
```

### Missing data

```bash
python - <<'PY'
from pathlib import Path
for p in [
    "data/audiocaps/train-memmap.tsv",
    "data/audiocaps/train-npz-t5-clap/0.npz",
    "data/audiocaps/test-features",
]:
    print(p, Path(p).exists())
PY
```

### Full eval fails

Check:

```text
<eval_output>/eval_driver.log
<eval_output>/evaluate.log
```

Verify:

```bash
test -f av-benchmark/evaluate.py && echo ok
```

### Auto-resume starts from an old checkpoint

Use a new `EXP_ID`, remove old experiment files, or disable auto-resume:

```bash
AUTO_RESUME=0 EXP_ID=your_exp bash drifting/scripts/flux/train_flux_1x4090.sh
```

## 16. Minimal First Run

```bash
conda activate meanaudio
pip install -e .

bash tst/phase0_00_download_assets.sh
bash tst/phase0_00_check_assets.sh
CUDA_VISIBLE_DEVICES=0 bash tst/phase1_00_check_teacher_features.sh
DEVICE=cuda DTYPE=bfloat16 bash tst/phase2_00_check_tfd_loss.sh

MODE=assets bash drifting/scripts/flux/test_flux.sh
MODE=loss bash drifting/scripts/flux/test_flux.sh
CUDA_VISIBLE_DEVICES=0 MODE=train-step BATCH_SIZE=2 bash drifting/scripts/flux/test_flux.sh

CUDA_VISIBLE_DEVICES=0 EXP_ID=flux_debug ITERATIONS=100 EVAL_INTERVAL=0 \
  bash drifting/scripts/flux/train_flux_1x4090.sh
```

After this succeeds, start a full training run or one of the sweep scripts.
