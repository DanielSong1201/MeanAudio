# DriftingAudio Setup and Usage

This document describes a minimal, hardware-agnostic setup for running the
DriftingAudio scripts in this repository. It covers environment setup, assets,
sanity checks, training, sweeps, checkpoint resume, and evaluation.

For the complete GPU/CUDA-focused Python environment, including pinned
PyTorch versions, Flux and Resonate validation, training, and AV-Benchmark
evaluation, see [`README_ENVIRONMENT.md`](README_ENVIRONMENT.md).

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

Use Python 3.11. The canonical GPU/CUDA setup, compatibility explanation, and
verification commands are in [`README_ENVIRONMENT.md`](README_ENVIRONMENT.md).
The current unified Flux and Resonate environment uses the repository-pinned
PyTorch 2.5.1 CUDA 12.4 wheels:

```bash
conda create -n meanaudio python=3.11 -y
conda activate meanaudio
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -r drifting/Resonate/requirements.txt
python -m pip install -e .
python -m pip check
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

Run the teacher evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash tst/phase0_01_eval_fluxaudio_teacher.sh
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

## 5. Flux Route

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

## 6. Drifting Tests

```bash
MODE=assets bash drifting/scripts/flux/test_flux.sh
MODE=loss bash drifting/scripts/flux/test_flux.sh
CUDA_VISIBLE_DEVICES=0 MODE=train-step BATCH_SIZE=2 bash drifting/scripts/flux/test_flux.sh
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

## 8. Auto Resume

Training auto-resumes by default.

Flux route checks:

```text
exps/drifting_flux/<exp_id>/
```

The preferred resume file is:

```text
<exp_id>_train_state_last.pth
```

It contains the student, EMA, AdamW optimizer, iteration, and sampler epoch.
It is atomically replaced at each save interval. Numeric checkpoints remain
available for evaluation:

```text
<exp_id>_<iteration>.pth
```

For experiments created before full training-state checkpoints were added, the
trainer falls back to the largest numeric checkpoint and matching EMA:

```text
<exp_id>_<iteration>_ema.pth
```

That first legacy resume initializes the optimizer fresh. After the next save,
future resumes restore the complete optimizer state.

Each base training script also holds:

```text
exps/drifting_flux/<exp_id>/.train.lock
```

A second process using the same `EXP_ID` exits instead of writing into the same
checkpoint and metrics files.

Skip the same-`EXP_ID` lookup and start fresh (when `RESUME_PATH` is also unset):

```bash
AUTO_RESUME=0 bash drifting/scripts/flux/train_flux_1x4090.sh
```

`ITERATIONS` is the final target iteration. If `60000.pth` exists and
`ITERATIONS=100000`, training runs from `60001` to `100000`.

### Explicit checkpoint branching

An explicit checkpoint can be resumed into a new `EXP_ID` while preserving the
source iteration. Recovery is keyed by `EXP_ID`: with the default
`AUTO_RESUME=1`, training first looks for a full state or numeric checkpoint in
`exps/drifting_flux/<EXP_ID>/`. `RESUME_PATH` is used only as a fallback when
that target experiment has no checkpoint yet. This makes the same branch command
safe to rerun: its first run starts from the source checkpoint, while later runs
continue the target experiment instead of branching from the source again.

Interfaces:

```text
RESUME_PATH=/path/to/checkpoint.pth
RESUME_ITERATION=40000       # optional when encoded in the filename or full state
RESUME_EMA_PATH=/path/to/ema.pth
RESET_OPTIMIZER=0            # 0=restore when available, 1=fresh AdamW
ITERATIONS=100000            # absolute target, not additional iterations
```

`RESET_OPTIMIZER` and `RESUME_EMA_PATH` apply to the explicit `RESUME_PATH`
fallback. Once the target `EXP_ID` has its own full training state, auto-resume
restores that experiment's optimizer and EMA regardless of the original branch
settings. To bypass the target lookup deliberately, set `AUTO_RESUME=0`; then an
explicit `RESUME_PATH` is selected directly, or the run starts from
`STUDENT_INIT` when no explicit path is supplied.

To preserve optimizer state from arbitrary historical iterations, enable
numbered full-state archives during the source run. The interval must be a
multiple of `SAVE_INTERVAL`:

```bash
SAVE_INTERVAL=1000 \
ARCHIVE_TRAIN_STATE_INTERVAL=10000 \
bash drifting/scripts/flux/train_flux_2x4090.sh
```

This keeps the normal rolling state and additionally writes:

```text
<exp_id>_10000_train_state.pth
<exp_id>_20000_train_state.pth
...
```

The default `ARCHIVE_TRAIN_STATE_INTERVAL=0` writes no numbered full states,
so storage and legacy save behavior remain unchanged. Numbered full states are
substantially larger than weights-only checkpoints because they include AdamW
moments; choose a coarse archive interval and monitor disk usage.

Example: branch from iteration 40000, remove the flow loss, keep optimizer and
EMA state, write into a new experiment, and continue at iteration 40001:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
EXP_ID=flux_from40k_tfd_only_keepopt \
RESUME_PATH=exps/drifting_flux/source_exp/source_exp_40000_train_state.pth \
RESUME_ITERATION=40000 \
RESET_OPTIMIZER=0 \
ITERATIONS=100000 \
LAMBDA_FLOW=0 \
LAMBDA_TFD=100 \
LAMBDA_ANCHOR=1 \
POOL_TOKENS=4 \
bash drifting/scripts/ablation/train_pool4_hybridpos.sh
```

Set `RESET_OPTIMIZER=1` to keep the same student/EMA weights and iteration but
start with a fresh AdamW optimizer using the new command-line configuration.
With `RESET_OPTIMIZER=0`, Adam moments and saved optimizer parameter groups are
retained. The current `LEARNING_RATE` is applied by the training loop at the
first resumed step, but changing optimizer-level settings such as weight decay
or Adam betas requires `RESET_OPTIMIZER=1`.

A weights-only checkpoint also supports iteration-preserving branching, but it
cannot restore optimizer state:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
EXP_ID=flux_from40k_tfd_only_freshopt \
RESUME_PATH=exps/drifting_flux/source_exp/source_exp_40000.pth \
RESUME_ITERATION=40000 \
RESUME_EMA_PATH=exps/drifting_flux/source_exp/source_exp_40000_ema.pth \
RESET_OPTIMIZER=1 \
ITERATIONS=100000 \
LAMBDA_FLOW=0 \
LAMBDA_TFD=100 \
LAMBDA_ANCHOR=1 \
POOL_TOKENS=4 \
bash drifting/scripts/ablation/train_pool4_hybridpos.sh
```

Both examples log `TRAIN_START ... resume_iteration=40000
start_iteration=40001`, and the first completed step (`40001`) is written to
the new experiment's `metrics.csv` even when it is not divisible by
`LOG_INTERVAL`.

## 9. Logs and Outputs

```text
exps/drifting_flux/<exp_id>/train.log
exps/drifting_flux/<exp_id>/metrics.csv
exps/drifting_flux/<exp_id>/eval_metrics.csv
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>.pth
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>_ema.pth
exps/drifting_flux/<exp_id>/<exp_id>_last.pth
exps/drifting_flux/<exp_id>/<exp_id>_ema_last.pth
exps/drifting_flux/<exp_id>/<exp_id>_train_state_last.pth
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>_train_state.pth  # optional archive
```

Training metrics are shown in the tqdm postfix. They are also written to
`metrics.csv`.

## 10. Training-Time Evaluation

Default:

```text
EVAL_INTERVAL=10000
```

```text
exps/drifting_flux_eval/<exp_id>/it_00010000/
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

Training-time evaluation uses the standard Hugging Face resolution configured
by the server environment. It does not pre-download models or redirect the Hub
cache into this repository. An evaluation failure is logged under the
iteration's `eval_driver.log` and does not terminate the training loop. Set
`EVAL_FAILURE_FATAL=1` only when an eval failure should stop training.

## 11. Manual Evaluation

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

## 12. Evaluate Latest Flux Checkpoints

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

## 13. Sweeps

The sweep directory contains three scripts:

```text
sweep_flux_4gpu_lr1e6_tfd1_anchor1_flow005_200k.sh
sweep_flux_4gpu_lr1e6_tfd100_anchor1_flow01_200k.sh
parallel_flux_2x2gpu_lr1e6_tfd1_tfd100_200k.sh
```

The first two run the retained experiments independently on four GPUs. The
third runs them concurrently as two two-GPU DDP jobs:

```bash
bash drifting/scripts/flux/sweeps/parallel_flux_2x2gpu_lr1e6_tfd1_tfd100_200k.sh
```

Both experiments use `lr=1e-6`, 1000 warmup steps, and the same hybrid set of
four prompt-matched positives. Their loss configurations are:

```text
tfd=1,   anchor=1, flow=0.05
tfd=100, anchor=1, flow=0.1
```

All distributed sweep entries set `DDP_TIMEOUT_MINUTES=180`, which configures
the PyTorch NCCL process-group timeout to 180 minutes.

## 14. Troubleshooting

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

## 15. Minimal First Run

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
