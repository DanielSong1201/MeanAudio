# DriftingAudio GPU/CUDA Environment and End-to-End Workflow

This guide describes the recommended Linux environment for the `drifting`
branch, including FluxAudio TFD and Resonate TFD training, validation,
checkpoint generation, and AV-Benchmark evaluation.

Run every command from the MeanAudio repository root unless a section says
otherwise:

```bash
cd /path/to/MeanAudio
git switch drifting
git pull --ff-only origin drifting
```

## 1. Supported environment

The unified environment used by the current scripts is:

```text
Linux x86_64
Python 3.11
NVIDIA GPU
PyTorch 2.5.1+cu124
torchvision 0.20.1+cu124
torchaudio 2.5.1+cu124
```

This combination supports both routes:

- FluxAudio TFD requires `torch>=2.5.1` through `pyproject.toml`.
- Resonate pins the complete CUDA stack in
  `drifting/Resonate/requirements.txt`.

Do not mix `torch 2.5.1` with `torchvision 0.27.1`; they use incompatible
binary interfaces.

### Driver CUDA versus PyTorch CUDA

Check the NVIDIA driver before creating the environment:

```bash
nvidia-smi
```

The `CUDA Version` printed by `nvidia-smi` is the newest CUDA runtime supported
by the installed driver. It is not necessarily the CUDA toolkit used by
PyTorch. The `+cu124` PyTorch wheels include their CUDA 12.4 runtime libraries;
a separate CUDA 12.4 toolkit or `nvcc` is normally unnecessary unless compiling
custom CUDA extensions.

The authoritative runtime check is:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is false, fix the driver, container GPU
passthrough, or PyTorch installation before proceeding.

## 2. Create the Conda environment

```bash
conda create -n meanaudio python=3.11 -y
conda activate meanaudio

python -m pip install --upgrade pip wheel
conda install -c conda-forge "ffmpeg<7" libsndfile -y
```

Install the repository's pinned CUDA packages and Resonate runtime
dependencies first:

```bash
python -m pip install --no-cache-dir \
  -r drifting/Resonate/requirements.txt
```

Then install MeanAudio and its Flux dependencies:

```bash
python -m pip install -e .
```

The editable installation makes `meanaudio` importable while preserving the
working-tree source files. Flux launchers also add the repository root to
`PYTHONPATH`, so imports such as `drifting.eval_helpers` work under `torchrun`.

To prevent packages from `~/.local` contaminating the Conda environment, this
is recommended on shared servers:

```bash
export PYTHONNOUSERSITE=1
```

Check dependency consistency:

```bash
python -m pip check
```

## 3. Verify Python, CUDA, GPU, and repository imports

```bash
which python
python --version

python - <<'PY'
import sys
import torch
import torchvision
import torchaudio

print("python", sys.executable)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("nccl_available", torch.distributed.is_nccl_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, "capability", f"{props.major}.{props.minor}")
    x = torch.ones(16, device=f"cuda:{index}")
    print("  tensor_sum", x.sum().item())
PY

python - <<'PY'
import drifting
import drifting.eval_helpers
import meanaudio

print("drifting", drifting.__file__)
print("eval_helpers", drifting.eval_helpers.__file__)
print("meanaudio", meanaudio.__file__)
PY
```

For a two-GPU job, confirm that both selected devices are visible:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -c "import torch; assert torch.cuda.device_count() == 2; print([torch.cuda.get_device_name(i) for i in range(2)])"
```

`CUDA_VISIBLE_DEVICES=2,3` remaps physical GPUs 2 and 3 to process-local CUDA
devices 0 and 1. `NPROC_PER_NODE` must equal the number of visible GPUs for the
DDP launchers.

## 4. Install AV-Benchmark

Training can run without AV-Benchmark only when periodic evaluation is
disabled. Complete validation and evaluation require a checkout named exactly
`av-benchmark` at the MeanAudio repository root:

```bash
git clone https://github.com/hkchengrex/av-benchmark.git
python -m pip install -e av-benchmark
test -f av-benchmark/evaluate.py
python -m pip check
```

The first benchmark run may populate Hugging Face and PyTorch caches. Keep
these caches outside the Git repository on shared servers if necessary:

```bash
export HF_HOME=/path/to/cache/huggingface
export TORCH_HOME=/path/to/cache/torch
```

Resonate's asset bootstrap obtains its explicitly required benchmark weights:

```bash
ASSET_DOWNLOAD_GPU=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

## 5. Prepare Flux assets and data

Download the public MeanAudio/FluxAudio weights:

```bash
bash tst/phase0_00_download_assets.sh
```

The Flux route requires at least:

```text
weights/fluxaudio_s_full.pth
weights/v1-16.pth
weights/best_netG.pt
sets/latent_mean.pt
sets/latent_std.pt
```

AudioCaps training and evaluation use the paths in
`config/data/t5_clap.yaml`:

```text
data/audiocaps/train-memmap.tsv
data/audiocaps/train-npz-t5-clap/
data/audiocaps/val-memmap.tsv
data/audiocaps/val-npz-t5-clap/
data/audiocaps/test-memmap.tsv
data/audiocaps/test-npz-t5-clap/
data/audiocaps/test-features/
```

Validate the assets and configured paths:

```bash
bash tst/phase0_00_check_assets.sh
```

The hybrid-positive Flux experiments additionally require:

```text
data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6/complete.json
```

The Flux-positive ablation creates a separate bank when it is missing:

```text
data/audiocaps/train-teacher-positives-fluxaudio-s-full-25step-cfg6/
```

It never overwrites the MeanAudio-L bank.

## 6. Validate the Flux environment before training

Run the checks in increasing order of cost:

```bash
# Files, checkpoints, and data configuration.
bash tst/phase0_00_check_assets.sh

# Teacher and one-step baseline generation plus AV-Benchmark.
CUDA_VISIBLE_DEVICES=0 bash tst/phase0_run_all.sh

# Frozen teacher features and gradient flow to the input latent.
CUDA_VISIBLE_DEVICES=0 bash tst/phase1_00_check_teacher_features.sh

# TFD and anchor loss on CUDA.
DEVICE=cuda DTYPE=bfloat16 bash tst/phase2_00_check_tfd_loss.sh

# Flux-specific assets, loss, and one optimization step.
MODE=assets bash drifting/scripts/flux/test_flux.sh
MODE=loss bash drifting/scripts/flux/test_flux.sh
CUDA_VISIBLE_DEVICES=0 MODE=train-step BATCH_SIZE=2 \
  bash drifting/scripts/flux/test_flux.sh
```

Do not start a long DDP job until all checks above pass.

## 7. Flux smoke training and full training

Run a short single-GPU smoke test without periodic AV-Benchmark first:

```bash
CUDA_VISIBLE_DEVICES=0 \
EXP_ID=flux_environment_smoke \
ITERATIONS=100 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
LOG_INTERVAL=10 \
SAVE_INTERVAL=50 \
EVAL_INTERVAL=0 \
AUTO_RESUME=0 \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

Two-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash drifting/scripts/flux/train_flux_2x4090.sh
```

Current two-GPU TFD sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash drifting/scripts/flux/sweeps/sweep_flux_2gpu_lr1e6_tfd100_anchor1_flow005_200k.sh
```

Current pooling/positive-source ablations should first be run for 50k
iterations:

```bash
ITERATIONS=50000 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash drifting/scripts/ablation/train_pool64_hybridpos_control.sh

ITERATIONS=50000 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash drifting/scripts/ablation/train_pool4_hybridpos.sh

ITERATIONS=50000 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash drifting/scripts/ablation/train_pool4_realpos.sh
```

`BATCH_SIZE` is per GPU. The effective batch size is
`BATCH_SIZE * NPROC_PER_NODE`; `SAMPLES_PER_CONDITION` expands the number of
latents processed by the model inside each condition batch.

Flux training auto-resumes by default and treats `EXP_ID` as the experiment
identity. It first restores the target `EXP_ID`; only when that experiment has
no checkpoint does it use `RESUME_PATH` as a branch source. Use a new `EXP_ID`
and `AUTO_RESUME=0` for a deliberately fresh run.

To branch from a specific checkpoint into a new experiment while preserving
its iteration, use `RESUME_PATH`, optional `RESUME_ITERATION` and
`RESUME_EMA_PATH`, and `RESET_OPTIMIZER=0|1`. Full examples, including a 40k
checkpoint continued as iteration 40001 with `LAMBDA_FLOW=0`, are documented
in the explicit checkpoint branching section of `drifting/README.md`.

## 8. Flux validation and evaluation

Periodic evaluation defaults to every 10k optimizer iterations:

```text
SAVE_INTERVAL=1000
EVAL_INTERVAL=10000
EVAL_NUM_STEPS=1
EVAL_CFG_STRENGTH=4.5
EVAL_GT_CACHE=data/audiocaps/test-features
```

Training outputs:

```text
exps/drifting_flux/<exp_id>/metrics.csv
exps/drifting_flux/<exp_id>/eval_metrics.csv
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>.pth
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>_ema.pth
exps/drifting_flux/<exp_id>/<exp_id>_train_state_last.pth
```

Evaluation outputs:

```text
exps/drifting_flux_eval/<exp_id>/it_XXXXXXXX/
  audio/
  cache/
  evaluate.log
  eval_driver.log
```

Manually evaluate one checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
TEST_ENTRYPOINT=drifting/flux/test.py \
MODEL_PATH=exps/drifting_flux/<exp_id>/<checkpoint>_ema.pth \
OUTPUT_PATH=exps/drifting_flux_eval/<exp_id>/manual \
GT_CACHE=data/audiocaps/test-features \
NUM_STEPS=1 \
CFG_STRENGTH=4.5 \
USE_ROPE=1 \
bash drifting/scripts/eval_drifting_checkpoint.sh
```

Evaluate the latest numeric checkpoint of every Flux experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash drifting/scripts/flux/eval_latest_flux_checkpoints.sh
```

Successful full evaluation produces audio, prediction caches, and AV-Benchmark
metrics. Inspect both `evaluate.log` and `eval_driver.log` when it fails.

## 9. Prepare and validate Resonate

Prepare model and evaluation assets:

```bash
ASSET_DOWNLOAD_GPU=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

The Resonate dataset configuration is:

```text
config/data/resonate_flant5_44k.yaml
```

Check preprocessing and teacher positives:

```bash
bash drifting/scripts/resonate/check_preprocessing_complete.sh
```

Use a deeper sample inspection when preparing a new server:

```bash
DEEP=1 SAMPLE_COUNT=64 \
bash drifting/scripts/resonate/check_preprocessing_complete.sh
```

Validate model construction, checkpoint loading, features, and gradients:

```bash
DEVICE=cuda \
bash drifting/scripts/resonate/validate_phase_0_3.sh
```

Run the complete three-iteration train/save/eval/resume smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
bash drifting/scripts/resonate/smoke_train_eval.sh
```

For a generation-only smoke test without benchmark metrics:

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
EVAL_SKIP_AV_BENCHMARK=1 \
bash drifting/scripts/resonate/smoke_train_eval.sh
```

## 10. Resonate training and evaluation

Base training entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash drifting/scripts/resonate/train_tfd100_anchor1_flow01_4gpu.sh
```

Feature-layer ablation launchers are under
`drifting/scripts/resonate/`. The two-GPU launcher runs two independent
single-GPU experiments and exposes `GPU0`, `GPU1`, `BATCH_SIZE`, `SAVE_ITS`,
and `EMA_DEVICE` near the top:

```bash
GPU0=0 GPU1=1 \
bash drifting/scripts/resonate/parallel_tfd100_feature_layer_ablation_30k_2x1gpu.sh
```

Evaluate the released Resonate teacher at one and 25 steps:

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_STEPS_LIST="1 25" \
bash drifting/scripts/resonate/test_resonate_avbench.sh
```

Evaluate a trained checkpoint by overriding `MODEL_PATH` and `RUN_ID`:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=exps/drifting_resonate/<exp_id>/<checkpoint>.pth \
RUN_ID=<exp_id>_avbench \
NUM_STEPS_LIST="1" \
bash drifting/scripts/resonate/test_resonate_avbench.sh
```

The summary is written to:

```text
exps/drifting_resonate_eval/<run_id>/avbench_summary.csv
```

Each evaluated setting must also contain:

```text
cache/output_metrics.json
```

## 11. Recommended server acceptance sequence

Use this order for a newly configured GPU server:

1. Confirm `nvidia-smi` works inside the same shell/container.
2. Create and activate the Conda environment.
3. Install the pinned CUDA requirements, MeanAudio, and AV-Benchmark.
4. Run the Python/CUDA/import checks from Section 3.
5. Prepare weights, processed data, positive banks, and GT caches.
6. Run Flux Phase 0-2 checks and the Flux one-step test.
7. Run the 100-iteration Flux smoke training.
8. Run a manual Flux checkpoint evaluation and confirm metrics are generated.
9. Run Resonate preprocessing and Phase 0-3 checks if using that route.
10. Run the Resonate train/save/eval/resume smoke test.
11. Only then launch 30k, 50k, or 200k experiments.

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'drifting'`

Pull a revision containing the Flux launcher `PYTHONPATH` fix, activate the
environment, and run from the repository root:

```bash
git pull --ff-only origin drifting
conda activate meanaudio
python -c "import drifting, drifting.eval_helpers; print(drifting.__file__)"
```

Temporary diagnosis only:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### CUDA is unavailable

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If the installed torch version has no `+cu...` suffix or reports
`torch.version.cuda=None`, reinstall from the CUDA requirements file.

### Torch/torchvision binary mismatch

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-cache-dir \
  -r drifting/Resonate/requirements.txt
python -m pip check
```

### CUDA out of memory

Reduce `BATCH_SIZE` first, then `SAMPLES_PER_CONDITION`, and finally the number
of feature layers or pooled tokens. `BATCH_SIZE` is per GPU.

### NCCL timeout or worker hang

Inspect logs from every rank. Then retry with fewer workers and the conservative
settings already used by the Flux launchers:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export DDP_TIMEOUT_MINUTES=180
```

Do not assume the rank-0 log contains the first failing stack trace.

### AV-Benchmark is missing

```bash
test -f av-benchmark/evaluate.py
python -c "import torch, librosa, soundfile; print('benchmark runtime imports ok')"
```

If generation succeeds but metrics fail, inspect the evaluation log and verify
that `data/audiocaps/test-features/` contains the required GT caches.

### A failed job leaves `.train.lock`

The file itself may remain, but the operating-system lock is released when the
process exits. Do not delete it merely because the pathname exists. A second
live job with the same `EXP_ID` is intentionally rejected.
