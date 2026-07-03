# Resonate Teacher-Feature Drifting

This directory contains an isolated Resonate backend for teacher-feature
drifting. Existing files under `drifting/flux/` are not imported or modified
by the model implementation, so the FluxAudio-S route keeps its current
behavior.

The architecture is aligned to upstream
[`xiquan-li/Resonate`](https://github.com/xiquan-li/Resonate) revision
`b08fb6f7887e129623e0efae3e84653783be5c69`.

Environment installation and checkpoint download instructions are in
[`INSTALL.md`](INSTALL.md). The current Python dependencies are in
[`requirements.txt`](requirements.txt), including the server-specific CUDA
12.4 pins for torch, torchvision, and torchaudio.

## Phase 0-3 status

- The released `fluxaudio_m_44k` architecture is defined by
  `ResonateModelConfig`.
- `ResonateFluxAudio` preserves the official checkpoint parameter hierarchy.
- Teacher and student can be constructed independently from
  `Resonate_GRPO.pth` or `Resonate_PT.pth`.
- `extract_features()` exposes differentiable intermediate audio-token states
  for TFD. Positive features are extracted under `torch.no_grad()` by
  `train.py`; generated features are not wrapped in `torch.no_grad()`.
- All 44.1 kHz asset paths and model dimensions live in `config.py`.

The default TFD layers preserve the relative placement of the FluxAudio-S
layers while accounting for Resonate's greater depth:

```text
joint_15, fused_17, fused_35
```

## Assets

```text
weights/Resonate_GRPO.pth
weights/Resonate_PT.pth
weights/v1-44.pth
```

`Resonate_GRPO.pth` is required for teacher-positive generation and training;
`v1-44.pth` is required for AudioCaps preprocessing and waveform decoding.
Training/evaluation additionally uses:

```text
weights/bigvgan_v2_44khz_128band_512x/
sets/latent_mean_44k.pt
sets/latent_std_44k.pt
av-benchmark/weights/music_speech_audioset_epoch_15_esc_89.98.pt
av-benchmark/weights/synchformer_state_dict.pth
```

One-step TFD training and AV-Benchmark waveform evaluation are implemented in
`train.py` and `test.py`.

All fixed runtime checkpoints can be prepared in one locked, standalone
process:

```bash
ASSET_DOWNLOAD_GPU=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

`train.sh` and `smoke_train_eval.sh` call this automatically before any
`torchrun` process starts. Parallel sweeps share a lock and completion marker,
so only one launcher downloads while the others reuse the result. See
`INSTALL.md` for the complete asset table and AV-Benchmark installation.

## AudioCaps preprocessing interface

The raw AudioCaps symlink is expected at:

```text
drifting/data/AudioCaps_CVSSP/
├── train/
├── eval/
└── test/
```

`prepare_audiocaps.py` accepts explicit CSV, TSV, or JSONL metadata with
`id`/`audio_id`/`youtube_id` and `caption`/`prompt` columns. Without
`--manifest`, it searches the source directory first and then falls back to
the existing Flux manifests under `data/audiocaps/`.

Inspect one split without loading a model or writing outputs:

```bash
python drifting/Resonate/prepare_audiocaps.py \
  --split train \
  --dry-run
```

### Single-GPU preprocessing

The preprocessing entrypoint runs without distributed initialization when it
is launched with ordinary `python`. Select one GPU and process each split:

```bash
export CUDA_VISIBLE_DEVICES=0

for split in train eval test; do
  python drifting/Resonate/prepare_audiocaps.py \
    --split "${split}" \
    --batch-size 4 \
    --num-workers 4
done
```

This extracts 44.1 kHz VAE distributions and Flan-T5 conditions. Outputs are
isolated from Flux:

```text
data/audiocaps_resonate/train.tsv
data/audiocaps_resonate/train-npz-flant5-44k/
data/audiocaps_resonate/eval.tsv
data/audiocaps_resonate/eval-npz-flant5-44k/
data/audiocaps_resonate/test.tsv
data/audiocaps_resonate/test-npz-flant5-44k/
```

The corresponding data-path interface is
`config/data/resonate_flant5_44k.yaml`.

Run a small isolated preprocessing check before processing the full dataset:

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/prepare_audiocaps.py \
  --split train \
  --limit 4 \
  --output-root data/audiocaps_resonate_smoke
```

This does not affect the full output directory. Re-running the same command
skips existing NPZ files. Use `--overwrite` only when deliberately rebuilding
them.

The same interface can still be scaled to multiple GPUs later:

```bash
torchrun --standalone --nproc_per_node=4 \
  drifting/Resonate/prepare_audiocaps.py \
  --split train
```

## Four-positive completion interface

The positive-bank builder uses the frozen Resonate-GRPO model with 25 Euler
steps and CFG 4.5. It generates three normalized latent positives for every
prompt:

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/build_teacher_positive_bank.py
```

The default output is:

```text
data/audiocaps_resonate/
  train-teacher-positives-resonate-grpo-25step-cfg4.5/
```

`ResonateNpzDataset` validates and loads these three generated positives. The
training loss samples one real latent from the AudioCaps posterior
and concatenate it with the three bank entries, giving four prompt-matched
positives without duplicating the real sample on disk.

### Single-GPU three-positive generation

Before generation, the following files must exist:

```text
weights/Resonate_GRPO.pth
data/audiocaps_resonate/train.tsv
data/audiocaps_resonate/train-npz-flant5-44k/complete.json
```

Download the model if necessary:

```bash
bash drifting/scripts/resonate/download_resonate_model.sh
```

First run a small isolated generation check:

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/build_teacher_positive_bank.py \
  --tsv data/audiocaps_resonate_smoke/train.tsv \
  --npz-dir data/audiocaps_resonate_smoke/train-npz-flant5-44k \
  --output-dir data/audiocaps_resonate_smoke/train-teacher-positives \
  --limit 4
```

For the full training set, use:

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/build_teacher_positive_bank.py \
  --positives-per-condition 3 \
  --num-steps 25 \
  --cfg-strength 4.5
```

For each prompt the script performs the following:

1. Loads its precomputed `[77, 1024]` Flan-T5 token features and `[1024]`
   pooled feature.
2. Creates three independent normalized Gaussian noise latents.
3. Expands the same prompt condition to a batch of three.
4. Runs reverse-flow Euler sampling from `t=1` to `t=0` for 25 steps.
5. Uses CFG 4.5, so every step evaluates both conditional and empty-prompt
   paths.
6. Stores the three generated normalized latents as
   `latents_normalized`.

Each output has shape:

```text
[3, latent_tokens, 40]
```

The files are indexed exactly like the processed training NPZ files:

```text
0.npz
1.npz
2.npz
...
config.json
complete.json
```

Inspect one generated entry:

```bash
python -c "import numpy as np; x=np.load('data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5/0.npz')['latents_normalized']; print(x.shape, x.dtype)"
```

Expected output is similar to:

```text
(3, 431, 40) float16
```

The exact token count follows the preprocessed real latent and can differ if
the configured audio duration changes.

Generation is resumable. Re-running the full command skips every existing
`<index>.npz` and only generates missing entries. `complete.json` is written
only after all expected files exist. Do not use `--overwrite` when merely
resuming an interrupted run.

During training, the three stored latents are combined with one newly
sampled real AudioCaps posterior latent:

```text
1 real positive + 3 Resonate-generated positives = 4 positives
```

The positive bank contains latents rather than decoded waveform audio. Audio
decoding is unnecessary because TFD consumes the teacher's latent-space hidden
features.

To scale the same generation interface to four GPUs later:

```bash
torchrun --standalone --nproc_per_node=4 \
  drifting/Resonate/build_teacher_positive_bank.py
```

### Custom-GPU positive launcher

`drifting/scripts/resonate/build_teacher_positive_bank.sh` wraps the same
Python interface and automatically derives the process count from
`CUDA_VISIBLE_DEVICES`.

One GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

Two selected GPUs:

```bash
CUDA_VISIBLE_DEVICES=1,3 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

Four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

The optional `NPROC_PER_NODE` value must match the number of visible GPU IDs;
the launcher rejects mismatches instead of silently oversubscribing a card.

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
POSITIVES_PER_CONDITION=3 \
NUM_STEPS=25 \
CFG_STRENGTH=4.5 \
LIMIT=100 \
OUTPUT_DIR=data/audiocaps_resonate/positive-test-100 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

Supported environment variables include:

```text
CUDA_VISIBLE_DEVICES
NPROC_PER_NODE
TSV
NPZ_DIR
OUTPUT_DIR
TEACHER_WEIGHTS
POSITIVES_PER_CONDITION
NUM_STEPS
CFG_STRENGTH
SEED
STORAGE_DTYPE
LOG_LEVEL
AMP
LIMIT
OVERWRITE
ALLOW_INCOMPLETE_DATA
COMPLETION_POLL_SECONDS
COMPLETION_TIMEOUT_MINUTES
SHOW_ALL_GPU_PROGRESS
```

`AMP=0` selects full precision. `OVERWRITE=0` is the safe default and resumes
by skipping existing files. The output directory contains a non-blocking
`.positive-bank.lock`, preventing two launchers from generating the same bank
simultaneously. To inspect the resolved command without loading a model:

```bash
CUDA_VISIBLE_DEVICES=0,2,3 \
DRY_RUN=1 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

Multi-GPU positive generation does not initialize a Torch/NCCL process group.
Each `torchrun` worker owns a disjoint index shard and writes atomic NPZ files.
Workers report completion through files under `.run_state/`; rank0 writes
`complete.json` only after every worker marker and every expected NPZ exist.
This avoids collective timeouts when GPU shards finish hours apart.

Interrupted generation resumes by default. Run the same command again with
`OVERWRITE=0` (the default); existing `<index>.npz` files are retained and only
missing indices are generated. The GPU count may be changed when resuming:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OVERWRITE=0 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

`COMPLETION_TIMEOUT_MINUTES=0` means rank0 waits without a fixed timeout for
slower workers. Set a positive value only when an explicit upper bound is
desired.

By default, every GPU/rank has a fixed tqdm row in the same terminal:

```text
resonate-positives-gpu0-rank0:  31%|...
resonate-positives-gpu1-rank1:  33%|...
resonate-positives-gpu2-rank2:  32%|...
resonate-positives-gpu3-rank3:  33%|...
```

The worker processes do not write ANSI progress output directly. Each worker
atomically publishes `completed/total` under the current `.run_state/`
directory, and one rank0 renderer owns all four terminal rows. This avoids
stale duplicate or "ghost" progress lines caused by independent torchrun
processes racing to reposition the same terminal cursor.

For redirected logs or terminals without reliable ANSI cursor control, keep
only the rank0 progress row:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SHOW_ALL_GPU_PROGRESS=0 \
bash drifting/scripts/resonate/build_teacher_positive_bank.sh
```

These are direct Python/CLI interfaces, not phase-specific sweep or launch
scripts. Both operations support deterministic seeds, distributed sharding,
atomic files, completion markers, resume-by-existing-file, `--overwrite`, and
small `--limit` runs.

## tqdm and INFO behavior

Both preprocessing and positive generation use the shared tqdm-aware logger in
`drifting/Resonate/progress.py`.

- Only rank0 displays a progress bar and INFO messages.
- INFO messages are written through `tqdm.write()`, so they do not overwrite or
  freeze the active progress bar.
- Preprocessing reports progress in remaining batches.
- Positive generation reports progress in remaining prompts.
- Before starting, each command reports `remaining`, `existing`, and `total`.
- Resuming therefore shows only work that still needs to be completed.
- Non-main distributed ranks perform their shards without duplicating console
  output.

The default level is INFO. It can be changed without disabling tqdm:

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/build_teacher_positive_bank.py \
  --log-level WARNING
```

## Single-GPU end-to-end data path

The complete currently implemented path is:

### 1. Raw AudioCaps

```text
drifting/data/AudioCaps_CVSSP/
├── train/
├── eval/
└── test/
```

Each directory contains waveform files. Caption metadata is discovered from
the source directory or from:

```text
data/audiocaps/train-memmap.tsv
data/audiocaps/val-memmap.tsv
data/audiocaps/test-memmap.tsv
```

### 2. Required weights

```bash
# Resonate-GRPO flow model used to generate positives.
bash drifting/scripts/resonate/download_resonate_model.sh

# Resonate 44.1 kHz VAE used to encode real AudioCaps waveforms.
MODEL_FILE=v1-44.pth \
bash drifting/scripts/resonate/download_resonate_model.sh
```

They are stored at:

```text
weights/Resonate_GRPO.pth
weights/v1-44.pth
```

### 3. Validate source discovery

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/prepare_audiocaps.py \
  --split train \
  --dry-run
```

This validates the symlink, caption manifest, audio IDs, and missing-audio
count without loading the VAE or writing processed data.

### 4. Preprocess all splits

```bash
export CUDA_VISIBLE_DEVICES=0

for split in train eval test; do
  python drifting/Resonate/prepare_audiocaps.py \
    --split "${split}" \
    --batch-size 4 \
    --num-workers 4
done
```

For each waveform, `prepare_audiocaps.py` performs:

```text
waveform
  -> mono
  -> resample/pad/crop to 44.1 kHz x 10 seconds
  -> 128-bin log-mel
  -> v1-44 VAE posterior mean/std
```

For each caption it performs:

```text
caption
  -> google/flan-t5-large
  -> text_features [77, 1024]
  -> mean-pooled text_features_c [1024]
```

The resulting paths are:

```text
data/audiocaps_resonate/
├── train.tsv
├── train-npz-flant5-44k/
│   ├── 0.npz
│   ├── 1.npz
│   ├── config.json
│   └── complete.json
├── eval.tsv
├── eval-npz-flant5-44k/
├── test.tsv
└── test-npz-flant5-44k/
```

Each processed NPZ contains:

```text
mean                 [latent_tokens, 40]
std                  [latent_tokens, 40]
text_features        [77, 1024]
text_features_c      [1024]
```

The training data-path registry is:

```text
config/data/resonate_flant5_44k.yaml
```

### 5. Generate three teacher positives

After train preprocessing completes:

```bash
CUDA_VISIBLE_DEVICES=0 \
python drifting/Resonate/build_teacher_positive_bank.py \
  --positives-per-condition 3 \
  --num-steps 25 \
  --cfg-strength 4.5
```

The generation path is:

```text
train NPZ text condition
  -> repeat prompt condition three times
  -> three independent Gaussian latent noises
  -> frozen Resonate_GRPO.pth
  -> 25 reverse-flow Euler steps with CFG 4.5
  -> three normalized Resonate latents
```

Output:

```text
data/audiocaps_resonate/
└── train-teacher-positives-resonate-grpo-25step-cfg4.5/
    ├── 0.npz
    ├── 1.npz
    ├── config.json
    └── complete.json
```

Each file contains:

```text
latents_normalized [3, latent_tokens, 40]
```

### 6. Validate the complete preprocessing pipeline

Before training, run the fast completeness check:

```bash
bash drifting/scripts/resonate/check_preprocessing_complete.sh
```

It checks:

- `train`, `eval`, and `test` TSV row counts;
- matching preprocessing `config.json` and `complete.json`;
- every expected numeric NPZ filename from `0` through `num_items - 1`;
- required arrays and Resonate dimensions in evenly distributed samples;
- matching positive-bank `config.json` and `complete.json`;
- one positive NPZ for every training index;
- at least three positives per checked prompt;
- sampled `item_index`, `item_id`, and real/positive latent-shape alignment.

The command exits with status 1 if any expected preprocessing or positive file
is missing. A successful fast check ends with:

```text
[success] Resonate preprocessing is complete (deep=False, warnings=0)
```

For a full content scan of every NPZ, including NaN/infinity checks:

```bash
DEEP=1 \
bash drifting/scripts/resonate/check_preprocessing_complete.sh
```

Fast mode still checks the existence of every indexed NPZ; only expensive
array-content checks are sampled. `drifting/scripts/resonate/train.sh` runs the
fast check automatically before model loading. It can be controlled with:

```text
VALIDATE_PREPROCESSING=1   # default; refuse incomplete training data
VALIDATION_DEEP=0          # set 1 for a full pre-training scan
VALIDATION_SAMPLE_COUNT=16
```

### 7. Four-positive TFD training

`ResonateNpzDataset` reads the real posterior and the three generated entries.
The training step constructs:

```text
sample(mean, std)                  -> 1 real positive
latents_normalized from bank       -> 3 generated positives
                                      ---------------------
                                      4 positives per prompt
```

The real sample is normalized exactly once. The three generated entries are
already normalized and are concatenated without a second normalization.

The two initial four-GPU presets match the Flux experiments:

```bash
# lr=1e-6, TFD=100, anchor=1, flow=0.1, warmup=1000, 200k iterations
bash drifting/scripts/resonate/train_tfd100_anchor1_flow01_4gpu.sh

# lr=1e-6, TFD=1, anchor=1, flow=0.05, warmup=1000, 200k iterations
bash drifting/scripts/resonate/train_tfd1_anchor1_flow005_4gpu.sh
```

Both use:

```text
teacher = weights/Resonate_GRPO.pth
student init = weights/Resonate_GRPO.pth
1 real positive + 3 offline Resonate positives
feature layers = joint_15,fused_17,fused_35
save interval = 1,000
eval interval = 10,000
eval sampler = 1 Euler step, CFG 4.5
```

Before workers start, each preset invokes the shared runtime-asset bootstrap.
The bootstrap exposes only `ASSET_DOWNLOAD_GPU` (default: the first training
GPU), prints every missing model as `INFO`, and displays download progress.
It then exits; model preparation is not part of the distributed training
program. Two presets started concurrently do not repeat the download.

The generic launcher derives its process count from `CUDA_VISIBLE_DEVICES`, so
the same training can use another GPU count:

```bash
CUDA_VISIBLE_DEVICES=0,2 \
LAMBDA_TFD=1 \
LAMBDA_ANCHOR=1 \
LAMBDA_FLOW=0.05 \
bash drifting/scripts/resonate/train.sh
```

Inspect the resolved command without starting training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DRY_RUN=1 \
bash drifting/scripts/resonate/train_tfd100_anchor1_flow01_4gpu.sh
```

### 8. Periodic AV-Benchmark evaluation

Every `--eval-interval` iterations, rank0 evaluates the EMA checkpoint by
default. The evaluation:

1. loads precomputed Flan-T5 conditions from the Resonate test NPZ directory;
2. runs the one-step Resonate sampler;
3. decodes with `v1-44.pth` and the local 44.1 kHz BigVGAN-v2;
4. writes FLAC files under the iteration-specific eval directory;
5. calls `av-benchmark/evaluate.py` with the same audio-only settings as Flux.

Required evaluation assets are:

```text
av-benchmark/evaluate.py
gt_audio/
data/audiocaps/test-features/
weights/v1-44.pth
weights/bigvgan_v2_44khz_128band_512x/
av-benchmark/weights/music_speech_audioset_epoch_15_esc_89.98.pt
av-benchmark/weights/synchformer_state_dict.pth
data/audiocaps_resonate/test.tsv
data/audiocaps_resonate/test-npz-flant5-44k/
```

The default outputs are:

```text
exps/drifting_resonate/<exp_id>/
exps/drifting_resonate_eval/<exp_id>/it_00010000/
```

Evaluate a checkpoint manually with the same path:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=exps/drifting_resonate/<exp_id>/<exp_id>_10000_ema.pth \
OUTPUT_PATH=exps/drifting_resonate_eval/<exp_id>/manual_it10000 \
bash drifting/scripts/resonate/eval_checkpoint.sh
```

### 9. Train/eval smoke test

The smoke test uses a new timestamped experiment. It trains iteration 1,
saves and evaluates at iteration 2, then verifies training continued through
iteration 3:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash drifting/scripts/resonate/smoke_train_eval.sh
```

Missing released model assets are downloaded before the smoke-test file
checks. Dataset outputs, `gt_audio/`, and the AV-Benchmark source checkout
remain explicit prerequisites because they are not model files.

Use multiple GPUs by listing them; the process count is inferred:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash drifting/scripts/resonate/smoke_train_eval.sh
```

By default it generates sixteen eval samples and runs AV-Benchmark. For a faster
generation-only diagnostic:

```bash
CUDA_VISIBLE_DEVICES=0 \
EVAL_LIMIT=2 \
EVAL_SKIP_AV_BENCHMARK=1 \
bash drifting/scripts/resonate/smoke_train_eval.sh
```

## Phase 0-3 server validation

After placing `Resonate_GRPO.pth` under `weights/`, run:

```bash
bash drifting/scripts/resonate/validate_phase_0_3.sh
```

The validation checks:

1. Flux-protected files still match the Phase 0 baseline commit.
2. The released Resonate architecture and 44.1 kHz dimensions.
3. Generated-feature gradients reach the input latent while the frozen teacher
   receives no parameter gradients.
4. Positive features are detached.
5. The released checkpoint loads with `strict=True`.

The tiny gradient check runs on CPU by default. To use CUDA:

```bash
DEVICE=cuda bash drifting/scripts/resonate/validate_phase_0_3.sh
```

For a structural check before downloading the checkpoint:

```bash
bash drifting/scripts/resonate/validate_phase_0_3.sh --skip-checkpoint
```
