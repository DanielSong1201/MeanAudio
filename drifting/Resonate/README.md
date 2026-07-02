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
[`requirements.txt`](requirements.txt).

## Phase 0-3 status

- The released `fluxaudio_m_44k` architecture is defined by
  `ResonateModelConfig`.
- `ResonateFluxAudio` preserves the official checkpoint parameter hierarchy.
- Teacher and student can be constructed independently from
  `Resonate_GRPO.pth` or `Resonate_PT.pth`.
- `extract_features()` exposes differentiable intermediate audio-token states
  for TFD. Positive features must be extracted under `torch.no_grad()` by the
  future training entrypoint; generated features must not be wrapped in
  `torch.no_grad()`.
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

`Resonate_GRPO.pth` is required for teacher-positive generation and
`v1-44.pth` is required for AudioCaps preprocessing. The following assets are
reserved for later waveform evaluation:

```text
weights/bigvgan_v2_44khz_128band_512x/
sets/latent_mean_44k.pt
sets/latent_std_44k.pt
```

Training and waveform evaluation remain to be implemented.

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
future training loss will sample one real latent from the AudioCaps posterior
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

During future training, the three stored latents are combined with one newly
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

These are direct Python/CLI interfaces, not phase-specific sweep or launch
scripts. Both operations support deterministic seeds, distributed sharding,
atomic files, completion markers, resume-by-existing-file, `--overwrite`, and
small `--limit` runs.

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
