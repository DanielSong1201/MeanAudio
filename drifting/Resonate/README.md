# Resonate Teacher-Feature Drifting

This directory contains an isolated Resonate backend for teacher-feature
drifting. Existing files under `drifting/flux/` are not imported or modified
by the model implementation, so the FluxAudio-S route keeps its current
behavior.

The architecture is aligned to upstream
[`xiquan-li/Resonate`](https://github.com/xiquan-li/Resonate) revision
`b08fb6f7887e129623e0efae3e84653783be5c69`.

Environment installation and checkpoint download instructions are in
[`INSTALL.md`](INSTALL.md). Phase 0-3's minimal Python dependencies are in
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

## Required assets for later phases

```text
weights/Resonate_GRPO.pth
weights/Resonate_PT.pth
weights/v1-44.pth
weights/bigvgan_v2_44khz_128band_512x/
sets/latent_mean_44k.pt
sets/latent_std_44k.pt
```

Data preprocessing, positive-bank generation, training, and evaluation are
implemented in later phases.

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
