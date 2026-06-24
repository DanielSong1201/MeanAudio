# Phase 0 Scripts

These scripts validate the server environment and produce the baseline numbers needed before implementing FluxAudio teacher-feature drifting distillation.

Run from the `MeanAudio` project root:

```bash
bash tst/phase0_00_download_assets.sh
bash tst/phase0_00_check_assets.sh
bash tst/phase0_01_eval_fluxaudio_teacher.sh
bash tst/phase0_02_eval_meanaudio_baseline.sh
```

Or run all:

```bash
bash tst/phase0_run_all.sh
```

Default checkpoints:

- Teacher: `weights/fluxaudio_s_full.pth`
- One-step baseline: `weights/meanaudio_s_full.pth`
- VAE: `weights/v1-16.pth`
- Vocoder: `weights/best_netG.pt`

The downloader uses the Hugging Face repo:

```text
https://huggingface.co/AndreasXi/MeanAudio/tree/main
```

It downloads the Phase-0 required files into `weights/`:

- `fluxaudio_s_full.pth`
- `meanaudio_s_full.pth`
- `v1-16.pth`
- `best_netG.pt`

Override paths with environment variables:

```bash
TEACHER_WEIGHTS=/path/to/fluxaudio_s_full.pth \
BASELINE_WEIGHTS=/path/to/meanaudio_s_full.pth \
CUDA_VISIBLE_DEVICES=0 \
bash tst/phase0_run_all.sh
```

Downloader overrides:

```bash
HF_REPO_ID=AndreasXi/MeanAudio \
WEIGHTS_DIR=weights \
HF_TOKEN=... \
bash tst/phase0_00_download_assets.sh
```

To also download optional project weights such as `meanaudio_l_full.pth`, `meanaudio_s_ac.pth`, and empty-string condition tensors:

```bash
INCLUDE_OPTIONAL_WEIGHTS=1 bash tst/phase0_00_download_assets.sh
```

The evaluation scripts write outputs under:

- `exps/phase0_fluxaudio_s_full/`
- `exps/phase0_meanaudio_s_full/`

They reuse the repository's `eval.py` and `av-benchmark/evaluate.py` paths, so install `av-benchmark` on the 4090 server before running the evaluation steps.

## Phase 1

After Phase 0 succeeds, validate that the frozen `FluxAudio-S-Full` teacher can expose intermediate hidden states while still allowing gradients to flow back to the generated latent:

```bash
bash tst/phase1_00_check_teacher_features.sh
```

Default feature layers:

- `joint_3`
- `fused_3`
- `fused_7`

The script checks that each extracted feature has shape `B x 312 x 448`, that teacher parameters remain gradient-free, and that the input latent receives gradients through the frozen teacher feature path.

## Phase 2

Phase 2 adds a standalone teacher-feature drifting loss implementation:

```text
meanaudio/model/teacher_feature_drifting.py
```

Run the synthetic gradient check:

```bash
bash tst/phase2_00_check_tfd_loss.sh
```

On a CUDA server, you can also run:

```bash
DEVICE=cuda DTYPE=bfloat16 bash tst/phase2_00_check_tfd_loss.sh
```

If the active shell does not expose `python`, specify the interpreter:

```bash
PYTHON_BIN=/path/to/conda/env/bin/python bash tst/phase2_00_check_tfd_loss.sh
```

This check does not load the FluxAudio teacher. It verifies the drifting loss and anchor-margin coverage loss on synthetic feature dictionaries with the same layer names used by Phase 1.
