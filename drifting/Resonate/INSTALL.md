# Resonate TFD Installation

## Environment

Phase 0-3 has been designed for:

- Linux
- Python 3.11
- NVIDIA RTX 4090
- PyTorch 2.5.1 or newer
- CUDA-enabled PyTorch

The current implementation constructs the Resonate architecture, validates
the TFD feature-gradient contract, preprocesses AudioCaps, generates the
offline teacher-positive bank, trains the one-step TFD student, and evaluates
it with AV-Benchmark. It does not need the full Resonate GRPO/RL dependency
stack.

## 1. Create the Conda environment

From the MeanAudio repository root:

```bash
conda create -n resonate-tfd python=3.11 -y
conda activate resonate-tfd
python -m pip install --upgrade pip
```

## 2. Install the CUDA 12.4 environment

First inspect the server driver:

```bash
nvidia-smi
```

The existing `requirements.txt` is pinned to the server's CUDA configuration:

```bash
python -m pip install \
  -r drifting/Resonate/requirements.txt
```

It installs:

```text
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
```

Do not combine PyTorch 2.5.1 with `torchvision 0.27.1`. That torchvision
release requires a different torch ABI.

The same file also includes the direct Resonate runtime dependencies:
Transformers/SentencePiece, einops, NumPy/SciPy, librosa/soundfile,
OmegaConf/PyYAML, tqdm, Hugging Face Hub/Xet, Cython, and a
`setuptools<81` compatibility bound required by the benchmark dependency
stack.

### Repair the currently conflicting environment

If the environment already contains `torch 2.5.1+cu124`,
`torchaudio 2.5.1+cu124`, and the incompatible `torchvision 0.27.1`, repair
all three together:

```bash
python -m pip uninstall -y torch torchvision torchaudio

python -m pip install --no-cache-dir \
  -r drifting/Resonate/requirements.txt
```

Reinstalling the three packages in one command prevents pip from retaining a
binary-incompatible torchvision build.

Verify CUDA access:

```bash
python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available())"
```

Expected versions are:

```text
torch 2.5.1+cu124
torchvision 0.20.1+cu124
torchaudio 2.5.1+cu124
cuda 12.4
available True
```

## 3. Install this repository

```bash
python -m pip install -e . --no-deps
python -m pip check
```

`--no-deps` avoids installing MeanAudio's unrelated full dependency stack.
`pip check` must finish without a torch/torchvision/torchaudio conflict.

## 4. Install AV-Benchmark

Periodic training evaluation uses the official benchmark as a separate,
untracked checkout:

```bash
git clone https://github.com/hkchengrex/av-benchmark.git
python -m pip install -e av-benchmark
test -f av-benchmark/evaluate.py
```

The checkout must be named `av-benchmark` at the MeanAudio repository root.
It is ignored by this repository and remains decoupled from the Resonate
implementation.

## 5. Prepare all runtime model assets

The standalone bootstrap covers every fixed checkpoint directly referenced by
Resonate training and periodic evaluation:

| Use | Local path | Source |
| --- | --- | --- |
| teacher and student initialization | `weights/Resonate_GRPO.pth` | `AndreasXi/Resonate` |
| 44.1 kHz latent normalization | `sets/latent_mean_44k.pt` | pinned official Resonate revision |
| 44.1 kHz latent normalization | `sets/latent_std_44k.pt` | pinned official Resonate revision |
| periodic-eval VAE | `weights/v1-44.pth` | `AndreasXi/Resonate` |
| periodic-eval vocoder config | `weights/bigvgan_v2_44khz_128band_512x/config.json` | NVIDIA BigVGAN-v2 |
| periodic-eval vocoder | `weights/bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt` | NVIDIA BigVGAN-v2 |
| AV-Benchmark LAION-CLAP | `av-benchmark/weights/music_speech_audioset_epoch_15_esc_89.98.pt` | `lukewys/laion_clap` |
| AV-Benchmark Synchformer | `av-benchmark/weights/synchformer_state_dict.pth` | MMAudio release |

Run it explicitly with one selected GPU visible:

```bash
ASSET_DOWNLOAD_GPU=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

The downloader itself does not initialize CUDA or distributed training.
`CUDA_VISIBLE_DEVICES` is restricted to the single
`ASSET_DOWNLOAD_GPU`, downloads show Hugging Face or tqdm progress, and every
missing file is announced with an `INFO` line.

Concurrent launchers share:

```text
weights/.resonate-runtime-assets/prepare.lock
weights/.resonate-runtime-assets/<request-hash>.json
```

Only the process holding the lock checks remote sources and downloads.
Other sweeps wait, then reuse the completion marker and local files. Hugging
Face transfers use its resumable cache; direct URL transfers retain a
`.part` file and resume when the server supports byte ranges.

The formal `train.sh` and `smoke_train_eval.sh` invoke this bootstrap before
starting `torchrun`. Therefore normal training requires no separate checkpoint
download command. To prepare only training assets without eval assets:

```bash
WITH_EVAL=0 WITH_AV_BENCHMARK=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

For a generation-only smoke test, AV-Benchmark and its two weights can be
omitted:

```bash
WITH_EVAL=1 WITH_AV_BENCHMARK=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

`Resonate_PT.pth` is optional and is not downloaded by the default training
route. Download it only when explicitly needed:

```bash
MODEL_FILE=Resonate_PT.pth \
bash drifting/scripts/resonate/download_resonate_model.sh
```

Custom, already existing teacher/student checkpoint paths are accepted. A
missing custom checkpoint cannot be inferred automatically and produces an
actionable error. Disable automatic preparation only when assets are managed
externally:

```bash
PREPARE_ASSETS=0 bash drifting/scripts/resonate/train.sh
```

If direct Hugging Face access is unavailable, a compatible endpoint may be
used without changing the scripts:

```bash
HF_ENDPOINT=https://hf-mirror.com \
ASSET_DOWNLOAD_GPU=0 \
bash drifting/scripts/resonate/prepare_runtime_assets.sh
```

AV-Benchmark dependencies such as PaSST, PANNs, VGGish, ImageBind, and
MS-CLAP are installed by `pip install -e av-benchmark`. Their upstream
packages may populate the standard PyTorch/user cache on the first benchmark
run. The two manually required AV-Benchmark checkpoints are handled by the
bootstrap above.

## 6. Prepare non-model data

Model download automation does not fabricate datasets or evaluation caches.
Before formal training, prepare these separately:

```text
drifting/data/AudioCaps_CVSSP/                         # raw AudioCaps symlink
data/audiocaps_resonate/train-npz-flant5-44k/         # processed train data
data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5/
data/audiocaps_resonate/test-npz-flant5-44k/           # processed eval data
gt_audio/                                               # AV-Benchmark GT audio
data/audiocaps/test-features/                           # AV-Benchmark GT cache
```

The exact preprocessing and positive-bank commands are documented in
`drifting/Resonate/README.md`. `train.sh` validates them after model
preparation and before launching workers.

## 7. Validate Phase 0-3

Run the complete validation, including strict checkpoint loading:

```bash
bash drifting/scripts/resonate/validate_phase_0_3.sh
```

To put the tiny gradient check on CUDA:

```bash
DEVICE=cuda bash drifting/scripts/resonate/validate_phase_0_3.sh
```

Successful completion ends with:

```text
[success] Resonate TFD Phase 0-3 validation passed
```
