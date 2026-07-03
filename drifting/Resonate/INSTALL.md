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

## 4. Download the released Resonate checkpoint

The default Phase 0-3 model is `Resonate_GRPO.pth`. Run:

```bash
bash drifting/scripts/resonate/download_resonate_model.sh
```

The resulting file is:

```text
weights/Resonate_GRPO.pth
```

The downloader is resumable through Hugging Face Hub. Hugging Face download
metadata may additionally appear under:

```text
weights/.cache/huggingface/
```

To download the pre-training checkpoint instead:

```bash
MODEL_FILE=Resonate_PT.pth \
bash drifting/scripts/resonate/download_resonate_model.sh
```

This produces:

```text
weights/Resonate_PT.pth
```

The AudioCaps preprocessing interface additionally needs the released 44.1 kHz
VAE:

```bash
MODEL_FILE=v1-44.pth \
bash drifting/scripts/resonate/download_resonate_model.sh
```

This produces:

```text
weights/v1-44.pth
```

Waveform evaluation also requires a local 44.1 kHz BigVGAN-v2 snapshot:

```bash
huggingface-cli download \
  nvidia/bigvgan_v2_44khz_128band_512x \
  --local-dir weights/bigvgan_v2_44khz_128band_512x
```

Install AV-Benchmark at the repository root using the existing MeanAudio
instructions. The expected entrypoint is:

```text
av-benchmark/evaluate.py
```

To use another storage root:

```bash
WEIGHTS_DIR=/data/checkpoints/resonate \
bash drifting/scripts/resonate/download_resonate_model.sh
```

Then pass the resulting path explicitly to later commands. For example:

```bash
CHECKPOINT=/data/checkpoints/resonate/Resonate_GRPO.pth \
bash drifting/scripts/resonate/validate_phase_0_3.sh
```

If direct access to Hugging Face is unavailable but the server uses a
compatible mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com \
bash drifting/scripts/resonate/download_resonate_model.sh
```

## 5. Validate Phase 0-3

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
