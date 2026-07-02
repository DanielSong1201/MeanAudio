# Resonate TFD Phase 0-3 Installation

## Environment

Phase 0-3 has been designed for:

- Linux
- Python 3.11
- NVIDIA RTX 4090
- PyTorch 2.5.1 or newer
- CUDA-enabled PyTorch

The current phase only constructs the Resonate architecture, loads its
checkpoint, and validates the TFD feature-gradient contract. It does not need
the full Resonate GRPO/RL dependency stack.

## 1. Create the Conda environment

From the MeanAudio repository root:

```bash
conda create -n resonate-tfd python=3.11 -y
conda activate resonate-tfd
python -m pip install --upgrade pip
```

## 2. Install CUDA PyTorch

First inspect the server driver:

```bash
nvidia-smi
```

For a server compatible with CUDA 12.1 wheels:

```bash
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

If the existing server environment already has PyTorch 2.5.1 or newer, keep
that installation instead of reinstalling it.

Verify CUDA access:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The second line should be `True` for later GPU work. Phase 0-3's tiny gradient
contract can also run on CPU.

## 3. Install the Phase 0-3 dependencies

```bash
python -m pip install -r drifting/Resonate/requirements.txt
python -m pip install -e . --no-deps
```

`--no-deps` avoids installing MeanAudio's full audio-generation dependency
stack during this phase. Later data preprocessing and 44.1 kHz evaluation
phases will extend the requirements.

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
