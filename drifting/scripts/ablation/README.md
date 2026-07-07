# FluxAudio TFD Ablation Scripts

This directory contains three launchers for the current FluxAudio teacher-feature drifting bottleneck check. Each script wraps `drifting/scripts/flux/train_flux_4gpu_custom_loss.sh`, so the same command supports one GPU or multiple GPUs.

## Scripts

- `train_pool4_hybridpos.sh`: paper-aligned pooling (`POOL_TOKENS=4`) with one real posterior positive plus three cached teacher positives.
- `train_pool4_realpos.sh`: paper-aligned pooling with only real posterior positives. This isolates whether cached teacher positives are pulling the student toward a mismatched distribution.
- `train_pool64_hybridpos_control.sh`: current pooling control (`POOL_TOKENS=64`) with hybrid positives.

## Usage

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash drifting/scripts/ablation/train_pool4_hybridpos.sh
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash drifting/scripts/ablation/train_pool4_hybridpos.sh
```

Four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash drifting/scripts/ablation/train_pool4_hybridpos.sh
```

Common overrides:

```bash
EXP_ID=my_ablation_run \
ITERATIONS=50000 \
BATCH_SIZE=1 \
LAMBDA_FLOW=0.05 \
LAMBDA_TFD=1 \
LAMBDA_ANCHOR=1 \
CUDA_VISIBLE_DEVICES=2,3 \
NPROC_PER_NODE=2 \
bash drifting/scripts/ablation/train_pool4_realpos.sh
```

Hybrid-positive scripts expect:

```text
data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6/complete.json
```

Override `TEACHER_POSITIVE_DIR` if the bank is stored elsewhere.
