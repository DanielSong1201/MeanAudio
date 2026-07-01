# Pure FluxAudio Drifting Distillation

This experiment keeps both teacher and student in the `FluxAudio` architecture:

```text
FluxAudio-S-Full multi-step teacher -> FluxAudio-S one-step student
```

The frozen teacher hidden states are produced by the same architecture family
as the student, so this route follows the Teacher-Feature Drifting design
directly.

## Objective

For each batch, the student receives pure noise `x0` and predicts a one-step flow at `t=1`:

```text
x_student = x0 - FluxStudent(x0, t=1, text)
```

The loss is:

```text
L = lambda_flow * L_one_step_flow
  + lambda_tfd * L_teacher_feature_drifting
  + lambda_anchor * L_anchor_margin
```

For conditional TFD, use multiple samples for each caption:

```text
condition batch:          B captions
student samples:          K one-step samples per caption
positive samples:         1 real AudioCaps sample + (K - 1) generated samples
teacher features:         B x K x tokens x hidden_dim
```

Drifting and anchor neighborhoods are computed independently inside each
caption group, then averaged across captions. This prevents unrelated captions
in the same data-loader batch from becoming each other's positive or repulsion
samples. For the paper-aligned `K=4` experiment, one positive is sampled from
the caption's stored AudioCaps VAE posterior and the other three are generated
offline by `meanaudio_l_full.pth` with 25-step MeanFlow and CFG 6. The generated
normalized VAE latents are cached by dataset index, so training does not run
the 25-step generator inside every optimization step.

`BATCH_SIZE=1`, `SAMPLES_PER_CONDITION=4`, and
`TEACHER_POSITIVE_COUNT=3` therefore give four student samples and four
prompt-matched positives for each caption.

where:

- `L_one_step_flow` regresses the student flow to `x0 - x_real`.
- `L_teacher_feature_drifting` is computed in frozen `FluxAudio-S-Full` hidden states.
- `L_anchor_margin` encourages student features to cover real/anchor teacher-feature regions.

Default teacher feature layers:

```text
joint_3,fused_3,fused_7
```

## Files

```text
drifting/flux/
  build_teacher_positive_bank.py
  train.py
  test.py
  README.md
drifting/scripts/flux/
  build_teacher_positive_bank_4gpu.sh
  train_flux_1x4090.sh
  train_flux_2x4090.sh
  test_flux.sh
  sweeps/
    sweep_flux_4gpu_lr1e6_tfd1_anchor1_flow005_200k.sh
    sweep_flux_4gpu_lr1e6_tfd100_anchor1_flow01_200k.sh
    parallel_flux_2x2gpu_lr1e6_tfd1_tfd100_200k.sh
```

## Logs

Training always writes logs and metrics to:

```text
exps/drifting_flux/<exp_id>/train.log
exps/drifting_flux/<exp_id>/metrics.csv
exps/drifting_flux/<exp_id>/eval_metrics.csv
```

Checkpoints are saved to:

```text
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>.pth
exps/drifting_flux/<exp_id>/<exp_id>_<iteration>_ema.pth
exps/drifting_flux/<exp_id>/<exp_id>_last.pth
exps/drifting_flux/<exp_id>/<exp_id>_ema_last.pth
```

EMA is enabled by default and stored on CPU to reduce GPU memory pressure:

```text
EMA_DECAY=0.9999
EMA_START=0
EMA_UPDATE_INTERVAL=1
EMA_DEVICE=cpu
```

`metrics.csv` contains:

```text
iteration,total_loss,flow_loss,tfd_loss,drifting_loss,anchor_loss,grad_norm,lr
```

By default, training runs a full evaluation every 10000 iterations. This uses
the FluxAudio flow-matching evaluation path:

```text
eval.py --variant fluxaudio_s
av-benchmark/evaluate.py
```

Per-iteration eval artifacts are stored under:

```text
exps/drifting_flux_eval/<exp_id>/it_<iteration>/
  audio/
  cache/
  evaluate.log
  eval_driver.log
```

`eval_metrics.csv` records the iteration, checkpoint path, evaluation output
path, `evaluate.log`, `eval_driver.log`, and parsed metrics JSON. The same
metrics JSON is also written into `train.log`.
Training-time eval uses the EMA checkpoint by default when EMA is enabled. Run
`python drifting/flux/train.py ... --eval-raw` if you need eval on raw student
weights instead.

Periodic eval is terminal-silent: all shell, generation, and benchmark output
is redirected to `eval_driver.log` or `evaluate.log`. The training tqdm bars
remain in place while eval runs and resume naturally afterward; eval time is
excluded from the displayed training speed and ETA.

## Train

From the repository root:

```bash
bash drifting/scripts/flux/train_flux_1x4090.sh
```

Use two 4090 cards for one training job:

```bash
bash drifting/scripts/flux/train_flux_2x4090.sh
```

`BATCH_SIZE` is per GPU in the two-card entrypoint, so the default effective
batch size is `2 * BATCH_SIZE`.

The default training length is `1000000` iterations. Override it with
`ITERATIONS=<steps>`.

Training auto-resumes by default. Before the first training step, the script
looks in:

```text
exps/drifting_flux/<exp_id>/
```

for the latest numeric raw checkpoint:

```text
<exp_id>_<iteration>.pth
```

If found, it loads that student checkpoint, loads the matching
`<exp_id>_<iteration>_ema.pth` when present, and continues from
`iteration + 1` until the target `ITERATIONS`. Existing checkpoints are
weights-only, so the optimizer is initialized fresh on resume. Disable this
behavior with:

```bash
AUTO_RESUME=0 bash drifting/scripts/flux/train_flux_1x4090.sh
```

Common overrides:

```bash
EXP_ID=flux_debug \
BATCH_SIZE=2 \
ITERATIONS=100 \
LOG_INTERVAL=10 \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

The learning rate uses a linear warmup over the first 500 global iterations by
default. Both settings can be overridden:

```bash
LR_WARMUP_STEPS=500 \
SAMPLES_PER_CONDITION=4 \
BATCH_SIZE=1 \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

The warmup learning rate at iteration `i` is:

```text
base_lr * min(i / LR_WARMUP_STEPS, 1)
```

Because it depends only on the restored global iteration, no separate scheduler
state is needed for auto-resume.

Run either four-GPU configuration with condition-local TFD:

```bash
bash drifting/scripts/flux/sweeps/sweep_flux_4gpu_lr1e6_tfd1_anchor1_flow005_200k.sh
bash drifting/scripts/flux/sweeps/sweep_flux_4gpu_lr1e6_tfd100_anchor1_flow01_200k.sh
```

Both use `lr=1e-6`, 1000 warmup iterations, four student samples, and
`1 real + 3 MeanAudio-L-generated` positives for each caption. Their loss
weights are:

```text
tfd=1,   anchor=1, flow=0.05
tfd=100, anchor=1, flow=0.1
```

Before training, the sweep builds or resumes this bank:

```text
data/audiocaps/train-teacher-positives-meanaudio-l-full-25step-cfg6/
```

Place the checkpoint at `weights/meanaudio_l_full.pth`. Its source is:

```text
https://huggingface.co/AndreasXi/MeanAudio/resolve/main/meanaudio_l_full.pth
```

Override it with `TEACHER_POSITIVE_WEIGHTS=/path/to/meanaudio_l_full.pth`.
This variable is intentionally separate from `TEACHER_WEIGHTS`, which still
selects the frozen FluxAudio feature teacher used by the training loss.

To build the bank independently:

```bash
bash drifting/scripts/flux/build_teacher_positive_bank_4gpu.sh
```

Each `<dataset_index>.npz` contains three FP16 normalized latent tensors under
`latents_normalized`; `complete.json` is written only after every training item
exists. Existing item files are skipped, so interrupted generation is
resumable. To train from an already completed bank without invoking the
generator:

```bash
PREPARE_TEACHER_POSITIVES=0 \
bash drifting/scripts/flux/sweeps/sweep_flux_4gpu_lr1e6_tfd100_anchor1_flow01_200k.sh
```

After training, the sweep evaluates the same final EMA checkpoint once without
external CFG (one network forward) and once with `CFG=4.5` (two network
forwards). Disable the final comparison with `RUN_CFG_COMPARISON=0`, or run it
separately:

```bash
EXP_ID=flux_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k_4gpu \
bash drifting/scripts/flux/compare_cfg_one_forward.sh
```

Evaluation overrides:

```bash
EVAL_INTERVAL=5000 \
EVAL_NUM_STEPS=1 \
EVAL_CFG_STRENGTH=4.5 \
EMA_DECAY=0.9999 \
EVAL_OUTPUT_ROOT=exps/drifting_flux_eval \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

Set `EVAL_INTERVAL=0` to disable training-time evaluation.

## 200K Sweeps

The sweep directory contains only the two retained experiments and one
parallel launcher. To run both experiments concurrently, with one two-GPU DDP
job on `0,1` and the other on `2,3`:

```bash
bash drifting/scripts/flux/sweeps/parallel_flux_2x2gpu_lr1e6_tfd1_tfd100_200k.sh
```

Override the GPU pairs when needed:

```bash
TFD1_GPUS=0,2 TFD100_GPUS=1,3 \
bash drifting/scripts/flux/sweeps/parallel_flux_2x2gpu_lr1e6_tfd1_tfd100_200k.sh
```

All three scripts export `DDP_TIMEOUT_MINUTES=180`. This value is passed to
`torch.distributed.init_process_group(..., timeout=timedelta(minutes=180))`
for the NCCL process group.

## Tests

Check paths and dataset availability:

```bash
MODE=assets bash drifting/scripts/flux/test_flux.sh
```

Check the standalone drifting loss:

```bash
MODE=loss bash drifting/scripts/flux/test_flux.sh
```

Run one real train step with teacher/student weights and verify gradients:

```bash
MODE=train-step BATCH_SIZE=2 bash drifting/scripts/flux/test_flux.sh
```

Run the same full evaluation entrypoint used by training:

```bash
MODE=eval \
MODEL_PATH=exps/drifting_flux/flux_debug/flux_debug_ema_last.pth \
OUTPUT_PATH=exps/drifting_flux_eval/manual_flux_debug \
bash drifting/scripts/flux/test_flux.sh
```

If the training script fails on the server, run these tests in order:

```bash
MODE=assets bash drifting/scripts/flux/test_flux.sh
MODE=loss bash drifting/scripts/flux/test_flux.sh
MODE=train-step BATCH_SIZE=2 bash drifting/scripts/flux/test_flux.sh
```

This separates path/data problems, loss implementation problems, and full model forward/backward problems.
