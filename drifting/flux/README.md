# Pure FluxAudio Drifting Distillation

This experiment keeps both teacher and student in the `FluxAudio` architecture:

```text
FluxAudio-S-Full multi-step teacher -> FluxAudio-S one-step student
```

It is closer to the original Teacher-Feature Drifting idea than the existing `drifting/train.py` path because the frozen teacher hidden states are produced by the same architecture family as the student.

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
  train.py
  test.py
  README.md
drifting/scripts/flux/
  train_flux_1x4090.sh
  train_flux_2x4090.sh
  test_flux.sh
  sweeps/
    run_all_flux_sweeps_200k.sh
    sweep_flux_tfd1_anchor05_200k.sh
    sweep_flux_tfd1_anchor1_flow03_200k.sh
    sweep_flux_lr1e5_tfd1_anchor1_flow05_200k.sh
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

Run these from the repository root. Each script uses a distinct `EXP_ID` and
trains for `200000` iterations.

Run all three sweeps sequentially. Each sweep uses two GPUs for one training
job via `torchrun --nproc_per_node=2`:

```bash
bash drifting/scripts/flux/sweeps/run_all_flux_sweeps_200k.sh
```

Run the same three sweeps sequentially on one GPU:

```bash
TRAIN_SCRIPT=drifting/scripts/flux/train_flux_1x4090.sh \
bash drifting/scripts/flux/sweeps/run_all_flux_sweeps_200k.sh
```

Use a different visible GPU pair if needed:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash drifting/scripts/flux/sweeps/run_all_flux_sweeps_200k.sh
```

To keep running the remaining sweeps after one failure:

```bash
CONTINUE_ON_ERROR=1 bash drifting/scripts/flux/sweeps/run_all_flux_sweeps_200k.sh
```

```bash
bash drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor05_200k.sh
bash drifting/scripts/flux/sweeps/sweep_flux_tfd1_anchor1_flow03_200k.sh
bash drifting/scripts/flux/sweeps/sweep_flux_lr1e5_tfd1_anchor1_flow05_200k.sh
```

Sweep configurations:

```text
flux_sweep_tfd1_anchor05_200k:
  LR=5e-5, lambda_flow=1.0, lambda_tfd=1.0, lambda_anchor=0.5

flux_sweep_tfd1_anchor1_flow03_200k:
  LR=5e-5, lambda_flow=0.3, lambda_tfd=1.0, lambda_anchor=1.0

flux_sweep_lr1e5_tfd1_anchor1_flow05_200k:
  LR=1e-5, lambda_flow=0.5, lambda_tfd=1.0, lambda_anchor=1.0
```

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
