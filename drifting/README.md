# DriftingAudio

This directory contains the FluxAudio teacher-feature drifting experiment entrypoints.

The layout follows the rest of this repository:

```text
drifting/
  train.py                         # compatibility entrypoint for MeanAudio-student training
  test.py                          # compatibility entrypoint for MeanAudio-student testing/eval
  mean/                            # MeanAudio-S one-step student route
  flux/                            # FluxAudio-S one-step student route
  scripts/
    train_drifting_1x4090.sh
    test_drifting.sh
    eval_drifting_checkpoint.sh
    mean/
      train_mean_1x4090.sh
      test_mean.sh
      eval_mean_checkpoint.sh
    flux/
      train_flux_1x4090.sh
      test_flux.sh
```

## Training

Original MeanAudio-student route:

```bash
bash drifting/scripts/mean/train_mean_1x4090.sh
```

Pure FluxAudio-student route:

```bash
bash drifting/scripts/flux/train_flux_1x4090.sh
```

Compatibility entrypoint for the original MeanAudio-student route:

Run from the repository root:

```bash
bash drifting/scripts/train_drifting_1x4090.sh
```

The default training length is `1000000` iterations. Override it with
`ITERATIONS=<steps>`.

The training entrypoint writes logs and metrics to:

```text
exps/drifting/<exp_id>/train.log
exps/drifting/<exp_id>/metrics.csv
exps/drifting/<exp_id>/eval_metrics.csv
```

This is the required training log outlet. `metrics.csv` records training losses.
`eval_metrics.csv` records full evaluation runs with the iteration, checkpoint,
evaluation output path, `evaluate.log`, and parsed metrics JSON. The final
weights are saved as:

```text
exps/drifting/<exp_id>/<exp_id>_last.pth
exps/drifting/<exp_id>/<exp_id>_ema_last.pth
```

For the MeanAudio-student route, EMA is enabled by default and training-time
eval uses EMA weights unless `--eval-raw` is passed to `drifting/train.py`.
The raw checkpoint is still saved for comparison.

By default, training runs a full evaluation every 10000 iterations using the
same method as `drifting/scripts/eval_drifting_checkpoint.sh`: it calls
`eval.py` to generate AudioCaps audio and then `av-benchmark/evaluate.py` to
write evaluator output. Per-iteration eval artifacts are stored under:

```text
exps/drifting_eval/<exp_id>/it_<iteration>/
  audio/
  cache/
  evaluate.log
  eval_driver.log
```

Common evaluation overrides:

```bash
EVAL_INTERVAL=5000 \
EVAL_NUM_STEPS=1 \
EVAL_CFG_STRENGTH=0.9 \
EVAL_OUTPUT_ROOT=exps/drifting_eval \
bash drifting/scripts/train_drifting_1x4090.sh
```

Set `EVAL_INTERVAL=0` to disable training-time evaluation.

Common overrides:

```bash
EXP_ID=drifting_debug \
BATCH_SIZE=2 \
ITERATIONS=100 \
LOG_INTERVAL=10 \
bash drifting/scripts/train_drifting_1x4090.sh
```

## Testing

Synthetic loss and gradient test:

```bash
bash drifting/scripts/test_drifting.sh
```

Teacher feature extraction test:

```bash
MODE=teacher bash drifting/scripts/test_drifting.sh
```

Evaluate a trained checkpoint:

```bash
MODEL_PATH=exps/drifting/drifting_debug/drifting_debug_last.pth \
OUTPUT_PATH=exps/drifting_eval/drifting_debug \
bash drifting/scripts/eval_drifting_checkpoint.sh
```

The evaluation wrapper reuses `eval.py` and `av-benchmark/evaluate.py`, and writes the evaluator output to:

```text
<OUTPUT_PATH>/evaluate.log
```
