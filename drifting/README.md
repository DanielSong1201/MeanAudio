# DriftingAudio

This directory contains the FluxAudio teacher-feature drifting experiment entrypoints.

The layout follows the rest of this repository:

```text
drifting/
  train.py                         # training entrypoint
  test.py                          # testing/evaluation entrypoint
  scripts/
    train_drifting_1x4090.sh
    test_drifting.sh
    eval_drifting_checkpoint.sh
```

## Training

Run from the repository root:

```bash
bash drifting/scripts/train_drifting_1x4090.sh
```

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
```

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
