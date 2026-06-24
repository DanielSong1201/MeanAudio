# MeanAudio-Student Drifting Distillation

This is the original drifting route:

```text
FluxAudio-S-Full teacher -> MeanAudio-S one-step student
```

The implementation is shared with the compatibility entrypoints at
`drifting/train.py` and `drifting/test.py`. This directory only provides a
clear path parallel to `drifting/flux/`.

## Train

From the repository root:

```bash
bash drifting/scripts/mean/train_mean_1x4090.sh
```

Common debug run:

```bash
EXP_ID=mean_debug \
BATCH_SIZE=2 \
ITERATIONS=100 \
LOG_INTERVAL=10 \
EVAL_INTERVAL=0 \
bash drifting/scripts/mean/train_mean_1x4090.sh
```

Training logs and metrics are written to:

```text
exps/drifting/<exp_id>/train.log
exps/drifting/<exp_id>/metrics.csv
exps/drifting/<exp_id>/eval_metrics.csv
```

Weights are saved to:

```text
exps/drifting/<exp_id>/<exp_id>_<iteration>.pth
exps/drifting/<exp_id>/<exp_id>_last.pth
```

## Training-Time Eval

By default, training runs a full eval every 10000 iterations using the
MeanAudio one-step eval path:

```text
eval.py --variant meanaudio_s --use_meanflow
av-benchmark/evaluate.py
```

Eval artifacts are stored under:

```text
exps/drifting_eval/<exp_id>/it_<iteration>/
  audio/
  cache/
  evaluate.log
  eval_driver.log
```

Use environment variables to customize:

```bash
EVAL_INTERVAL=5000 \
EVAL_NUM_STEPS=1 \
EVAL_CFG_STRENGTH=0.9 \
bash drifting/scripts/mean/train_mean_1x4090.sh
```

Set `EVAL_INTERVAL=0` to disable training-time eval.

## Tests

Synthetic drifting loss test:

```bash
MODE=loss bash drifting/scripts/mean/test_mean.sh
```

Teacher feature extraction test:

```bash
MODE=teacher bash drifting/scripts/mean/test_mean.sh
```

Manual full eval:

```bash
MODEL_PATH=exps/drifting/mean_debug/mean_debug_last.pth \
OUTPUT_PATH=exps/drifting_eval/manual_mean_debug \
bash drifting/scripts/mean/eval_mean_checkpoint.sh
```
