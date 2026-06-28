# Post-Sweep Modification Plan

This note records practical follow-up changes after the three FluxAudio drifting
sweeps finish. It is intended as a local checklist before changing the training
code again.

## First: summarize the sweep results

Before changing the model or losses, parse the three experiments' `eval_metrics.csv`
files and compare:

- best checkpoint per experiment
- moving average over the latest 3 or 5 evaluations
- raw checkpoint versus EMA checkpoint behavior
- metric trend slope, not only the single best value
- output path and checkpoint path for every selected iteration

If one checkpoint looks unusually good, re-run evaluation for the same checkpoint
once before treating it as real progress.

## Case 1: all sweeps improve slowly

Likely interpretation: the direction is useful, but optimization is inefficient.

Recommended changes:

- add a learning-rate schedule with warmup and cosine decay
- support gradient accumulation to increase effective batch size on 4090 GPUs
- keep full evaluation every 10k iterations
- optionally reduce checkpoint save frequency to 10k if 1k checkpoint files are
  not useful
- schedule the loss weights: start from stronger base generation loss, then ramp
  up TFD and anchor losses

Highest-priority implementation:

- add `--gradient-accumulation-steps`
- add `--lr-warmup-steps`
- add `--lr-schedule`
- add schedules for `lambda_tfd` and `lambda_anchor`

## Case 2: stronger TFD or anchor sweep wins

Likely interpretation: teacher hidden-state constraints are useful.

Recommended changes:

- ramp up `lambda_tfd` instead of using the full value from iteration 0
- ramp up `lambda_anchor` later than `lambda_tfd`
- test more feature-layer choices
- add local temporal feature alignment instead of relying only on pooled/global
  features

Suggested next sweep:

- keep the winning learning rate and base loss weight
- sweep feature layers around middle layers
- compare global-only TFD versus global plus local temporal TFD

## Case 3: stronger TFD or anchor sweep is worse

Likely interpretation: the teacher-feature loss is too strong or too early.

Recommended changes:

- lower `lambda_tfd`
- lower or delay `lambda_anchor`
- keep the base flow/generation loss dominant early in training
- first train a stable one-step Flux student, then fine-tune with TFD

This means TFD should be treated as a regularizer, not the main training signal.

## Case 4: lower learning rate wins clearly

Likely interpretation: one-step distillation is sensitive to parameter updates.

Recommended changes:

- avoid simply lowering LR forever
- use warmup plus decay
- optionally compare `EMA_DECAY=0.999` against `EMA_DECAY=0.9999`
- keep EMA evaluation as the default

## Case 5: all sweeps are poor

Likely interpretation: this is not only a scalar hyperparameter problem.

Recommended structural changes:

- add a feature memory bank so TFD is not limited to the current small batch
- add local temporal teacher-feature losses
- optionally add teacher-generated positives after the real-latent baseline is
  stable
- move to a two-stage recipe:
  1. train a stable one-step student with base loss
  2. fine-tune with TFD and anchor-margin coverage

## Recommended next implementation package

If there is no obvious winner after the three 200k sweeps, implement a phase-3
training upgrade with:

- sweep result summarizer for `eval_metrics.csv`
- repeat-eval helper for a selected checkpoint
- gradient accumulation
- LR warmup and cosine decay
- scheduled `lambda_tfd` and `lambda_anchor`
- optional local temporal feature drifting
- EMA evaluation kept as default
- full evaluation kept at every 10k iterations unless overridden

This package should make the next experiments diagnose whether the bottleneck is
optimization speed, evaluation noise, loss weighting, or the current feature
space design.
