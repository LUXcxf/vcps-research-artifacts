# Data-scale records

These files contain the D30, D60, and D120 training records used for the current scale-adapted public reports. The D240 records are the primary datasets under `data/actor/` and `data/feedback/`.

The actor datasets are nested by training episode. Verifier-feedback pair counts and the predeclared scale-adapted training settings are recorded in `configs/data_scales.json`. Each scale uses a fresh actor and a verifier residual trained from zero over all observed train-side features; the scale reports therefore document the selected scale-adapted protocol rather than a one-variable causal learning curve.

The current valid-unseen134 VCPS completion counts are D30 97/134, D60 103/134, D120 109/134, and D240 124/134. The corresponding episode reports are under `results/reports/`. Supervision-budget sensitivity experiments and their unfinished curves are not part of this release.
