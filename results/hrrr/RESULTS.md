# HRRR 8x super-resolution: physics vs no-physics vs bicubic

Run: `scripts/run_hrrr_experiment.py` on one A100, 3 seeds, 8000 steps, bicubic
degradation. Full per-seed numbers in `summary.json`.

## Headline (mean over 3 seeds)

| metric      | bicubic | no-physics | physics |
|-------------|---------|------------|---------|
| RMSE        | 0.290   | 0.843      | 0.868   |
| PSNR (dB)   | 48.1    | 38.8       | 38.6    |
| grad_rmse   | 0.238   | 0.475      | 0.494   |
| hf_rmse     | 8.34    | 22.60      | 23.68   |
| mass_resid  | 0.0029  | 0.568      | 0.606   |
| div_resid   | 0.0029  | 0.0916     | 0.0972  |

Physics conformal coverage (nominal 0.90): 0.934.

## Honest read

Two findings, both negative for "physics helps here":

1. The physics-constrained model is slightly worse than the unconstrained model on
   every metric. The physics term is active (it moves the metrics), so this is a
   real effect, not a dead-gradient artifact.
2. Plain bicubic upsampling beats both learned models by roughly 3x on RMSE.

The cause is the testbed, not the method. The low-resolution input is produced by
bicubic downsampling, and bicubic-down followed by bicubic-up is close to identity
on these smooth wind and precipitation fields. That makes interpolation
near-optimal and leaves the learned models solving a near-degenerate task badly,
with the physics penalties only adding constraint error on top. This says the HRRR
8x super-resolution setup as configured cannot demonstrate the value of physics
constraints; it does not show that physics constraints are harmful in general.

The one result worth keeping is the calibrated uncertainty: split-conformal
coverage lands at 0.934 against a 0.90 nominal target, independent of the
physics-vs-no-physics question.

## Status

HRRR 8x super-resolution is retired as a de-risking vehicle for the
physics-constrained claim. The de-risking milestone (a real public-benchmark beat
with calibrated coverage) moves to the xView3-SAR dark-vessel detection track.

Earlier avg_pool-degradation numbers, in which physics appeared to win, are not
reported here: that degradation makes bicubic structurally near-optimal, so the
comparison is not trustworthy. The raw avg_pool run remains in scratch for the
record but is intentionally not committed.
