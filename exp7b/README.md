# Experiment 7b: gamma-prime-stabilized correlated DP-AdamBC

Experiment 7b leaves Experiment 7 and its results untouched.  It reuses the
same clipped query, correlated BandInvMF mechanism, privacy calibration,
fixed-cycle schedule, and four-way shadow decomposition.  The baseline is the
unchanged Experiment 7 correlated DP-AdamW trajectory.

The BC trajectory uses the DP-AdamBC paper's numerical-stability form exactly:

```text
corrected_v = vhat_private - Phi_t
denom_v = max(corrected_v, gamma_prime)
p_bc = 1 / sqrt(denom_v)
update = mhat_private * p_bc
```

There is no Adam epsilon outside the square root in the BC branch.  The fixed
default is `gamma_prime = phi_infinity`, where
`phi_infinity = iid_noise_std^2 * sum(noising_coef**2)`.  Gamma-prime is only
deterministic post-processing of the existing DP output and adds no privacy
cost.

The reported p50/p99/p99.9 are conservative upper-bin estimates from a fixed
4096-bin `[0, implied_p_max]` histogram accumulated over every coordinate and
step in each 16-step window.  `p_bc_max` is exact.

Early/late stability summaries use separate exact per-step records rather
than overlap-weighted window extrema.  Norm means/std/min/max are pooled over
the actual steps in the stage, and p50/p99/p99.9 come from a histogram pooled
over every coordinate and actual step in the stage.  The per-window CSV schema
is unchanged.  In `window_summary.csv`, cross-seed extrema remain extrema,
norm std is pooled, and averages of per-seed window quantiles are explicitly
named `mean_window_p_bc_*` rather than pooled quantiles.

Smoke run:

```bash
python exp7b/run.py --smoke --output-dir /tmp/exp7b-smoke --seeds 0 \
  --gamma-prime-ratio 1.0
```

Full CIFAR-10 run (one fixed ratio, no sweep):

```bash
python exp7b/run.py \
  --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp7b/results \
  --seeds 0 1 2 \
  --gamma-prime-ratio 1.0
```
