# Experiment 7: second-moment decomposition and correlated AdamBC

Experiment 7 extends Experiment 6's online, non-overlapping 16-step
cancellation diagnostics.  Each real trajectory performs exactly one clipped
gradient query and one BandInvMF sample per logical step.  All shadow moments
consume those already-computed values.

The baseline trajectory maintains `v00`, `v10`, `v01`, and `v11` for the clean,
cross-only, square-only, and full noisy second moments.  It also subtracts the
exact time-varying BandInvMF marginal variance in a DP-AdamBC shadow.  For FIR
coefficients `d = noising_coef` and latent standard deviation `sigma`, step
`t` has `phi[t] = sigma^2 * sum(d[:min(t+1, bandwidth)]^2)`.

The paired BC trajectory uses the same initialization, fixed-cycle batches,
and RNG key as baseline.  Its only update change is replacing `vhat11` by
`max(vhat11 - Phi_t, v_floor)` in the real preconditioner.  The default
`v_floor` is `1e-30`; it only prevents an invalid square root and does not
alter clipping, noise calibration, or privacy accounting.

Smoke run:

```bash
python exp7/run.py --smoke --output-dir /tmp/exp7-smoke --seeds 0
```

Full CIFAR-10 run:

```bash
python exp7/run.py \
  --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp7/results \
  --seeds 0 1 2
```
