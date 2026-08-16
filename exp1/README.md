# Experiment 1: Does Muon Q weaken BandInvMF cancellation?

The hypothesis is deliberately narrow: a BandInvMF noise transcript has useful
linear cross-iteration prefix cancellation, but applying Muon's post-Nesterov
nonlinearity \(Q\) may weaken it.  This experiment does not alter covariance
design, privacy accounting, or end-to-end DP training.

For a frozen clean post-Nesterov/pre-\(Q\) trajectory \(U_t\), the replay draws
IID \(Z_t\sim N(0,I)\), forms

\[
E = H_\beta^{\rm Nes} C^{-1} Z,
\qquad
\Delta_t^{(j)}=Q_j(U_t+E_t)-Q_j(U_t),
\qquad x_t^{(j)}=\eta_t\Delta_t^{(j)},
\]

and estimates, at each prefix \(k\),

\[
J_{j,k}=\mathbb E\left\|\sum_{t=1}^k x_t^{(j)}\right\|_F^2,
\quad D_{j,k}=\sum_{t=1}^k\mathbb E\|x_t^{(j)}\|_F^2,
\quad R_{j,k}=J_{j,k}/D_{j,k}.
\]

The primary number is the ratio of sums
\(R_j=\sum_k J_{j,k}/\sum_k D_{j,k}\), not an average of prefix or sample
ratios.  \(R_j<1\) means net prefix cancellation; smaller means stronger
cancellation.  `delta_R` is \(R_j-R_{j-1}\), so a positive value says that
the corresponding stage weakened cancellation.

`U_t` is frozen so every Monte Carlo sample and every stage sees exactly the
same real clean optimizer trajectory.  The replay must use
\(H_\beta^{\rm Nes}C^{-1}Z\), rather than only \(C^{-1}Z\), because in the
actual correlated DP-Muon mechanism BandInvMF noise is added to gradients
before classic momentum/Nesterov and only then enters \(Q\).

## Collect

The collector runs the existing CIFAR-10 Muon path without sampling DP noise
(clipping stays in place), and records one selected rank-two Muon leaf directly
after Nesterov and before `Q`:

```bash
python exp1/collect_trajectory.py \
  --parameter-name blocks/0/attention/query/kernel \
  --steps 64 --output exp1/trajectory.npz
```

The archive includes `u`, `learning_rates`, `parameter_name`, `start_step`,
`momentum`, `ns_steps`, and `consistent_rms` (plus the BF16 setting).  Select
or fit a BandInvMF strategy whose horizon is the same `T` as the trajectory:

```bash
python scripts/fit_bandinvmf.py --horizon 64 --bandwidth 4 --min-sep 1 \
  --output artifacts/strategies/bandinvmf_t64.npz
```

Then set its path in `exp1/config.yaml`.

## Replay

```bash
python exp1/run.py --config exp1/config.yaml
```

The default replay uses 1,000 seeded samples and target per-trajectory median
relative norms 0.01, 0.1, and 1.0.  Each sampled transcript receives one global
scalar so `median_t(||E_t||_F / ||U_t||_F)` equals its target; it is never
rescaled one step at a time.  Outputs are `results.csv`,
`prefix_results.csv`, and `summary.json` in `output_dir`.

Read results in this order: first verify `linear` has `R < 1`; then compare
`scale` with `linear`; finally inspect the cumulative sequence
`linear → bf16 → norm → ns → scale` and its `delta_R` values.  The code does
not encode an expectation about which stage should dominate.
