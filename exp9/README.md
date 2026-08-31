# Experiment 9: Muon nonlinear cancellation decomposition

Experiment 9 runs one real correlated DP-Muon trajectory per seed. The
clipped-clean gradient is retained in the train state and drives independent
shadow diagnostics on Muon-labelled rank-two leaves only.

The primary float32 paths are:

* `P0`: `s_W R_t,W`, where `s_W = consistent_rms * sqrt(max(shape_W))`;
* `P1`: JVP of the production-equivalent smooth Muon Q map;
* `P2`: antithetic odd response;
* `P3`: raw private-clean response minus `Bhat=(BA+BB)/2`, with two independent
  A/B antithetic probe replicates. The reported reliability rule is
  `probe_error_to_P3_D <= 0.1 AND probe_error_to_P3_endpoint <= 0.1`.
  `||BA-BB||/(||Bhat||+eps)` is retained only as the auxiliary
  `probe_disagreement_relative_to_bias` diagnostic.

`P3_reliable_paired = P3_reliable_corr && P3_reliable_iid`. `Delta_even` is
only strongly interpretable when this paired flag is true; otherwise its
original numeric value is retained and marked unreliable.

`G_C` and `G_J` are computed from exact raw-step endpoints. Bias and raw
private-clean gaps are reported separately and are not cancellation metrics.
If `P3_reliable` is false, `P3` and its gains remain in the output, but the
`Delta_even` interpretation is marked unreliable due to bias-probe Monte Carlo
error.
The IID branch matches the correlated branch's raw gradient-noise marginal at
each step and uses the same latent `z` as the correlated branch; it is a
mechanism control, not a formal same-DP baseline. Primary stage diagnostics
(`linear`, `norm`, `ns`, `scale`) report their own `J,D,C,G_C,G_J`. All
primary stages are consistently weighted by the final per-block Muon
consistent-RMS scale; this avoids interpreting fixed block reweighting as
nonlinear degradation. BF16 is secondary precision only.

Smoke check:

```bash
conda run -n curve python exp9/run.py --smoke --output-dir /tmp/exp9-smoke --seeds 0 --bias-probes 2
```

Formal run:

```bash
conda run -n curve python exp9/run.py \
  --config config/cifar10_bandinv_dpmuon_naive.yaml \
  --output-dir exp9/results \
  --seeds 0 1 2 \
  --bias-probes 8
```
