# Experiment 10: real MF/IID DP-AdamW second-moment diagnostics

Exp10 runs two real, closed-loop trajectories in one JAX training state:

- `mf`: clipped clean gradient at the MF branch's current parameters, plus
  the BandInvMF causal-filter output;
- `iid`: clipped clean gradient at the IID branch's current parameters, plus
  `sqrt(phi_t) * z_t`.

Both branches start from the same parameters and consume the same logical
batch sequence. Each optimizer step samples one standard-normal latent tree
`z_t`; the MF filter and the matched-marginal IID construction consume that
same tree. There is no shadow trajectory and no independent diagnostic noise.
Each branch has its own parameters and standard AdamW moments. The existing
naive DP-AdamW optimizer is used unchanged: no noise subtraction, oracle
preconditioner correction, or other correction is applied to the update.

For each branch and step, the component definitions are:

```text
g2       = g_t ** 2
g2_cross = g_t ** 2 + 2 * g_t * xi_t
xi2      = xi_t ** 2
```

`V_g`, `V_g_cross`, and `V_xi` are beta2-matched EMAs of those components,
reported with Adam's `1 - beta2**t` bias correction. After every update Exp10
reads the actual Optax Adam `nu / (1 - beta2**t)` and records the maximum
absolute and RMS error in

```text
private_v_hat ~= V_g_cross + V_xi
```

The CSV metrics include the requested means, RMS, negative fraction, four
ratios, and the explicit `mean_g2_cross_minus_g2` negative-control/feedback
quantity. Stage rows use exact raw-step aggregation rather than averaging
per-step ratios:

- `early`: steps 1--97 (or 1--horizon for a shorter run);
- `late`: steps 98--horizon (omitted when `horizon < 98`);
- `full`: steps 1--horizon.

`paired_stage_metrics.csv` pairs the MF and IID row from the same seed and
stage. It records `delta_traj`, `delta_feedback`, `delta_noise`,
`delta_total`, and `delta_decomposition_residual`, where

```text
delta_total ~= delta_traj + delta_feedback + delta_noise
```

`summary.json` contains `paired_stage_aggregate`. Every paired metric has
cross-seed `mean`, sample `std`, `se`, and `ci95_low`/`ci95_high`. For two or
more seeds the CI uses `scipy.stats.t.ppf(.975, n-1)`; if SciPy is unavailable
the implementation falls back to the normal critical value 1.96. For one
seed the standard error and CI width are zero. `feedback_summary` separately
reports the IID negative control `mean_2gxi_iid`, MF feedback
`mean_2gxi_mf`, and paired `delta_feedback` with the same uncertainty fields.

## Output format

Each output directory contains:

- `summary.json`: resolved run metadata, `phi_t`, stage aggregates, and the
  histogram schema/check information. Its `expectation_checks` section
  explicitly reports the IID `E[g2_cross-g2]` negative control, MF
  `E[2*g*xi]` feedback quantity, and cross-seed means of the
  `private_v_hat-(V_g_cross+V_xi)` maximum error;
- `step_metrics.csv`: one row per `(seed, step, branch)`;
- `stage_metrics.csv`: one row per `(seed, stage, branch)`;
- `paired_stage_metrics.csv`: one paired row per `(seed, stage)`;
- `histograms.npz`: optional per-seed compact histogram data;
- `pooled_histograms.npz`: required cross-seed pooled histogram data;
- `histograms.png`: pooled comparison of the last checkpoint;
- `paired_statistics.png`: paired delta plot with cross-seed 95% CI.

Both NPZ artifacts use `format_version=exp10-histograms-v2` and contain:

```text
steps                 (K,)
group_bin_edges       (K, 4, bins+1)
counts                (K, 2, 4, 2, bins)
relative_frequency    (K, 2, 4, 2, bins)
branch_names          ["mf", "iid"]
group_names           ["instantaneous_signal_cross", "instantaneous_noise",
                       "ema_signal_cross", "ema_noise"]
group_component_names [["g2", "g2_cross"], ["xi2", ""],
                       ["V_g", "V_g_cross"], ["V_xi", ""]]
```

The four groups deliberately avoid scale contamination:

- Group A: `g2`, `g2_cross` share bins across MF/IID/all seeds;
- Group B: `xi2` has independent bins;
- Group C: `V_g`, `V_g_cross` share bins across MF/IID/all seeds;
- Group D: `V_xi` has independent bins.

`g2_cross` and `V_g_cross` are histogrammed directly, so negative values are
retained; no floor or clipping is applied. Histogram counts remain in the
original linear value domain. The runner uses two passes: pass one keeps only
per-group extrema, and pass two replays the deterministic trajectories to
accumulate raw counts with common cross-seed edges. The pooled artifact sums
raw counts first and computes relative frequencies afterward. No full
parameter-coordinate tensor is persisted.

Histograms are stored at steps 16, 32, ..., horizon, with the final step added
when horizon is not a multiple of 16. Without `seed`, the plotting helper
automatically prefers sibling `pooled_histograms.npz`; with an explicit seed it
uses `histograms.npz`. `xscale="symlog"` is available for display only—the
stored counts remain linear-domain counts.

The plotting helper can be used independently:

```bash
conda run -n curve python -c \
  'from exp10.plotting import plot_histograms; plot_histograms("exp10/results/histograms.npz", "exp10/results/histograms_last.png")'
```

## Interpretation and privacy disclaimers

The IID branch is a `matched-marginal mechanism control`, not a `formal
same-(epsilon,delta) IID DP baseline`. It reuses the MF calibration and
matches the MF row marginal variance; it is not calibrated with a separate IID
privacy accountant.

Exp10 records clean clipped-gradient-derived quantities such as `g^2` and
`g*xi`. These diagnostic artifacts are not DP releases. Exp10 is an internal
research diagnostic, and its statistics must not be treated as a
privacy-protected published mechanism.

## Checks

Run focused tests and the tiny synthetic smoke job with the required
environment:

```bash
conda run -n curve pytest exp10/tests
conda run -n curve python exp10/run.py --smoke --output-dir exp10/results_smoke --seeds 0
```

Do not use the smoke job as a CIFAR accuracy result. It only validates the
paired closed-loop mechanics and output artifacts without loading CIFAR-10.

## Formal CIFAR-10 run

The requested ten-seed run is:

```bash
conda run -n curve python exp10/run.py \
  --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp10/results \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

This command is intentionally not run as part of the implementation check.
