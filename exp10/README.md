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
- `late`: steps 98--horizon;
- `full`: steps 1--horizon.

## Output format

Each output directory contains:

- `summary.json`: resolved run metadata, `phi_t`, stage aggregates, and the
  histogram schema/check information. Its `expectation_checks` section
  explicitly reports the IID `E[g2_cross-g2]` negative control, MF
  `E[2*g*xi]` feedback quantity, and cross-seed means of the
  `private_v_hat-(V_g_cross+V_xi)` maximum error;
- `step_metrics.csv`: one row per `(seed, step, branch)`;
- `stage_metrics.csv`: one row per `(seed, stage, branch)`;
- `histograms.npz`: compact histogram data;
- `histograms.png`: a plot of the last stored checkpoint (when checkpoints
  exist).

`histograms.npz` uses `format_version=exp10-histograms-v1` and contains:

```text
seeds                 (K,)
steps                 (K,)
bin_edges             (K, bins+1)
counts                (K, 2, 6, bins)
relative_frequency    (K, 2, 6, bins)
branch_names          ["mf", "iid"]
component_names       ["g2", "g2_cross", "xi2", "V_g", "V_g_cross", "V_xi"]
```

For every `(seed, step)`, the one `bin_edges` row is shared by both branches
and all six components. `g2_cross` is passed to `numpy.histogram` directly,
so negative values are retained; no floor or clipping is applied. Only edges
and counts/frequencies are saved, never full parameter-coordinate arrays.
Histograms are stored at steps 16, 32, ..., horizon, with the final step added
when horizon is not a multiple of 16.

The plotting helper can be used independently:

```bash
conda run -n curve python -c \
  'from exp10.plotting import plot_histograms; plot_histograms("exp10/results/histograms.npz", "exp10/results/histograms_last.png")'
```

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
