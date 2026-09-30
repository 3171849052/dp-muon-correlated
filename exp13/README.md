# Exp13: numerical study and closed-loop CIFAR-10 training

Run the two independent stages from the repository root, in order:

```bash
bash exp13/run_numerical.sh
bash exp13/run_training.sh
```

Both use conda environment `curve`, existing `data/`, and the local checkpoint
in `config.yaml`. No data downloads or automatic artifact regeneration occur.
`launch_full.sh` is now an alias for Stage 2 only.

## Stage 1 (GPU 1)

`numerical.py` runs fitting and cross-evaluation, then collects the frozen
trajectory once, then runs replay and creates numerical tables/plots.
Outputs are under `results/stage1_numerical/`:

- `strategies/{C_m_raw,C_m_bc,C_v_raw,C_v_bc}.npz`
- `strategy_summary.{csv,json}`, `cross_eval.{csv,json}` (all 16 cells)
- `trajectory/gradients_sampled.npy`, `coordinate_indices.npy`, `metadata.json`
- `replay/metrics.{csv,json}`, `per_step_diagnostics.{csv,json}`,
  `workload_validation.{csv,json}`, `metrics.png`, `provenance.json`
- `provenance.json` with stage and subprocess timings

The streaming workload is `H[t,s]=(1-beta)*beta**(t-s)` for `s<=t`.
The BC workload divides row `t` by `1-beta**(t+1)` (zero-based indexing).
Official `jax_privacy.banded.optimize` uses mean reduction, the configured
bandwidth (47 in the current config), 1000 fit steps, column normalization,
and the same participation sensitivity constraints for raw and BC. Bias
correction changes utility only.

The trajectory is **clipped DP-query gradients evaluated along a clean Adam
parameter trajectory**. At the same pre-update parameters and batch, collection
computes the clipped query and the gradient of the entire batch's mean loss.
Only the latter advances the parameters. Uniform sampling without replacement
selects 262144 global PyTree coordinates with seed 1203. Only this sampled
trajectory is saved; replay never calls collection.

Replay compares `iid_ime`, `bandmf_ime_sep_raw`, `bandmf_ime_sep_vbc`, and
`bandmf_ime_sep_bc`, with paired latent keys, 3 draws and chunks of 32768.
GPU reductions return per-step sufficient statistics, minima and maxima.
Only a deterministic 8192-coordinate subsample returns denominators and absolute
directions for quantiles. There are no full-coordinate direction/denominator
memmaps. Quantiles are computed from a deterministic coordinate subsample.
Threshold fractions and extrema use all 262144 replay coordinates.
The quantile suffixes p0001/p001/p01/p1 denote probabilities .0001/.001/.01/.1.

Raw and bias-corrected linear theoretical MSE are computed as
`sigma**2 * mean(per_query_error(C,A=H))`. Absolute second-moment readout is
nonlinear and is reported separately, without applying the linear prediction.

## Stage 2 (GPUs 1, 2, 3)

Stage 2 validates all four existing Stage 1 artifacts against current metadata
and configuration hash. Missing or inconsistent artifacts fail immediately.
It never fits, cross-evaluates, collects or replays. Three workers run method-major,
seed-minor jobs with seeds `[0,1,2]`: 21 jobs total, exactly 7 per GPU.
Any worker failure causes nonzero launcher exit. Aggregation requires all 21
summaries before producing tables and `accuracy.png`.

| Method | Definition |
|---|---|
| nonprivate_adam | Clean whole-batch mean-loss gradient; standard Adam; no clipping, noise or privacy calibration |
| iid_adam | Repository `calibrate_nonamplified_iid` and `make_nonamplified_dpadamw_train_step`, weight decay 0 |
| bandmf_single_m | C_m_raw first channel; square of private first gradient for second moment |
| iid_ime | Identity / identity |
| bandmf_ime_sep_raw | C_m_raw / C_v_raw |
| bandmf_ime_sep_vbc | C_m_raw / C_v_bc |
| bandmf_ime_sep_bc | C_m_bc / C_v_bc |

IME keeps the linear raw recurrence and uses `abs(v_hat_raw)` only in the
readout. It splits GDP equally (`mu1=mu2=mu/sqrt(2)`), with sensitivities
`a1=L/B`, `a2=(2B-1)*L**2/B**2`, and noise `sigma_i=a_i*sqrt(k)/mu_i`.
All privacy accounting is fixed-cycle, full-transcript, non-amplified GDP.
IID epoch privacy uses `epsilon_spent_for_iid_prefix`.
Adam epsilon remains **1e-8**, beta1=.9, beta2=.999, learning rate .005.
No optimizer epsilon search or automatic hyperparameter adjustment is used.

Outputs under `results/stage2_training/` include per-run `training/`,
`worker_logs/`, `runs`, `aggregate`, `paired_differences`, and `accuracy.png`.
All five utility metrics report n, mean, sample std, SE, and Student-t 95% CI.
Paired comparisons use matching seeds for single_m minus iid_adam, raw minus
iid_ime, vbc minus raw, bc minus vbc, and bc minus iid_ime.

## Smoke

```bash
conda run -n curve pytest exp13/tests
bash exp13/run_numerical.sh --smoke
bash exp13/run_training.sh --smoke
```

Smoke uses 64 local train/test examples, one epoch of four batches of 16,
five fitting steps, and two replay draws. Coordinate sample sizes remain the
same as formal runs. Training smoke exercises all 21 jobs on the same GPUs.
Outputs have the same stage layout under `results_smoke/`.
Tests use synthetic inputs and do not require preexisting smoke output.
