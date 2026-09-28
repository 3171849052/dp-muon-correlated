# Exp12: standard Adam with paired temporal Gaussian mechanisms

All experiment files and outputs live under `exp12/`. Existing code is imported
for CIFAR preprocessing, pretrained ViT-Tiny, fixed-cycle participation and batch
ordering, global per-example clipping, BandInvMF sensitivity/fitting, streaming
noise, and non-amplified GDP calibration. CIFAR loading always sets
`download=False`; checkpoint loading uses the existing local NPZ. No source
outside this directory is changed.

## Methods and objectives

Every training run uses `optax.adam(lr=0.005, b1=0.9, b2=0.999, eps=1e-8)`
(with Optax's `learning_rate` keyword), no weight decay, clipping 1, five epochs,
logical batch 512 and microbatch 256. Seeds are 0 through 9.

* `dp_adam`: IID Gaussian noise, independently calibrated for the full transcript.
* `bandinv_momentum`: minimize `J_mom = ||L H_beta1 S||_F² / T`, with
  `H[t,s]=(1-beta1) beta1^(t-s)`, without bias correction or learning-rate scale.
* `adam_exact`: minimize `J_exact = E ||L R_Adam(sigma_corr S z)||² / T`.
  R uses full Adam first/second moments and both bias corrections, including eps.
* `adam_mf`: minimize `J_mf = ||L D(S) H_beta1_BC sigma_corr S||_F² / T`,
  where `D=diag(1/(sqrt(vbar_hat)+eps))` and
  `vbar_hat=H_beta2_BC [sigma_corr² diag(SSᵀ)]`.

The strategy objectives are per parameter coordinate; K counts independent scalar
Gaussian trajectories. L is the lower-triangular prefix-sum matrix. All correlated S are lower-triangular
Toeplitz with four coefficients, S=C⁻¹. Each objective evaluation normalizes
`S = Delta(S_raw⁻¹) S_raw`, so `Delta(S⁻¹)=1`. Sensitivity is the existing
`jax_privacy.compute_banded_inverse_sensitivity_squared` convention, with
`use_matrix_upper_bound=False` (absolute coefficient non-increasing majorant
when the exact monotone-coefficient formula does not apply). The same convention
is used for fitting, artifacts, and prefix privacy accounting.

Momentum uses the repository fitter. Exact/MF directly optimize their objectives
with JAX autodiff and 1000 Adam steps at 0.001, starting from momentum. The raw
diagonal is fixed at 1 to remove redundant global scale. The lowest fitting loss
iterate is saved; these are numerical local solutions, not global-optimum claims.
Exact fitting fixes 256 common Gaussian samples (seed 1200). Independent evaluation
uses 4096 samples (seed 1201), shared across all strategies. Cross-evaluation saves
all three objectives and `mf_exact_gap=J_exact(adam_mf)/J_exact(adam_exact)-1`.
Artifacts include raw/normalized noising coefficients, inverse strategy
coefficients, sensitivities before/after normalization, objective, configuration,
optimizer trace, and independent evaluation metadata.

## Privacy and pairing

The repository GDP calibrator targets epsilon=3, delta=1e-5, add/remove adjacency,
without amplification. Correlated latent standard deviation is
`(clip_norm/batch_size)/mu` after normalization; IID uses
`(clip_norm/batch_size)*sqrt(max_participations)/mu`. Horizon, minimum separation,
and participation cap come from the actual local dataset and fixed-cycle helper.
The resulting formal CIFAR contract is also saved to `results/contract.json`.

Within each seed, initialization key, batch schedule, clipping and optimizer are
identical. All mechanisms use the same streaming Gaussian sampler and noise key;
the IID filter has coefficient [1], so latent draws are paired even though noise
scales and filters differ. Each run saves the schedule and checkpoint SHA256,
strategy metadata and calibration. Epoch metrics include train loss, test loss,
test accuracy, best metrics, actual prefix epsilon and elapsed seconds.

## Frozen-gradient replay and metrics

One clean **Adam** run (seed 0) records every parameter's post-clipping,
pre-optimizer mean gradient in `replay/gradients.npy`, a disk-backed array.
Replay uses ten paired Gaussian draws (seed 1202), processed in coordinate blocks
of 8192. Moments start at zero and the exact nonlinear Adam recurrence is replayed
on `g+noise` against the same clean trajectory. All norms below include **all
parameters**, not a selected layer, and are not divided by parameter count.

* Parameter trajectory RMSE: sqrt(mean over steps/draws of squared L2 parameter
  displacement); displacement includes the learning rate.
* Endpoint RMSE: sqrt(mean over draws of final squared L2 displacement).
* Direction RMSE: sqrt(mean over steps/draws of squared L2 Adam-direction error).
* Update cosine: mean cosine of noisy and clean directions across steps/draws.
* Sign agreement: fraction of matching signs across coordinates, steps and draws.
* Preconditioner RMSE: sqrt(mean squared L2 error in `1/(sqrt(vhat)+eps)`).
* Noise/gradient ratio: `||noise_t||/(||g_t||+small_eps)`; mean, median, p10, p90.
  Global energy ratio is `sum ||noise||² / sum ||g||²`.

The optional Exp2 J_k/D_k/R_k diagnostics are omitted. Replay saves metrics as
CSV/JSON, coordinate-summed sufficient statistics, provenance, and a PNG figure.
Clean gradients and train-set diagnostics are internal experiment artifacts;
the reported privacy calibration describes the training mechanism.

Utility aggregation uses final/best test accuracy, final/best test loss, and
normalized accuracy AUC (trapezoidal effective-epoch integration from the actual
initial test accuracy through the final evaluation). It reports mean, sample standard
deviation (ddof=1), SE, and Student-t 95% CI across ten seeds. Six paired-seed
comparisons cover every unordered method pair, with the named A-B direction;
positive accuracy differences favor A. Loss differences retain A-B too.

## Outputs and execution

`results/strategies/{bandinv_momentum,adam_exact,adam_mf}.npz`;
`results/cross_eval.{csv,json}`; `results/replay/`;
`results/training/<method>_seed<seed>/{metadata,metrics,summary}.json` and metrics CSV;
`results/{runs,aggregate,paired_differences}.{csv,json}`; `results/accuracy.png`;
`results/worker_logs/{launcher,gpu0,gpu1,gpu2,gpu3}.log`.

The deterministic job list is method-major then seed-major. Job i belongs to
GPU i%4; each worker sees only its assigned GPU and runs its ten jobs serially.
No resume or task skipping occurs. Failed runs propagate a nonzero worker status;
the launcher waits for all four workers and aggregates only after all succeed.
The launcher first fits/cross-evaluates strategies, collects gradients and runs
replay on GPU 0, then launches the four training workers.

From the repository root, tests and explicit small smoke runs:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve pytest exp12/tests
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp12/fit_strategies.py --smoke
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp12/replay.py --smoke
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp12/full_training.py --smoke
```

Smoke uses 64 existing examples, one epoch, batch/microbatch 16, four steps,
five fit steps, K_fit=16, K_eval=64 and two replay draws. It runs the actual
pretrained ViT and all four training mechanisms; outputs go only to
`exp12/results_smoke/`. Full training settings are never changed automatically.

The complete experiment (conda environment and all four GPUs selected internally):

```bash
bash exp12/launch_full.sh
```

Implementation files: `config.yaml`, `common.py`, `adam.py`, `objectives.py`,
`strategies.py`, `fit_strategies.py`, `cross_eval.py`, `collect_trajectory.py`,
`replay.py`, `full_training.py`, `worker.py`, `aggregate.py`, `plotting.py`,
`launch_full.sh`, `__init__.py`, `pytest.ini`, and `tests/` (seven test modules
plus `conftest.py`).
