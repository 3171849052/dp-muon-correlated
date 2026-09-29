# Exp13: Adam, single-channel BandMF, and separate-matrix IME

All implementation and generated artifacts live in `exp13/`. The experiment
imports existing `src/dp_muon` data, pretrained ViT-Tiny, global per-example
clipping, fixed-cycle scheduling and non-amplified GDP calibration. Data is loaded
from local `data/` with `download=False`; the checkpoint is the local exp12 NPZ.

The main configuration matches exp12: five epochs, batch 512, microbatch 256,
clip 1, Adam beta1=.9, beta2=.999, eps=1e-8, learning rate .005, epsilon=3,
delta=1e-5, bandwidth=4, seeds 0–9. There is no amplification or weight decay.
For 50,000 examples the actual contract is horizon 488, min_sep 97, k=5.

## Mechanisms

- `nonprivate_adam`: standard Adam on the clipped batch mean g.
- `iid_adam`: standard Adam on g + IID Gaussian noise.
- `bandmf_single_m`: standard Adam on g + C_m^{-1} z.
- `iid_ime`: separate independent channels for g and g squared.
- `bandmf_ime_sep`: C_m^{-1} z1 for g, C_v^{-1} z2 for g squared.

IME squares the clipped batch mean, not individual gradients. Its raw second
state is linear: v_raw = beta2 v_raw + (1-beta2) private_q. Bias correction is
applied at readout; only the denominator uses max(v_hat_raw, 0). Projection never
feeds back into the recurrence. With eps=1e-8, negative second-state estimates
can cause very large updates; the experiment preserves the requested formula.
No shared-C IME is implemented.

Fixed-denominator zero-out/add-remove sensitivities are a1=L/B and
 a2=(2B-1)L²/B². Single-channel methods use the full calibrated GDP mu. Dual
channels each use mu/sqrt(2), with latent stddev a_i sqrt(k)/mu_i. Identity and
column-normalized banded strategies both have temporal sensitivity squared k.

## BandMF and replay

`workloads.py` implements H_beta[t,s]=(1-beta) beta^(t-s) as the installed
jax_privacy StreamingMatrix. No bias correction, prefix sum, learning-rate,
weight-decay or denominator factors enter fitting. `strategies.py` directly
calls installed `banded.optimize`, `ColumnNormalizedBanded`, `per_query_error`,
`minsep_sensitivity_squared`, and `inverse_as_streaming_matrix`; noise uses
`noise_addition.matrix_factorization_privatizer`. The inspected version is
2.3.0.dev0. The general optimizer uses its installed optimization defaults and
1000 maximum steps (five in smoke). Strategies must satisfy bands <= min_sep.

`fit_strategies.py` saves params, materialized C, objective, beta, horizon,
bandwidth, min_sep, participation cap, sensitivity and library version in
`results/strategies/{C_m,C_v}.npz`. It saves all four cross-evaluation cells as
CSV/JSON and `1-J_v(C_v)/J_v(C_m)` in `specialization_gain.json`.

`replay.py` first collects a clean Adam trajectory using all model parameters,
then replays frozen clipped gradients for both IME methods. It reports negative
raw v_hat fraction, raw/projected second-moment MSE, first-moment MSE and Adam
direction RMSE, averaged over coordinates, steps and ten independent draws.
Moment errors compare bias-corrected readouts against their clean counterparts;
“raw” means unprojected v_hat_raw, not a clamped recurrence state.
Coordinate blocks keep memory bounded; replay keys are paired between methods.
Clean gradients are internal diagnostics, not private releases.

Within each training seed all methods share initialization, batch schedule,
clipping and channel-1 latent keys. Channel 2 has a distinct folded key. Training
records schedule/checkpoint hashes, schedule arrays, calibration, initial test
metrics, epoch metrics and summaries. Aggregate requires all 50 jobs and reports
final/best accuracy/loss and normalized accuracy AUC, mean, sample std, SE,
Student-t 95% CI, all ten method pairs, and each paired seed difference.
Accuracy AUC includes the initial evaluation and integrates effective epochs.

## Execution on GPUs 1–3

The user's GPU assignment overrides the attachment's four-GPU schedule. Jobs
are method-major, seed-minor, with job_index % 3 selecting physical GPU 1, 2, 3.
Each worker runs its assigned jobs serially (17, 17, 16 jobs). The launcher first
fits and replays on GPU 1, then starts three workers. It waits for all workers;
any failure returns nonzero, and aggregation/plotting require complete success.
There is no resume, missing-task skipping, download, or automatic retry.

Smoke uses 64 local train/test examples, batch/microbatch 16, one epoch/four
steps, five fitting steps and two replay draws. Run from the repository root:

```bash
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="$PWD/exp13/results_smoke/matplotlib"
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/fit_strategies.py --smoke
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n curve python -B exp13/replay.py --smoke
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n curve python -B exp13/full_training.py --smoke
conda run -n curve pytest exp13/tests
```

Tests cover workload equality, fitted strategy column norms/inverse/objective,
multi-participation sensitivity, calibration, GDP composition, square-after-mean,
linear raw recurrence and projection, standard Adam agreement, three-/four-worker
partitions, and real GPU smoke artifacts (run the smoke commands before these integration checks).
Only smoke was requested; the full ten-seed experiment is not launched during
development. All smoke artifacts and logs are in `results_smoke/`.

Full experiment command (uses GPUs 1, 2, 3):

```bash
bash exp13/launch_full.sh
```
