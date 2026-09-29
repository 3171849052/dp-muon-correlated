# Exp13: Adam, BandMF, and separate-matrix IME

All Exp13 code and generated artifacts live in this directory. The experiment
uses the repository's existing CIFAR-10 data, pretrained ViT-Tiny checkpoint,
global per-example clipping, fixed-cycle schedule, and non-amplified GDP
calibration. Data loading uses local files with `download=False`.

The formal configuration uses five epochs, batch size 512, microbatch size 256,
clip norm 1, Adam beta1=.9, beta2=.999, epsilon 1e-8, learning rate .005,
privacy epsilon 3, delta 1e-5, bandwidth 4, and seeds `[0, 1, 2]`. For 50,000
training examples the contract is horizon 488, minimum separation 97, and
maximum participations 5.

## Methods

- `nonprivate_adam`: Adam on the clipped batch mean.
- `iid_adam`: Adam on the clipped batch mean with IID Gaussian noise.
- `bandmf_single_m`: Adam on the clipped batch mean with the fitted first-moment BandMF strategy.
- `iid_ime`: independent IID Gaussian channels for the batch mean and its square.
- `bandmf_ime_sep`: separate fitted `C_m` and `C_v` strategies for those channels.

IME squares the clipped batch mean, not individual gradients. Its raw second
state is the linear recurrence
`v_raw = beta2 * v_raw + (1-beta2) * private_q`. Bias correction is applied
after that recurrence, and IME uses `abs(v_hat_raw)` only in the optimizer
denominator. This readout never enters the raw state update. Non-IME methods
continue to use squared private gradients for their second moment.

Early ReLU readout smoke diagnostics had many denominators equal to `adam_eps`
and unstable losses; formal Exp13 now uses abs readout.

## Privacy and BandMF workload

The fixed-denominator zero-out/add-remove sensitivities are
`a1 = clip_norm / batch_size` and
`a2 = (2*batch_size - 1) * clip_norm**2 / batch_size**2`. Single-channel
methods use `mu1=mu`; IME uses `mu1=mu2=mu/sqrt(2)`. Channel noise is
`sigma_i = a_i * sqrt(k) / mu_i`. The abs readout is DP post-processing and
does not change privacy accounting.

`workloads.py` defines the uncorrected EMA workload
`H_beta[t,s] = (1-beta) * beta**(t-s)` for `s <= t`. Strategy fitting excludes
bias correction, learning rate, prefix sum, weight decay, and Adam denominator.
Formal fitting uses 1000 optimizer steps; smoke fitting uses five. The four
cross-evaluation cells `J_m(C_m)`, `J_m(C_v)`, `J_v(C_m)`, and `J_v(C_v)`, plus
`1 - J_v(C_v)/J_v(C_m)`, are saved under `results/` and `results_smoke/`.

Frozen replay saves raw uncorrected `first_state_mse` and `second_state_mse`,
along with separately named bias-corrected `first_hat_mse`,
`second_hat_raw_mse`, and `second_hat_abs_mse`. Its
`workload_validation.{csv,json}` compares the theoretical per-coordinate
state MSE `sigma**2 * mean(banded.per_query_error(strategy, A=ema(beta)))`
against replay across draws, steps, and coordinates for all four IME channels.
IID uses an identity noising strategy in the same workload calculation.

`replay/metrics.{csv,json}` records negative raw second estimates, denominator
minimum and quantiles, explicitly named `1e_minus_*` denominator fractions,
and absolute direction magnitude and RMSE. `replay/per_step_diagnostics` has
step-level negative fractions, denominator minimum and p001, plus direction p99
and maximum. Quantiles pool all replay draws and coordinates.

## Jobs and GPUs

The formal run contains 5 methods × 3 seeds = 15 jobs. Method-major, seed-minor
ordering assigns five jobs each to GPUs 1, 2, and 3. `GPUS`, the config GPU IDs,
worker behavior, and launcher bindings remain `(1, 2, 3)`. Aggregation requires
all configured method/seed summaries and reports mean, sample standard
deviation, standard error, and Student-t 95% confidence intervals. Paired
comparisons use the same three seeds for:

- `bandmf_single_m - iid_adam`
- `bandmf_ime_sep - iid_ime`
- `bandmf_ime_sep - bandmf_single_m`

## Smoke and full experiment

Smoke uses 64 local train/test examples, batch and microbatch size 16, one epoch
and four steps, five fitting steps, and two replay draws. From the repository
root, run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/fit_strategies.py --smoke
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n curve python -B exp13/replay.py --smoke
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n curve python -B exp13/full_training.py --smoke
conda run -n curve pytest exp13/tests
```

The full experiment command uses GPUs 1, 2, and 3:

```bash
bash exp13/launch_full.sh
```
