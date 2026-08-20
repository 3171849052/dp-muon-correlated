# Experiment 5: IID warmup to Frozen-p BandInvMF

Experiment 5 switches a real Optax DP-AdamW state after step `tau`. The
uncorrected second moment is read from Optax and frozen as
`p_star = 1 / (sqrt(v_tau / (1-beta2**count)) + eps)`. Parameters, first
moment, and the global Adam count continue without reset. Segment boundaries
reset only the BandInvMF latent-noise filter.

The Phase-II workload is not an optimizer-only approximation. It is the full
linear map

```
A = A_time(tau, beta1, lr, weight_decay) tensor Diag(p_star)
```

where row zero uses bias correction `1-beta1**(tau+1)`. Temporal strategy
fitting uses separability: the shared temporal optimum can be fitted from
`A_time`, while objective checks, exact workload replay, and gap diagnostics
apply `p_star` explicitly on every coordinate.

Privacy uses `D=C^-1`. Each run constructs
`D_hybrid=blockdiag(I_tau,D_1,D_2,...)` (or one remaining continuous block),
inverts it blockwise to obtain `C_hybrid`, and maximizes `||C_hybrid x_pi||^2`
under the original full-horizon `min_sep` and total participation cap. Thus
warmup and phase boundaries do not create new participation contracts. One
GDP calibration targets the final `(epsilon=3, delta=1e-5, add_remove)`
transcript; there is no amplification or per-block privacy budget.

For the paired 5B comparison, continuous and segmented plans each use their
own exact full-hybrid sensitivity and are independently calibrated to the same
final `(epsilon=3, delta=1e-5)` target. Dynamic and frozen conditions within a
mechanism use the same plan, calibration, latent Gaussian base draws, and
actual noise transcript. Continuous and segmented mechanisms use the same
standard-normal seed/step-key scheme, but their calibrated scales and noising
matrices can differ, including during IID warmup. There is no shared
worst-case calibration.

The 5C replay is an online full-model PyTree computation, not a selected-leaf
approximation. Starting from the real switch parameters, moments, count, and
`p_star`, it advances dynamic clean/noisy and frozen clean/noisy shadow states
using each observed clipped gradient as an exogenous input. In parallel it
advances the exact frozen-p linear perturbation recurrence. At every Phase-II
step it sums squared norms over every parameter leaf into scalar numerators and
a denominator, so the full gradient trajectory is never retained. The
comparison isolates dynamic AdamW second-moment/preconditioner nonlinearity;
it does not claim that neural-network gradient dynamics are linear.

Smoke (small differentiable optimization, real Optax state, BandInvMF fit,
hybrid calibration, paired replay):

```bash
conda activate curve
python exp5/run.py --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp5/results --seeds 0 1 2 --smoke
```

Formal command:

```bash
conda activate curve
python exp5/run.py --config config/cifar10_bandinv_dpadamw_naive.yaml \
  --output-dir exp5/results --seeds 0 1 2
```
