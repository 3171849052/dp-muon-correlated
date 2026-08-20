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

For the paired 5B comparison, continuous and segmented plans use the larger
of their two full-hybrid sensitivities for one shared conservative calibration.
Consequently all four conditions satisfy the same final privacy bound and the
IID warmup perturbations are bit-identical, not merely generated from matching
standard-normal seeds. The lower-sensitivity mechanism may spend less than the
stated epsilon; it never exceeds it.

The frozen replay comparison is deliberately narrower than linearizing neural
network training. `G_frozen` tests removal of the dynamic AdamW second-moment
nonlinearity only; it makes no claim that model training dynamics are linear.

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
