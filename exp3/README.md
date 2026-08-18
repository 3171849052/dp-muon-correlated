# Experiment 3: online shadow diagnostics

This experiment runs the actual CIFAR-10 BandInvMF DP-AdamW trainer. Each
logical batch computes one clipped clean gradient `g_t` and one correlated DP
noise `n_t`; the real AdamW optimizer receives `g_t+n_t`. On that same step a
shadow AdamW recurrence (without a second model or extra forward/backward)
computes `Delta q_t`, while a first-moment-only recurrence computes `r_t^lin`.
Only scalar diagnostics are written to `diagnostics_*.csv`; vector moments are
part of the checkpointable online state and never exported.

The aggregate ratios are `R=sum_t J_t/sum_t D_t`, never a mean of prefix
ratios. `Delta_R_linear = R_M^linear-R_N^linear` and
`Delta_R_AdamW = R_M^AdamW-R_N^AdamW`; the interaction is
`Gamma_R = Delta_R_AdamW-Delta_R_linear`. A negative linear delta means
m-aware cancellation is better in the linear reference. Positive Gamma means
AdamW normalization weakens that advantage; near zero preserves it; negative
enhances it. No direction is assumed in advance.

Run with `conda activate curve` and:

```bash
python exp3/run.py --config config/cifar10_bandinv_dpadamw_naive.yaml --output-dir exp3/results --seeds 0
```
