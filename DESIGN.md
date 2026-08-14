# Design conventions

- `D = C^{-1}` is the banded noising matrix optimized by BandInvMF.
- `C = D^{-1}` is the strategy matrix.
- `noising_coef` denotes coefficients of `C^{-1}`; `strategy_coef` denotes
  coefficients of `C`.
- The streaming generator uses `D = C^{-1} = LTT(d0, ..., d_{p-1})` directly:
  `Z_t` is iid `N(0, tau^2 I)` and
  `E_t = sum_j d_j Z_{t-j}`, with pre-history `Z` equal to zero.  Thus
  `E = D Z` and `Cov(E) = tau^2 D D^T`.
- Its state stores the latent `Z` history in a circular buffer, never past
  correlated outputs `E`.  Here `tau` is the standard deviation of `Z_t`.
  M1 has already completed privacy scaling; M2 applies no sensitivity or noise
  multiplier of its own.
- Per-example clipping is delegated to `jax_privacy.clipping.clipped_grad`.
  A scalar `l2_clip_norm = L` means global L2 clipping over the entire gradient
  PyTree, and the query is `q_t = (1 / B0) * sum_i clipped_grad_i`, where
  `B0 = normalize_by` is fixed and public (never the realized batch size).
  Its add/remove sensitivity is `L / B0`; replace-one sensitivity is
  `2L / B0`. Per-example norms, clip factors, losses, and similar private
  diagnostics are neither returned nor logged.
- M3 produces only this clipped query; it does not add BandInvMF noise. A later
  phase will combine the M2 correlated noise with `q_t`.
- This phase prepares only a non-amplified linear baseline.
- `jax_privacy` provides the underlying mathematics. This repository imports
  it and does not fork or reimplement Toeplitz, BISR, sensitivity, or the
  optimizer.
- The current non-amplified BandInvMF mechanism is treated as one
  full-transcript Gaussian mechanism. It has neither subsampling amplification
  nor per-step composition.
- Opacus is used only for GDP dual conversion between `(epsilon, delta)` and
  `mu`; this project does not use its subsampled DP-SGD accountant APIs.
- `normalize_by` must be a fixed, public constant. Privacy calibration must
  not use an actual random batch size or any other quantity dependent on
  private data.
- Under add/remove adjacency,
  `query_sensitivity = clip_norm / normalize_by`. Under replace-one adjacency,
  `query_sensitivity = 2 * clip_norm / normalize_by`.
- For query sensitivity `s_q` and matrix sensitivity `S(C)`, the final iid
  Gaussian standard deviation is `tau = m * s_q * S(C)`, where `m = 1 / mu`.
- Clipping calculations, participation scheduling, optimizer integration, and
  the training loop remain out of scope for this phase.
