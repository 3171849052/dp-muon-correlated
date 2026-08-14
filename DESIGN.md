# Design conventions

- `D = C^{-1}` is the banded noising matrix optimized by BandInvMF.
- `C = D^{-1}` is the strategy matrix.
- `noising_coef` denotes coefficients of `C^{-1}`; `strategy_coef` denotes
  coefficients of `C`.
- This phase prepares only a non-amplified linear baseline.
- `jax_privacy` provides the underlying mathematics. This repository imports
  it and does not fork or reimplement Toeplitz, BISR, sensitivity, or the
  optimizer.
- The current non-amplified BandInvMF mechanism is treated as one
  full-transcript Gaussian mechanism. It has neither subsampling amplification
  nor per-step composition.
- Opacus is used only for GDP dual conversion between `(epsilon, delta)` and
  `mu`; this project does not use its subsampled DP-SGD accountant APIs.
- For query sensitivity `s_q` and matrix sensitivity `S(C)`, the final iid
  Gaussian standard deviation is `tau = m * s_q * S(C)`, where `m = 1 / mu`.
- Clipping calculations, streaming noise, and the training loop remain out of
  scope for this phase.
