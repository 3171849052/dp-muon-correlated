# Design conventions

- `D = C^{-1}` is the banded noising matrix optimized by BandInvMF.
- `C = D^{-1}` is the strategy matrix.
- `noising_coef` denotes coefficients of `C^{-1}`; `strategy_coef` denotes
  coefficients of `C`.
- This phase prepares only a non-amplified linear baseline.
- `jax_privacy` provides the underlying mathematics. This repository imports
  it and does not fork or reimplement Toeplitz, BISR, sensitivity, or the
  optimizer.
- Privacy calibration, clipping, streaming noise, and the training loop are
  intentionally deferred to the next phase.
