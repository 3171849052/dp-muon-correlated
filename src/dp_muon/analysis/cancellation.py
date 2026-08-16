"""Prefix-cancellation calculations for frozen Muon trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dp_muon.optim import nesterov_kernel_coef


@dataclass(frozen=True)
class CausalNoiseOperator:
  """Dense finite-horizon representation of ``H_beta^Nes C^-1``."""

  correlated: np.ndarray
  nesterov: np.ndarray
  total: np.ndarray


def _lower_toeplitz(coef: np.ndarray, horizon: int) -> np.ndarray:
  coef = np.asarray(coef, dtype=np.float64)
  if coef.ndim != 1 or not 1 <= len(coef) <= horizon:
    raise ValueError("causal coefficients must be one-dimensional with length in [1, horizon]")
  offsets = np.arange(horizon)[:, None] - np.arange(horizon)[None, :]
  padded = np.pad(coef, (0, horizon - len(coef)))
  return np.where(offsets >= 0, padded[np.maximum(offsets, 0)], 0.0)


def make_causal_noise_operator(
    noising_coef: np.ndarray, horizon: int, momentum: float
) -> CausalNoiseOperator:
  """Builds the exact finite transcript map ``H_beta^Nes C^-1``.

  ``noising_coef`` is BandInvMF's lower-triangular Toeplitz ``C^-1``
  coefficient vector.  The second causal factor is deliberately applied after
  it, matching correlated DP-Muon where noise enters the gradient before
  classic momentum/Nesterov.
  """
  if horizon < 1:
    raise ValueError("horizon must be positive")
  if not 0 <= momentum < 1:
    raise ValueError("momentum must be in [0, 1)")
  correlated = _lower_toeplitz(noising_coef, horizon)
  nesterov = _lower_toeplitz(np.asarray(nesterov_kernel_coef(horizon, momentum)), horizon)
  return CausalNoiseOperator(correlated=correlated, nesterov=nesterov, total=nesterov @ correlated)


def rescale_noise_to_median_ratio(
    clean: np.ndarray, noise: np.ndarray, target_median_r: float
) -> tuple[np.ndarray, np.ndarray]:
  """Globally rescales every sampled noise transcript to its target median r.

  One scalar is used per complete trajectory; no individual time step is
  rescaled, so the temporal covariance direction remains untouched.
  """
  clean = np.asarray(clean, dtype=np.float64)
  noise = np.asarray(noise, dtype=np.float64)
  if clean.ndim != 3 or noise.ndim != 4 or noise.shape[1:] != clean.shape:
    raise ValueError("clean must be (T,m,n) and noise must be (S,T,m,n)")
  if target_median_r <= 0 or not np.isfinite(target_median_r):
    raise ValueError("target_median_r must be finite and positive")
  clean_norm = np.linalg.norm(clean, axis=(1, 2))
  if np.any(clean_norm == 0):
    raise ValueError("cannot define relative noise for a zero-norm clean update")
  ratios = np.linalg.norm(noise, axis=(2, 3)) / clean_norm[None, :]
  medians = np.median(ratios, axis=1)
  if np.any(medians == 0) or not np.all(np.isfinite(medians)):
    raise ValueError("sampled noise has an invalid median relative norm")
  scalars = target_median_r / medians
  return noise * scalars[:, None, None, None], scalars


def cancellation_statistics(
    deltas: np.ndarray, learning_rates: np.ndarray
) -> dict[str, np.ndarray | float]:
  """Returns Monte-Carlo ``J_k,D_k,R_k`` and the ratio-of-sums ``R``.

  Expectations are estimated before division.  In particular, ``R`` is not
  the mean of sample-wise or prefix-wise ratios.
  """
  deltas = np.asarray(deltas, dtype=np.float64)
  learning_rates = np.asarray(learning_rates, dtype=np.float64)
  if deltas.ndim != 4:
    raise ValueError("deltas must have shape (samples, T, m, n)")
  if learning_rates.shape != (deltas.shape[1],):
    raise ValueError("learning_rates must have shape (T,)")
  x = deltas * learning_rates[None, :, None, None]
  prefix = np.cumsum(x, axis=1)
  j = np.mean(np.sum(prefix * prefix, axis=(2, 3)), axis=0)
  d = np.cumsum(np.mean(np.sum(x * x, axis=(2, 3)), axis=0))
  if np.any(d <= 0):
    raise ValueError("all prefix denominators must be positive")
  r = j / d
  return {"J": j, "D": d, "R": r, "aggregate_R": float(np.sum(j) / np.sum(d))}


__all__ = [
    "CausalNoiseOperator",
    "cancellation_statistics",
    "make_causal_noise_operator",
    "rescale_noise_to_median_ratio",
]
