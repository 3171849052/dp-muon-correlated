"""Exact privacy accounting for an IID-prefix/continuous-BandInvMF hybrid."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import toeplitz
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import BandInvMFStrategy

from .nonamplified import PrivacyCalibration


def _phase_sensitivity_squared(
    *,
    tau: int,
    phase_horizon: int,
    noising_coef: Any,
    min_sep: int,
    max_participations: int,
) -> float:
  """Evaluates the exact global-contract DP over warmup count choices."""
  if phase_horizon < 1:
    return 0.0
  max_warmup = min(max_participations, 1 + (tau - 1) // min_sep)
  sensitivity = 0.0
  cpu = jax.devices("cpu")[0]
  for warmup_count in range(max_warmup + 1):
    phase_start = max(0, warmup_count * min_sep - tau)
    suffix_length = phase_horizon - phase_start
    remaining = max_participations - warmup_count
    if suffix_length <= 0 or remaining <= 0:
      phase_sensitivity = 0.0
    else:
      with jax.default_device(cpu):
        phase_sensitivity = float(
            toeplitz.compute_banded_inverse_sensitivity_squared(
                n=suffix_length,
                noising_coef=jnp.asarray(noising_coef),
                min_sep=min_sep,
                max_participations=remaining,
            )
        )
    sensitivity = max(sensitivity, float(warmup_count) + phase_sensitivity)
  return sensitivity


def continuous_hybrid_sensitivity_squared(
    tau: int,
    phase_strategy: BandInvMFStrategy,
    *,
    min_sep: int,
    max_participations: int,
) -> float:
  """Returns exact sensitivity for ``blockdiag(I_tau, D_phase)``.

  This is the continuous fast path: it only evaluates one Toeplitz
  sensitivity per feasible warmup participation count and never enumerates
  Phase-II participation tuples.
  """
  if not isinstance(tau, Integral) or tau < 1:
    raise ValueError("tau must be a positive integer")
  if not isinstance(min_sep, Integral) or min_sep < 1:
    raise ValueError("min_sep must be a positive integer")
  if not isinstance(max_participations, Integral) or max_participations < 1:
    raise ValueError("max_participations must be a positive integer")
  if not isinstance(phase_strategy, BandInvMFStrategy):
    raise TypeError("phase_strategy must be a BandInvMFStrategy")
  if phase_strategy.horizon < 1:
    raise ValueError("phase strategy horizon must be positive")
  return _phase_sensitivity_squared(
      tau=int(tau),
      phase_horizon=int(phase_strategy.horizon),
      noising_coef=phase_strategy.noising_coef,
      min_sep=int(min_sep),
      max_participations=int(max_participations),
  )


def continuous_hybrid_prefix_sensitivity_squared(
    prefix_steps: int,
    *,
    tau: int,
    phase_strategy: BandInvMFStrategy,
    min_sep: int,
    max_participations: int,
) -> float:
  """Returns exact sensitivity for a released hybrid transcript prefix."""
  if not isinstance(prefix_steps, Integral) or prefix_steps < 1:
    raise ValueError("prefix_steps must be a positive integer")
  if prefix_steps <= tau:
    return float(min(max_participations, 1 + (prefix_steps - 1) // min_sep))
  phase_horizon = int(prefix_steps) - int(tau)
  if phase_horizon > int(phase_strategy.horizon):
    raise ValueError("prefix_steps exceeds the hybrid horizon")
  return _phase_sensitivity_squared(
      tau=int(tau),
      phase_horizon=phase_horizon,
      noising_coef=phase_strategy.noising_coef,
      min_sep=int(min_sep),
      max_participations=int(max_participations),
  )


def epsilon_spent_for_continuous_hybrid_prefix(
    *,
    prefix_steps: int,
    tau: int,
    phase_strategy: BandInvMFStrategy,
    min_sep: int,
    max_participations: int,
    calibration: PrivacyCalibration,
    full_sensitivity_squared: float | None = None,
) -> float:
  """Converts a hybrid prefix sensitivity using the one final noise scale."""
  if not isinstance(calibration, PrivacyCalibration):
    raise TypeError("calibration must be a PrivacyCalibration")
  full_prefix = int(tau) + int(phase_strategy.horizon)
  if prefix_steps == full_prefix and full_sensitivity_squared is not None:
    sensitivity_squared = float(full_sensitivity_squared)
  else:
    sensitivity_squared = continuous_hybrid_prefix_sensitivity_squared(
        prefix_steps,
        tau=tau,
        phase_strategy=phase_strategy,
        min_sep=min_sep,
        max_participations=max_participations,
    )
  if not math.isfinite(sensitivity_squared) or sensitivity_squared <= 0:
    raise ValueError("hybrid sensitivity_squared must be positive and finite")
  noise_std = float(calibration.iid_noise_std)
  if not math.isfinite(noise_std) or noise_std <= 0:
    raise ValueError("calibration.iid_noise_std must be positive and finite")
  mu = calibration.query_sensitivity * math.sqrt(sensitivity_squared) / noise_std
  epsilon = float(gdp.eps_from_mu(mu=mu, delta=calibration.delta))
  if not math.isfinite(epsilon) or epsilon < 0:
    raise RuntimeError("Opacus GDP conversion returned an invalid epsilon")
  return epsilon


__all__ = [
    "continuous_hybrid_prefix_sensitivity_squared",
    "continuous_hybrid_sensitivity_squared",
    "epsilon_spent_for_continuous_hybrid_prefix",
]
