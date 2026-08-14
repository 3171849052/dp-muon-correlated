"""GDP calibration for one non-amplified BandInvMF Gaussian mechanism."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from opacus.accountants.analysis import gdp

Adjacency = Literal["add_remove", "replace_one"]


@dataclass(frozen=True)
class PrivacyCalibration:
  """Parameters of a full-transcript, non-amplified Gaussian mechanism."""

  epsilon: float
  delta: float
  adjacency: Adjacency
  clip_norm: float
  normalize_by: float
  query_sensitivity: float
  matrix_sensitivity: float
  total_sensitivity: float
  mu: float
  noise_multiplier: float
  iid_noise_std: float


def _positive_finite(name: str, value: float) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be a finite positive number") from error
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f"{name} must be a finite positive number")
  return value


def _validate_privacy_parameters(epsilon: float, delta: float) -> tuple[float, float]:
  epsilon = _positive_finite("epsilon", epsilon)
  try:
    delta = float(delta)
  except (TypeError, ValueError) as error:
    raise ValueError("delta must be finite and in (0, 1)") from error
  if not math.isfinite(delta) or not 0 < delta < 1:
    raise ValueError("delta must be finite and in (0, 1)")
  return epsilon, delta


def compute_query_sensitivity(
    clip_norm: float,
    normalize_by: float,
    adjacency: Adjacency,
) -> float:
  """Returns sensitivity of one clipped, normalized participation query."""
  clip_norm = _positive_finite("clip_norm", clip_norm)
  normalize_by = _positive_finite("normalize_by", normalize_by)
  if adjacency == "add_remove":
    factor = 1.0
  elif adjacency == "replace_one":
    factor = 2.0
  else:
    raise ValueError("adjacency must be 'add_remove' or 'replace_one'")
  return factor * clip_norm / normalize_by


def _solve_gdp_mu(epsilon: float, delta: float) -> float:
  """Solves Opacus' ``delta_eps_mu(epsilon, mu) == delta`` for ``mu``."""
  epsilon, delta = _validate_privacy_parameters(epsilon, delta)

  def opacus_delta(mu: float) -> float:
    value = float(gdp.delta_eps_mu(eps=epsilon, mu=mu))
    if not math.isfinite(value):
      raise RuntimeError("Opacus GDP conversion returned a non-finite delta")
    return value

  # Opacus 1.6.0 evaluates ``eps / mu`` internally, so it cannot be invoked
  # at exactly zero. The smallest positive float is the numerical zero limit.
  low, high = math.nextafter(0.0, 1.0), 1.0
  if opacus_delta(low) > delta:
    raise RuntimeError("could not bracket the GDP mu solution at mu=0")
  while opacus_delta(high) < delta:
    high *= 2.0
    if not math.isfinite(high):
      raise RuntimeError("could not bracket the GDP mu solution")

  # Bisection is sufficient for this scalar monotone inversion and invokes
  # Opacus' installed GDP conversion at every evaluation.
  for _ in range(80):
    middle = (low + high) / 2.0
    if opacus_delta(middle) < delta:
      low = middle
    else:
      high = middle
  return (low + high) / 2.0


def calibrate_gdp_noise_multiplier(epsilon: float, delta: float) -> float:
  """Returns ``m = 1 / mu`` for an Opacus GDP ``(epsilon, delta)`` target."""
  mu = _solve_gdp_mu(epsilon, delta)
  return 1.0 / mu


def calibrate_nonamplified_bandinv(
    *,
    epsilon: float,
    delta: float,
    clip_norm: float,
    normalize_by: float,
    adjacency: Adjacency,
    sensitivity_squared: float,
) -> PrivacyCalibration:
  """Calibrates iid noise for a single full-transcript BandInvMF mechanism.

  This deliberately has no sampling amplification or per-step composition:
  ``tau = m * s_q * sqrt(sensitivity_squared)``.
  """
  epsilon, delta = _validate_privacy_parameters(epsilon, delta)
  query_sensitivity = compute_query_sensitivity(
      clip_norm, normalize_by, adjacency
  )
  sensitivity_squared = _positive_finite(
      "sensitivity_squared", sensitivity_squared
  )
  matrix_sensitivity = math.sqrt(sensitivity_squared)
  total_sensitivity = query_sensitivity * matrix_sensitivity
  noise_multiplier = calibrate_gdp_noise_multiplier(epsilon, delta)
  mu = 1.0 / noise_multiplier
  return PrivacyCalibration(
      epsilon=epsilon,
      delta=delta,
      adjacency=adjacency,
      clip_norm=float(clip_norm),
      normalize_by=float(normalize_by),
      query_sensitivity=query_sensitivity,
      matrix_sensitivity=matrix_sensitivity,
      total_sensitivity=total_sensitivity,
      mu=mu,
      noise_multiplier=noise_multiplier,
      iid_noise_std=noise_multiplier * total_sensitivity,
  )
