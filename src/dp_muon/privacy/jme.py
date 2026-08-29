"""Privacy calibration helpers for the shadow-second-moment JME trainer.

The post-warmup release is treated as one Gaussian joint mechanism per
segment.  The first and second channels have independent latent Gaussian
streams, but use the same calibrated latent standard deviation.  Composition
is done in GDP space: independent Gaussian mechanisms compose with the square
root of the sum of squared Mahalanobis sensitivities.  This is also a valid
conservative accounting rule for the adaptive segment choices because each
new strategy is selected from an already-private shadow output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import BandInvMFStrategy

Adjacency = Literal["add_remove", "replace_one"]


@dataclass(frozen=True)
class ShadowJMEPrivacyCalibration:
  """One global GDP calibration for warmup plus all JME segments."""

  epsilon: float
  delta: float
  adjacency: Adjacency
  clip_norm: float
  normalize_by: float
  query_sensitivity: float
  warmup_sensitivity_squared: float
  segment_sensitivity_squared: tuple[float, ...]
  total_sensitivity_squared: float
  mu: float
  noise_multiplier: float
  iid_noise_std: float
  accounting: str = "gdp-composition"

  @property
  def noise_std(self) -> float:
    """Alias used by the mechanism equations."""
    return self.iid_noise_std


def _positive(name: str, value: float) -> float:
  value = float(value)
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f"{name} must be finite and positive")
  return value


def _strategy_matrix(strategy: BandInvMFStrategy) -> np.ndarray:
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  coef = np.asarray(strategy.strategy_coef)
  if coef.ndim != 1 or coef.shape != (strategy.horizon,):
    raise ValueError("strategy.strategy_coef must have shape (strategy.horizon,)")
  rows, columns = np.indices((strategy.horizon, strategy.horizon))
  return np.where(rows >= columns, coef[rows - columns], 0.0)


def bandinv_operator_norm_1_to_2_squared(strategy: BandInvMFStrategy) -> float:
  """Returns ``||C||_{1->2}^2`` for the fitted causal matrix ``C``."""
  matrix = _strategy_matrix(strategy)
  result = float(np.max(np.sum(matrix * matrix, axis=0)))
  if not math.isfinite(result) or result <= 0:
    raise ValueError("strategy operator norm must be finite and positive")
  return result


def jme_gamma_and_joint_sensitivity(
    first_strategy: BandInvMFStrategy,
    second_strategy: BandInvMFStrategy,
    *,
    zeta: float,
    adjacency: Adjacency = "add_remove",
) -> tuple[float, float]:
  """Returns the prescribed JME ``gamma`` and joint sensitivity ``s``.

  ``zeta`` is the already-normalized query radius, namely
  ``clip_norm / normalize_by``.  No additional clip or batch-size factor is
  applied here.  The supplied JME construction uses

  ``gamma = ||C_m||^2 / (2 zeta^2 ||C_v||^2)`` and
  ``s = 2 zeta ||C_m||``.
  """
  zeta = _positive("zeta", zeta)
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("adjacency must be 'add_remove' or 'replace_one'")
  first_norm = bandinv_operator_norm_1_to_2_squared(first_strategy)
  second_norm = bandinv_operator_norm_1_to_2_squared(second_strategy)
  gamma = first_norm / (2.0 * zeta * zeta * second_norm)
  adjacency_factor = 1.0 if adjacency == "add_remove" else 2.0
  sensitivity = 2.0 * adjacency_factor * zeta * math.sqrt(first_norm)
  return float(gamma), float(sensitivity)


def calibrate_shadow_jme(
    *,
    epsilon: float,
    delta: float,
    clip_norm: float,
    normalize_by: float,
    adjacency: Adjacency,
    warmup_strategy: BandInvMFStrategy,
    first_strategies: Sequence[BandInvMFStrategy],
    second_strategies: Sequence[BandInvMFStrategy],
) -> ShadowJMEPrivacyCalibration:
  """Calibrates one latent Gaussian scale for the whole hybrid transcript."""
  epsilon = _positive("epsilon", epsilon)
  delta = float(delta)
  if not math.isfinite(delta) or not 0.0 < delta < 1.0:
    raise ValueError("delta must be finite and in (0, 1)")
  clip_norm = _positive("clip_norm", clip_norm)
  normalize_by = _positive("normalize_by", normalize_by)
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("adjacency must be 'add_remove' or 'replace_one'")
  if len(first_strategies) != len(second_strategies):
    raise ValueError("first_strategies and second_strategies must have equal length")
  if not first_strategies:
    raise ValueError("at least one post-warmup segment is required")

  zeta = clip_norm / normalize_by
  adjacency_factor = 1.0 if adjacency == "add_remove" else 2.0
  warmup_norm = bandinv_operator_norm_1_to_2_squared(warmup_strategy)
  warmup_sensitivity_squared = (adjacency_factor * zeta) ** 2 * warmup_norm
  segment_sensitivities = tuple(
      jme_gamma_and_joint_sensitivity(
          first, second, zeta=zeta, adjacency=adjacency
      )[1] ** 2
      for first, second in zip(first_strategies, second_strategies, strict=True)
  )
  total = warmup_sensitivity_squared + sum(segment_sensitivities)
  if not math.isfinite(total) or total <= 0:
    raise ValueError("global JME sensitivity must be finite and positive")

  # Reuse the project's GDP target inversion.  With sigma = m*sqrt(total),
  # the composed GDP parameter is exactly 1/m at the full transcript.
  from .nonamplified import calibrate_gdp_noise_multiplier

  noise_multiplier = calibrate_gdp_noise_multiplier(epsilon, delta)
  mu = 1.0 / noise_multiplier
  return ShadowJMEPrivacyCalibration(
      epsilon=epsilon,
      delta=delta,
      adjacency=adjacency,
      clip_norm=clip_norm,
      normalize_by=normalize_by,
      query_sensitivity=adjacency_factor * zeta,
      warmup_sensitivity_squared=float(warmup_sensitivity_squared),
      segment_sensitivity_squared=tuple(float(value) for value in segment_sensitivities),
      total_sensitivity_squared=float(total),
      mu=mu,
      noise_multiplier=noise_multiplier,
      iid_noise_std=noise_multiplier * math.sqrt(total),
  )


def _epsilon_from_sensitivity(
    sensitivity_squared: float, calibration: ShadowJMEPrivacyCalibration
) -> float:
  sensitivity_squared = _positive("sensitivity_squared", sensitivity_squared)
  sigma = _positive("calibration.iid_noise_std", calibration.iid_noise_std)
  value = float(gdp.eps_from_mu(
      mu=math.sqrt(sensitivity_squared) / sigma, delta=calibration.delta
  ))
  if not math.isfinite(value) or value < 0:
    raise RuntimeError("Opacus GDP conversion returned an invalid epsilon")
  return value


def epsilon_spent_for_shadow_jme_prefix(
    *,
    prefix_steps: int,
    warmup_steps: int,
    segment_lengths: Sequence[int],
    calibration: ShadowJMEPrivacyCalibration,
) -> float:
  """Returns a conservative GDP bound for a released training prefix.

  An unfinished segment is charged its complete joint sensitivity.  This is
  intentionally conservative and avoids claiming a prefix formula for an
  adaptive strategy whose final shadow output has not yet been released.
  """
  if not isinstance(prefix_steps, Integral) or isinstance(prefix_steps, bool) or prefix_steps < 1:
    raise ValueError("prefix_steps must be a positive integer")
  if not isinstance(warmup_steps, Integral) or isinstance(warmup_steps, bool) or warmup_steps < 1:
    raise ValueError("warmup_steps must be a positive integer")
  lengths = tuple(int(value) for value in segment_lengths)
  if not lengths or any(value < 1 for value in lengths):
    raise ValueError("segment_lengths must contain positive integers")
  horizon = int(warmup_steps) + sum(lengths)
  if prefix_steps > horizon:
    raise ValueError("prefix_steps exceeds the JME horizon")
  if len(lengths) != len(calibration.segment_sensitivity_squared):
    raise ValueError("calibration does not match segment_lengths")
  if prefix_steps <= warmup_steps:
    sensitivity_squared = calibration.warmup_sensitivity_squared
  else:
    consumed = int(prefix_steps) - int(warmup_steps)
    sensitivity_squared = calibration.warmup_sensitivity_squared
    for length, segment_sensitivity in zip(
        lengths, calibration.segment_sensitivity_squared, strict=True
    ):
      if consumed <= 0:
        break
      sensitivity_squared += segment_sensitivity
      consumed -= length
  if prefix_steps == horizon:
    sensitivity_squared = calibration.total_sensitivity_squared
  return _epsilon_from_sensitivity(sensitivity_squared, calibration)


__all__ = [
    "ShadowJMEPrivacyCalibration",
    "bandinv_operator_norm_1_to_2_squared",
    "calibrate_shadow_jme",
    "epsilon_spent_for_shadow_jme_prefix",
    "jme_gamma_and_joint_sensitivity",
]
