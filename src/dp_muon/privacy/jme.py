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
  """One conservative aggregate-square GDP calibration.

  ``query_sensitivity`` is the record-level sensitivity ``Delta_1`` of the
  clean aggregate ``x``.  The second channel uses the conservative
  ``Delta_2`` bound for ``x * x``; ``segment_sensitivity_squared`` stores the
  resulting joint bounds after the two BandInvMF operators.  These fields are
  deliberately named as bounds rather than exact sensitivities of an
  unbounded optimizer trajectory.
  """

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
  accounting: str = "conservative-aggregate-square-gdp-composition"
  aggregate_delta1: float = 0.0
  aggregate_delta2: float = 0.0

  @property
  def noise_std(self) -> float:
    """Alias used by the mechanism equations."""
    return self.iid_noise_std

  @property
  def calibrated_segment_sensitivity_squared(self) -> tuple[float, ...]:
    """Explicit alias for the per-segment conservative calibration bounds."""
    return self.segment_sensitivity_squared


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


def aggregate_square_sensitivities(
    *, clip_norm: float, normalize_by: float, adjacency: Adjacency
) -> tuple[float, float]:
  """Returns conservative record-level bounds for ``x`` and ``x*x``.

  For ``x = sum_i clip(g_i, C) / B``, every released aggregate is in the
  radius-``C`` ball and an add/remove neighboring pair differs by at most
  ``C/B``.  The squared aggregate therefore has the conservative bound
  ``2*C*C/B`` from ``||x*x - x'*x'||_2 <= (||x||_2+||x'||_2)||x-x'||_2``.
  ``replace_one`` follows the repository's existing factor-two adjacency
  convention for both bounds.
  """
  clip_norm = _positive("clip_norm", clip_norm)
  normalize_by = _positive("normalize_by", normalize_by)
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("adjacency must be 'add_remove' or 'replace_one'")
  adjacency_factor = 1.0 if adjacency == "add_remove" else 2.0
  delta1 = adjacency_factor * clip_norm / normalize_by
  delta2 = adjacency_factor * 2.0 * clip_norm * clip_norm / normalize_by
  return float(delta1), float(delta2)


def jme_gamma_and_joint_sensitivity(
    first_strategy: BandInvMFStrategy,
    second_strategy: BandInvMFStrategy,
    *,
    clip_norm: float,
    normalize_by: float,
    adjacency: Adjacency = "add_remove",
) -> tuple[float, float]:
  """Returns ``gamma`` and the conservative joint sensitivity bound.

  The first query is ``x`` and the second is ``x*x``.  With
  ``Delta_1 = C/B`` and ``Delta_2 = 2*C*C/B`` (up to the adjacency factor),
  the two independent channels are balanced by

  ``gamma = Delta_1**2 * ||C_m||_{1->2}**2 /
            (Delta_2**2 * ||C_v||_{1->2}**2)``.

  Hence the joint conservative bound is
  ``s_joint**2 = 2 * Delta_1**2 * ||C_m||_{1->2}**2``.  This is an
  aggregate-square bound; it is not the exact sensitivity of a private-first
  square or a claim that the nonlinear optimizer trajectory is free.
  """
  delta1, delta2 = aggregate_square_sensitivities(
      clip_norm=clip_norm, normalize_by=normalize_by, adjacency=adjacency
  )
  first_norm = bandinv_operator_norm_1_to_2_squared(first_strategy)
  second_norm = bandinv_operator_norm_1_to_2_squared(second_strategy)
  gamma = (delta1 * delta1 * first_norm) / (delta2 * delta2 * second_norm)
  sensitivity = math.sqrt(2.0 * delta1 * delta1 * first_norm)
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
    segment_sensitivity_upper_bounds: Sequence[float] | None = None,
) -> ShadowJMEPrivacyCalibration:
  """Calibrates one latent Gaussian scale for the whole hybrid transcript.

  ``segment_sensitivity_upper_bounds`` can reserve an explicit operational
  envelope above the initial surrogate fits.  Every bound is checked against
  the exact conservative aggregate-square formula for that fitted pair, and
  host-side refits must pass the same bound before being installed.
  """
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

  delta1, delta2 = aggregate_square_sensitivities(
      clip_norm=clip_norm, normalize_by=normalize_by, adjacency=adjacency
  )
  warmup_norm = bandinv_operator_norm_1_to_2_squared(warmup_strategy)
  warmup_sensitivity_squared = delta1**2 * warmup_norm
  segment_sensitivities = tuple(
      jme_gamma_and_joint_sensitivity(
          first,
          second,
          clip_norm=clip_norm,
          normalize_by=normalize_by,
          adjacency=adjacency,
      )[1] ** 2
      for first, second in zip(first_strategies, second_strategies, strict=True)
  )
  if segment_sensitivity_upper_bounds is not None:
    supplied_bounds = tuple(float(value) for value in segment_sensitivity_upper_bounds)
    if len(supplied_bounds) != len(segment_sensitivities):
      raise ValueError(
          "segment_sensitivity_upper_bounds must match the number of segments"
      )
    for index, (bound, fitted) in enumerate(
        zip(supplied_bounds, segment_sensitivities, strict=True)
    ):
      if not math.isfinite(bound) or bound <= 0 or bound < fitted:
        raise ValueError(
            "segment_sensitivity_upper_bounds must be finite, positive, "
            f"and no smaller than fitted bound at segment {index}"
        )
    segment_sensitivities = supplied_bounds
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
      query_sensitivity=delta1,
      warmup_sensitivity_squared=float(warmup_sensitivity_squared),
      segment_sensitivity_squared=tuple(float(value) for value in segment_sensitivities),
      total_sensitivity_squared=float(total),
      mu=mu,
      noise_multiplier=noise_multiplier,
      iid_noise_std=noise_multiplier * math.sqrt(total),
      accounting="conservative-aggregate-square-gdp-composition",
      aggregate_delta1=delta1,
      aggregate_delta2=delta2,
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
  "aggregate_square_sensitivities",
  "bandinv_operator_norm_1_to_2_squared",
  "calibrate_shadow_jme",
  "epsilon_spent_for_shadow_jme_prefix",
  "jme_gamma_and_joint_sensitivity",
]
