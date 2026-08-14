"""Small adapter around jax_privacy's BandInvMF Toeplitz implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import toeplitz


@dataclass(frozen=True)
class BandInvMFStrategy:
  """A fitted BandInvMF factorization, using the project's matrix convention."""

  horizon: int
  bandwidth: int
  min_sep: int
  max_participations: int | None
  workload_coef: jax.Array
  noising_coef: jax.Array  # Coefficients of D = C^{-1}.
  strategy_coef: jax.Array  # Coefficients of C = D^{-1}.
  sensitivity_squared: jax.Array
  objective: jax.Array


def fit_bandinv_strategy(
    horizon: int,
    bandwidth: int,
    min_sep: int,
    *,
    max_participations: int | None = None,
    workload_coef: jax.Array | None = None,
    max_optimizer_steps: int = 1000,
    reduction: Literal["mean", "max", "last"] = "mean",
) -> BandInvMFStrategy:
  """Fits BandInvMF by directly delegating the math to ``jax_privacy``.

  ``bandwidth`` is the number of optimized lower-triangular Toeplitz
  coefficients, including the diagonal. If no workload is supplied, the public
  prefix-sum workload is used.
  """
  if horizon < 1:
    raise ValueError("horizon must be positive")
  if not 1 <= bandwidth <= horizon:
    raise ValueError("bandwidth must be in [1, horizon]")
  if min_sep < 1:
    raise ValueError("min_sep must be positive")
  if max_participations is not None and max_participations < 1:
    raise ValueError("max_participations must be positive when supplied")
  if max_optimizer_steps < 1:
    raise ValueError("max_optimizer_steps must be positive")

  workload = jnp.ones(horizon) if workload_coef is None else jnp.asarray(workload_coef)
  if workload.ndim != 1 or workload.shape[0] != horizon:
    raise ValueError("workload_coef must be a one-dimensional array of length horizon")
  reduction_fn = {
      "mean": jnp.mean,
      "max": jnp.max,
      "last": lambda values: values[-1],
  }.get(reduction)
  if reduction_fn is None:
    raise ValueError("reduction must be one of: mean, max, last")

  noising = toeplitz.optimize_banded_inverse_toeplitz(
      n=horizon,
      min_sep=min_sep,
      num_bands=bandwidth,
      workload_coef=workload,
      max_participations=max_participations,
      max_optimizer_steps=max_optimizer_steps,
      reduction_fn=reduction_fn,
  )
  strategy = toeplitz.inverse_coef(noising, horizon)
  sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon,
      noising_coef=noising,
      min_sep=min_sep,
      max_participations=max_participations,
  )
  objective = reduction_fn(
      toeplitz.per_query_error(
          noising_coef=noising, n=horizon, workload_coef=workload
      )
  ) * sensitivity_squared
  return BandInvMFStrategy(
      horizon=horizon,
      bandwidth=bandwidth,
      min_sep=min_sep,
      max_participations=max_participations,
      workload_coef=workload,
      noising_coef=noising,
      strategy_coef=strategy,
      sensitivity_squared=sensitivity_squared,
      objective=objective,
  )
