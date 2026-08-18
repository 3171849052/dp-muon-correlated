"""Small adapter around jax_privacy's BandInvMF Toeplitz implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import optimization, toeplitz


@dataclass(frozen=True)
class BandInvMFStrategy:
  """A fitted BandInvMF factorization, using the project's matrix convention."""

  horizon: int
  bandwidth: int
  min_sep: int
  max_participations: int | None
  workload_coef: jax.Array | None
  noising_coef: jax.Array  # Coefficients of D = C^{-1}.
  strategy_coef: jax.Array  # Coefficients of C = D^{-1}.
  sensitivity_squared: jax.Array
  objective: jax.Array
  workload_matrix: jax.Array | None = None


def general_workload_banded_toeplitz_product(
    workload_matrix: jax.Array, noising_coef: jax.Array
) -> jax.Array:
  """Computes ``A @ D`` without materializing the banded Toeplitz ``D``.

  ``D`` has coefficients ``noising_coef`` and therefore
  ``(A D)[t, s] = sum_k d[k] A[t, s + k]``.  The indexed implementation has
  ``O(p T^2)`` work for a bandwidth ``p`` and is differentiable with JAX.
  """
  workload_matrix = jnp.asarray(workload_matrix)
  noising_coef = jnp.asarray(noising_coef)
  if workload_matrix.ndim != 2 or workload_matrix.shape[0] != workload_matrix.shape[1]:
    raise ValueError("workload_matrix must be square")
  if noising_coef.ndim != 1 or noising_coef.shape[0] < 1:
    raise ValueError("noising_coef must be a non-empty one-dimensional array")
  n = workload_matrix.shape[0]
  p = min(noising_coef.shape[0], n)
  coefficients = noising_coef[:p]
  columns = jnp.arange(n)[:, None] + jnp.arange(p)[None, :]
  valid = columns < n
  entries = jnp.take(workload_matrix, jnp.minimum(columns, n - 1), axis=1)
  entries = jnp.where(valid, entries, 0)
  return jnp.einsum("tsp,p->ts", entries, coefficients)


def general_workload_per_query_error(
    workload_matrix: jax.Array, noising_coef: jax.Array
) -> jax.Array:
  """Returns ``|A D|_2^2`` row by row for a general causal workload."""
  product = general_workload_banded_toeplitz_product(workload_matrix, noising_coef)
  return jnp.sum(product**2, axis=1)


def _validate_workload_matrix(workload_matrix: jax.Array, horizon: int) -> jax.Array:
  matrix = jnp.asarray(workload_matrix)
  if matrix.ndim != 2 or matrix.shape != (horizon, horizon):
    raise ValueError("workload_matrix must have shape (horizon, horizon)")
  values = np.asarray(matrix)
  if not np.all(np.isfinite(values)):
    raise ValueError("workload_matrix must contain only finite values")
  if np.any(np.triu(values, k=1) != 0):
    raise ValueError("workload_matrix must be causal (upper triangle must be zero)")
  return matrix


def _toeplitz_workload_coef(matrix: jax.Array) -> jax.Array | None:
  """Returns the Toeplitz coefficients when ``matrix`` is Toeplitz, else None."""
  values = np.asarray(matrix)
  n = values.shape[0]
  rows, columns = np.indices((n, n))
  expected = np.where(rows >= columns, values[rows - columns, 0], 0)
  if np.array_equal(values, expected):
    return matrix[:, 0]
  return None


def fit_bandinv_strategy(
    horizon: int,
    bandwidth: int,
    min_sep: int,
    *,
    max_participations: int | None = None,
    workload_coef: jax.Array | None = None,
    workload_matrix: jax.Array | None = None,
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

  if workload_coef is not None and workload_matrix is not None:
    raise ValueError("at most one of workload_coef and workload_matrix may be provided")
  matrix = None if workload_matrix is None else _validate_workload_matrix(workload_matrix, horizon)
  workload = jnp.ones(horizon) if workload_coef is None and matrix is None else (
      jnp.asarray(workload_coef) if workload_coef is not None else None
  )
  if workload is not None and (workload.ndim != 1 or workload.shape[0] != horizon):
    raise ValueError("workload_coef must be a one-dimensional array of length horizon")
  reduction_fn = {
      "mean": jnp.mean,
      "max": jnp.max,
      "last": lambda values: values[-1],
  }.get(reduction)
  if reduction_fn is None:
    raise ValueError("reduction must be one of: mean, max, last")

  if matrix is None:
    noising = toeplitz.optimize_banded_inverse_toeplitz(
        n=horizon,
        min_sep=min_sep,
        num_bands=bandwidth,
        workload_coef=workload,
        max_participations=max_participations,
        max_optimizer_steps=max_optimizer_steps,
        reduction_fn=reduction_fn,
    )
  else:
    # Use the same initialization and optimization wrapper as jax_privacy.
    # A Toeplitz matrix gets the identical BISR initialization as the fast path,
    # which also makes the two routes numerically equivalent on that workload.
    matrix_coef = _toeplitz_workload_coef(matrix)
    initial = toeplitz.banded_inverse_square_root_noising_coefs(
        bandwidth, workload_coef=matrix_coef
    )

    def loss_fn(coef: jax.Array) -> jax.Array:
      error = reduction_fn(general_workload_per_query_error(matrix, coef))
      sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
          n=horizon,
          noising_coef=coef,
          min_sep=min_sep,
          max_participations=max_participations,
          use_matrix_upper_bound=False,
      )
      return error * sensitivity_squared

    noising = optimization.optimize(
        loss_fn, initial, max_optimizer_steps=max_optimizer_steps
    )
    noising = noising / noising[0]
  strategy = toeplitz.inverse_coef(noising, horizon)
  sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon,
      noising_coef=noising,
      min_sep=min_sep,
      max_participations=max_participations,
  )
  objective = reduction_fn(
      general_workload_per_query_error(matrix, noising)
      if matrix is not None
      else toeplitz.per_query_error(
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
      workload_matrix=matrix,
  )


__all__ = [
    "BandInvMFStrategy",
    "fit_bandinv_strategy",
    "general_workload_banded_toeplitz_product",
    "general_workload_per_query_error",
]
