"""Tests for general causal BandInvMF workloads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dp_muon.bandinvmf import (
    fit_bandinv_strategy,
    general_workload_banded_toeplitz_product,
    general_workload_per_query_error,
)


def _lower_toeplitz(coef, n=None):
  n = len(coef) if n is None else n
  coef = jnp.pad(coef, (0, n - len(coef)))
  rows, columns = jnp.indices((n, n))
  return jnp.where(rows >= columns, coef[rows - columns], 0.0)


def test_general_product_and_error_match_dense_multiplication():
  matrix = jnp.array([[1.0, 0, 0, 0], [2, 3, 0, 0], [4, 5, 6, 0], [7, 8, 9, 10]])
  coef = jnp.array([0.7, -0.2, 0.05])
  dense = matrix @ _lower_toeplitz(coef, matrix.shape[0])
  np.testing.assert_allclose(general_workload_banded_toeplitz_product(matrix, coef), dense)
  np.testing.assert_allclose(
      general_workload_per_query_error(matrix, coef), jnp.sum(dense**2, axis=1)
  )
  gradient = jax.grad(lambda d: jnp.sum(general_workload_per_query_error(matrix, d)))(coef)
  assert np.all(np.isfinite(np.asarray(gradient)))


def test_toeplitz_matrix_general_fit_matches_legacy_fit():
  coef = jnp.array([1.0, 0.7, 0.2, 0.05, 0.01])
  kwargs = dict(horizon=5, bandwidth=3, min_sep=1, max_optimizer_steps=5)
  legacy = fit_bandinv_strategy(workload_coef=coef, **kwargs)
  general = fit_bandinv_strategy(workload_matrix=_lower_toeplitz(coef), **kwargs)
  np.testing.assert_allclose(general.noising_coef, legacy.noising_coef, rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(general.objective, legacy.objective, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"workload_matrix": jnp.ones((3, 2))}, "shape"),
        ({"workload_matrix": jnp.tril(jnp.ones((3, 3))).at[0, 2].set(1)}, "causal"),
        ({"workload_matrix": jnp.tril(jnp.ones((3, 3))).at[1, 1].set(jnp.inf)}, "finite"),
        ({"workload_coef": jnp.ones(3), "workload_matrix": jnp.eye(3)}, "at most one"),
    ],
)
def test_general_workload_validation(kwargs, message):
  with pytest.raises(ValueError, match=message):
    fit_bandinv_strategy(3, 2, 1, max_optimizer_steps=1, **kwargs)
