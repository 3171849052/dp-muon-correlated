"""Tests for the public fixed-LR Muon-Nesterov linear workload."""

import jax.numpy as jnp
import numpy as np
import pytest

from dp_muon.optim import (
    decayed_prefix_sum_workload_coef,
    fixed_lr_nesterov_decayed_trajectory_workload_coef,
    fixed_lr_nesterov_trajectory_workload_coef,
    nesterov_kernel_coef,
)


def _lower_toeplitz(coef):
  horizon = len(coef)
  row, column = jnp.indices((horizon, horizon))
  return jnp.where(row >= column, coef[row - column], 0.0)


def test_nesterov_kernel_matches_theoretical_coefficients():
  beta = 0.7
  horizon = 6
  index = jnp.arange(horizon)
  expected = jnp.where(index == 0, 1.0 - beta**2, (1.0 - beta) * beta ** (index + 1))
  np.testing.assert_allclose(nesterov_kernel_coef(horizon, beta), expected)


def test_fixed_lr_workload_is_cumsum_and_closed_form():
  horizon, beta, learning_rate = 7, 0.8, 0.03
  h = nesterov_kernel_coef(horizon, beta)
  actual = fixed_lr_nesterov_trajectory_workload_coef(horizon, beta, learning_rate)
  np.testing.assert_allclose(actual, learning_rate * jnp.cumsum(h))
  expected = learning_rate * (1.0 - beta ** (jnp.arange(horizon) + 2))
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_zero_momentum_is_sgd_kernel_and_prefix_trajectory():
  horizon, learning_rate = 5, 0.2
  np.testing.assert_array_equal(nesterov_kernel_coef(horizon, 0.0), jnp.eye(1, horizon, 0)[0])
  np.testing.assert_allclose(
      fixed_lr_nesterov_trajectory_workload_coef(horizon, 0.0, learning_rate),
      learning_rate * jnp.ones(horizon),
  )


def test_zero_weight_decay_exactly_recovers_existing_workloads():
  horizon, beta, learning_rate = 7, 0.8, 0.03
  np.testing.assert_array_equal(
      decayed_prefix_sum_workload_coef(horizon, learning_rate, 0.0),
      jnp.ones(horizon),
  )
  np.testing.assert_array_equal(
      fixed_lr_nesterov_decayed_trajectory_workload_coef(
          horizon, beta, learning_rate, 0.0
      ),
      fixed_lr_nesterov_trajectory_workload_coef(horizon, beta, learning_rate),
  )


def test_adamw_decayed_prefix_workload_is_rho_powers():
  horizon, learning_rate, weight_decay = 5, 0.2, 0.3
  rho = 1.0 - learning_rate * weight_decay
  np.testing.assert_allclose(
      decayed_prefix_sum_workload_coef(horizon, learning_rate, weight_decay),
      rho ** jnp.arange(horizon),
  )


def test_muon_decayed_workload_matches_explicit_eta_p_rho_h():
  horizon, beta, learning_rate, weight_decay = 6, 0.7, 0.1, 0.2
  h = nesterov_kernel_coef(horizon, beta)
  rho = 1.0 - learning_rate * weight_decay
  explicit = learning_rate * _lower_toeplitz(rho ** jnp.arange(horizon)) @ h
  np.testing.assert_allclose(
      fixed_lr_nesterov_decayed_trajectory_workload_coef(
          horizon, beta, learning_rate, weight_decay
      ),
      explicit, rtol=2e-7, atol=2e-8,
  )


@pytest.mark.parametrize(
    "fn,args,message",
    [
        (nesterov_kernel_coef, (0, 0.5), "horizon"),
        (nesterov_kernel_coef, (2, -0.1), "momentum"),
        (nesterov_kernel_coef, (2, 1.0), "momentum"),
        (nesterov_kernel_coef, (2, float("nan")), "momentum"),
        (fixed_lr_nesterov_trajectory_workload_coef, (2, 0.5, 0.0), "learning_rate"),
        (fixed_lr_nesterov_trajectory_workload_coef, (2, 0.5, float("inf")), "learning_rate"),
    ],
)
def test_invalid_public_configuration_fails_fast(fn, args, message):
  with pytest.raises(ValueError, match=message):
    fn(*args)
