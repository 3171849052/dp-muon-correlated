"""Focused regression tests for Experiment 3's online shadow step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from exp3.online_shadow import (
    aggregate_ratio,
    init_online_shadow_state,
    make_online_shadow_train_step,
)


def _setup():
  horizon = 3
  coef = jnp.asarray([1.0, -0.2], jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=coef, min_sep=1, max_participations=1
  )
  strategy = BandInvMFStrategy(
      horizon=horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=jnp.ones((horizon,), jnp.float32), noising_coef=coef,
      strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=sensitivity, objective=jnp.array(0., jnp.float32),
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2., delta=1e-5, clip_norm=10., normalize_by=1.,
      adjacency="add_remove", sensitivity_squared=float(sensitivity),
  )
  participation = ParticipationSpec(horizon, 1, 1)
  def loss(params, batch):
    return jnp.sum((params["w"] - batch["target"]) ** 2)
  return strategy, calibration, participation, loss


def _leaves_equal(left, right):
  for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
    if jnp.issubdtype(jnp.asarray(a).dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
    else:
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_jit_step_and_standard_semantics_match():
  strategy, calibration, participation, loss = _setup()
  online_step, online_optimizer = make_online_shadow_train_step(
      loss, strategy, calibration, participation, learning_rate=.01, weight_decay=.01,
  )
  standard_step, standard_optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      loss, strategy, calibration, participation, learning_rate=.01, weight_decay=.01,
  )
  params = {"w": jnp.array([1., -1.], jnp.float32)}
  key = jax.random.key(7)
  online = init_online_shadow_state(params, strategy, key, online_optimizer)
  standard = init_nonamplified_bandinv_dpadamw_state(params, strategy, key, standard_optimizer)
  batch = {"target": jnp.array([.25, -.5], jnp.float32)}
  online = jax.jit(online_step)(online, batch)
  standard = jax.jit(standard_step)(standard, batch)
  _leaves_equal(online.params, standard.params)
  _leaves_equal(online.optimizer_state, standard.optimizer_state)
  _leaves_equal(online.noise_state, standard.noise_state)
  np.testing.assert_array_equal(jax.random.key_data(online.rng_key), jax.random.key_data(standard.rng_key))
  assert int(online.step) == int(standard.step) == 1


def test_aggregate_ratio_and_synthetic_recurrence():
  eta, rho = .5, .8
  response = jnp.array([1., -2.])
  x = -eta * response
  d = rho * jnp.zeros_like(x) + x
  expected_j, expected_d = float(jnp.sum(d * d)), float(jnp.sum(x * x))
  assert float(aggregate_ratio(expected_j, expected_d)) == pytest.approx(1.)
  assert float(jnp.sum(d * d)) == pytest.approx(expected_j)
  # A second prefix follows d_t = rho*d_(t-1) + x_t and D_t += ||x_t||^2.
  d2 = rho * d + x
  assert float(jnp.sum(d2 * d2)) == pytest.approx(float(jnp.sum((rho * d + x) ** 2)))


def test_participation_validation_is_applied():
  strategy, calibration, _, loss = _setup()
  with pytest.raises(ValueError):
    make_online_shadow_train_step(
        loss, strategy, calibration, ParticipationSpec(4, 1, 1), learning_rate=.01,
    )
