"""Regression tests: JAX Privacy microbatching must not alter M6 semantics."""

import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv, make_clipped_gradient_query
from dp_muon.training import init_nonamplified_bandinv_state, make_nonamplified_bandinv_train_step


def _loss(params, batch):
  return params * batch["x"][0]


def _artifacts():
  horizon, momentum, learning_rate = 3, 0.4, 0.1
  coef = jnp.asarray((1.0, -0.2), jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=coef, min_sep=1, max_participations=None
  )
  strategy = BandInvMFStrategy(
      horizon=horizon, bandwidth=2, min_sep=1, max_participations=None,
      workload_coef=fixed_lr_nesterov_trajectory_workload_coef(horizon, momentum, learning_rate),
      noising_coef=coef, strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=sensitivity, objective=jnp.array(0.0),
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2.0, delta=1e-5, clip_norm=1.0, normalize_by=4.0,
      adjacency="add_remove", sensitivity_squared=float(sensitivity),
  )
  return strategy, calibration, ParticipationSpec(horizon, 1, None), momentum, learning_rate


def test_clipped_query_is_independent_of_microbatch_size():
  batch = {"x": jnp.asarray([3.0, -1.0, 0.5, 2.0], jnp.float32)}
  outputs = [
      make_clipped_gradient_query(_loss, clip_norm=1.0, normalize_by=4.0, microbatch_size=size)(jnp.array(0.0), batch)
      for size in (None, 1, 2, 4)
  ]
  for output in outputs[1:]:
    np.testing.assert_allclose(output, outputs[0], rtol=1e-6, atol=1e-6)


def test_full_m6_state_is_independent_of_microbatch_size():
  # The preceding suite deliberately JITs many independent reference kernels.
  # Do not retain those executable caches while compiling four M6 variants.
  jax.clear_caches()
  strategy, calibration, participation, momentum, learning_rate = _artifacts()
  batch = {"x": jnp.asarray([3.0, -1.0, 0.5, 2.0], jnp.float32)}
  outputs = []
  for size in (None, 1, 2, 4):
    step = make_nonamplified_bandinv_train_step(
        _loss, strategy, calibration, participation, momentum, learning_rate, microbatch_size=size
    )
    state = init_nonamplified_bandinv_state(jnp.array(0.0), strategy, jax.random.key(9))
    outputs.append(jax.jit(step)(state, batch))
  for output in outputs[1:]:
    np.testing.assert_allclose(output.params, outputs[0].params, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output.nesterov_state.momentum, outputs[0].nesterov_state.momentum, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(output.noise_state.step, outputs[0].noise_state.step)
    np.testing.assert_array_equal(jax.random.key_data(output.rng_key), jax.random.key_data(outputs[0].rng_key))
    assert int(output.noise_state.step) == 1
    assert int(output.nesterov_state.step) == 1
