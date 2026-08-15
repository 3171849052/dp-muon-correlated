from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.privacy import calibrate_nonamplified_iid
from dp_muon.training import (
    init_nonamplified_dpsgd_state,
    make_nonamplified_dpsgd_train_step,
)
import dp_muon.training.nonamplified_dpsgd as dpsgd


def _loss(params, batch):
  return params * batch[0]


def _calibration(noise_std=0.0):
  actual = calibrate_nonamplified_iid(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=10.0,
      normalize_by=1.0,
      adjacency="add_remove",
      max_participations=2,
  )
  return replace(actual, iid_noise_std=noise_std)


def test_zero_noise_reduces_to_nonprivate_sgd_momentum():
  step = make_nonamplified_dpsgd_train_step(_loss, _calibration(), 0.5, 0.1)
  state = init_nonamplified_dpsgd_state(jnp.array(2.0, jnp.float32), jax.random.key(0))
  values = (1.0, -2.0, 3.0)
  actual = []
  for value in values:
    state = step(state, jnp.array([value], jnp.float32))
    actual.append(state.params)
  # velocities are [1, -1.5, 2.25], and parameters subtract 0.1 times each.
  np.testing.assert_allclose(jnp.stack(actual), jnp.array([1.9, 2.05, 1.825]))
  np.testing.assert_allclose(state.momentum_state.velocity, 2.25)


def test_fixed_seed_is_reproducible_and_jit_matches_eager():
  calibration = _calibration(noise_std=0.7)
  step = make_nonamplified_dpsgd_train_step(_loss, calibration, 0.4, 0.1)
  initial = init_nonamplified_dpsgd_state(jnp.array(0.0, jnp.float32), jax.random.key(9))
  batch = jnp.array([1.0], jnp.float32)
  eager = step(initial, batch)
  repeated = step(initial, batch)
  compiled = jax.jit(step)(initial, batch)
  np.testing.assert_allclose(repeated.params, eager.params)
  np.testing.assert_allclose(compiled.params, eager.params)
  np.testing.assert_allclose(compiled.momentum_state.velocity, eager.momentum_state.velocity)
  np.testing.assert_array_equal(
      jax.random.key_data(compiled.rng_key), jax.random.key_data(eager.rng_key)
  )


def test_noisy_gradient_not_clean_gradient_enters_momentum(monkeypatch):
  def fixed_noise(key, template, noise_std):
    return jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * 2.0, template), key

  monkeypatch.setattr(dpsgd, "_sample_iid_gaussian_noise", fixed_noise)
  step = make_nonamplified_dpsgd_train_step(_loss, _calibration(), 0.5, 0.1)
  state = init_nonamplified_dpsgd_state(jnp.array(0.0, jnp.float32), jax.random.key(1))
  state = step(state, jnp.array([1.0], jnp.float32))
  np.testing.assert_allclose(state.momentum_state.velocity, 3.0)
  np.testing.assert_allclose(state.params, -0.3)


def test_microbatch_clipping_path_runs():
  step = make_nonamplified_dpsgd_train_step(
      _loss, _calibration(), 0.0, 0.1, microbatch_size=2
  )
  state = init_nonamplified_dpsgd_state(jnp.array(0.0, jnp.float32), jax.random.key(2))
  result = step(state, jnp.array([1.0, 2.0, 3.0, 4.0], jnp.float32))
  # Fixed normalization is one, so all four unclipped example gradients sum.
  np.testing.assert_allclose(result.params, -1.0)
  np.testing.assert_allclose(result.momentum_state.velocity, 10.0)
