"""Tests for the streaming BandInvMF correlated-noise generator."""

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import (
    filter_latent_noise,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)


def _dense_toeplitz(coef: jax.Array, horizon: int) -> jax.Array:
  rows = jnp.arange(horizon)[:, None]
  cols = jnp.arange(horizon)[None, :]
  offsets = rows - cols
  return jnp.where(
      (offsets >= 0) & (offsets < coef.shape[0]), coef[jnp.clip(offsets, 0)], 0
  )


def _stream_scalar(latents: jax.Array, coef: jax.Array) -> jax.Array:
  state = init_bandinv_noise_state(latents[0], coef.shape[0])
  outputs = []
  for latent in latents:
    output, state = filter_latent_noise(state, latent, coef)
    outputs.append(output)
  return jnp.stack(outputs)


def test_deterministic_dense_vs_streaming():
  coef = jnp.array([1.0, -0.4, 0.2])
  latent = jnp.array([0.3, -2.0, 1.1, 0.5, -0.8])
  np.testing.assert_allclose(
      _stream_scalar(latent, coef), _dense_toeplitz(coef, len(latent)) @ latent, rtol=1e-6, atol=1e-6
  )


def test_startup_boundary():
  coef = jnp.array([1.0, -0.4, 0.2])
  latent = jnp.array([2.0, -3.0])
  output = _stream_scalar(latent, coef)
  np.testing.assert_allclose(output[0], coef[0] * latent[0])
  np.testing.assert_allclose(output[1], coef[0] * latent[1] + coef[1] * latent[0])


def test_bandwidth_one_is_scalar_multiplication():
  latent = jnp.array([0.2, -1.5, 3.0])
  np.testing.assert_allclose(_stream_scalar(latent, jnp.array([1.0])), latent)
  np.testing.assert_allclose(_stream_scalar(latent, jnp.array([-2.5])), -2.5 * latent)


def test_pytree_dense_vs_streaming_and_leaf_metadata():
  coef = jnp.array([1.0, 0.5])
  latents = [
      {"a": jnp.arange(3, dtype=jnp.float32), "b": jnp.arange(2, dtype=jnp.float32).reshape(2, 1)},
      {"a": jnp.full((3,), 2.0, dtype=jnp.float32), "b": jnp.full((2, 1), -1.0, dtype=jnp.float32)},
      {"a": jnp.full((3,), -3.0, dtype=jnp.float32), "b": jnp.full((2, 1), 0.5, dtype=jnp.float32)},
  ]
  state = init_bandinv_noise_state(latents[0], 2)
  outputs = []
  for latent in latents:
    output, state = filter_latent_noise(state, latent, coef)
    outputs.append(output)
  assert jax.tree_util.tree_structure(outputs[0]) == jax.tree_util.tree_structure(latents[0])
  dense = _dense_toeplitz(coef, len(latents))
  for name in ("a", "b"):
    actual = jnp.stack([output[name] for output in outputs])
    expected = jnp.tensordot(dense, jnp.stack([latent[name] for latent in latents]), axes=1)
    assert actual.shape == (len(latents), *latents[0][name].shape)
    assert actual.dtype == latents[0][name].dtype
    np.testing.assert_allclose(actual, expected)


def _sample_sequence(key: jax.Array, steps: int = 5):
  state = init_bandinv_noise_state(jnp.zeros((3,), dtype=jnp.float32), 2)
  outputs = []
  for _ in range(steps):
    output, state, key = sample_bandinv_noise(key, state, jnp.array([1.0, -0.25]), 0.7)
    outputs.append(output)
  return jnp.stack(outputs), state, key


def test_prng_reproducibility_and_key_progression():
  first, _, _ = _sample_sequence(jax.random.key(123))
  second, _, _ = _sample_sequence(jax.random.key(123))
  different, _, _ = _sample_sequence(jax.random.key(124))
  np.testing.assert_array_equal(first, second)
  assert not np.array_equal(np.asarray(first), np.asarray(different))
  assert not np.array_equal(np.asarray(first[0]), np.asarray(first[1]))


def test_checkpoint_resume_matches_uninterrupted_run():
  key = jax.random.key(9)
  state = init_bandinv_noise_state(jnp.zeros((2,), dtype=jnp.float32), 3)
  coef = jnp.array([1.0, -0.3, 0.1])
  uninterrupted = []
  for _ in range(7):
    output, state, key = sample_bandinv_noise(key, state, coef, 0.4)
    uninterrupted.append(output)

  key = jax.random.key(9)
  state = init_bandinv_noise_state(jnp.zeros((2,), dtype=jnp.float32), 3)
  resumed = []
  for _ in range(3):
    output, state, key = sample_bandinv_noise(key, state, coef, 0.4)
    resumed.append(output)
  checkpoint_state, checkpoint_key = state, key
  for _ in range(4):
    output, checkpoint_state, checkpoint_key = sample_bandinv_noise(checkpoint_key, checkpoint_state, coef, 0.4)
    resumed.append(output)
  np.testing.assert_array_equal(jnp.stack(uninterrupted), jnp.stack(resumed))


def test_empirical_covariance_matches_tau_squared_d_dt():
  # Treat Monte Carlo draws as independent vector coordinates: the filter acts
  # independently on every coordinate, so this is vectorized streaming MC.
  coefficient = jnp.array([1.0, -0.4, 0.2])
  horizon, draws, tau = 5, 12000, 0.7
  latent = tau * jax.random.normal(jax.random.key(2026), (horizon, draws))
  streamed = _stream_scalar(latent, coefficient)  # (horizon, draws)
  empirical = np.cov(np.asarray(streamed), bias=True)
  dense = np.asarray(_dense_toeplitz(coefficient, horizon))
  expected = tau**2 * dense @ dense.T
  np.testing.assert_allclose(empirical, expected, rtol=0.06, atol=0.012)
