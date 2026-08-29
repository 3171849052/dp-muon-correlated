"""Tests for explicit STP AdamW and its BandInvMF composition."""

import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy, sample_bandinv_noise
from dp_muon.optim import STPAdamW, STPAdamWState
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
    make_clipped_gradient_query,
)
from dp_muon.training import (
    init_nonamplified_bandinv_stp_dpadamw_state,
    make_nonamplified_bandinv_stp_dpadamw_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
import dp_muon.training.nonamplified_bandinv_stp_dpadamw as stp_training


def _loss(params, batch):
  return params["scalar"] * batch["scalar"][0] + jnp.vdot(
      params["vector"], batch["vector"][0]
  )


def _strategy_and_calibration(*, horizon=4, clip_norm=4.0):
  noising_coef = jnp.array([1.0, -0.25], dtype=jnp.float32)
  sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon,
      noising_coef=noising_coef,
      min_sep=1,
      max_participations=None,
  )
  strategy = BandInvMFStrategy(
      horizon=horizon,
      bandwidth=2,
      min_sep=1,
      max_participations=None,
      workload_coef=jnp.ones((horizon,), dtype=jnp.float32),
      noising_coef=noising_coef,
      strategy_coef=jnp.ones((horizon,), dtype=jnp.float32),
      sensitivity_squared=sensitivity_squared,
      objective=jnp.array(0.0, dtype=jnp.float32),
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=clip_norm,
      normalize_by=1.0,
      adjacency="add_remove",
      sensitivity_squared=float(sensitivity_squared),
  )
  return strategy, calibration, ParticipationSpec(horizon, 1, None)


def _tree_allclose(actual, expected):
  assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(actual),
      jax.tree_util.tree_leaves(expected),
      strict=True,
  ):
    if jnp.issubdtype(jnp.asarray(actual_leaf).dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(
          jax.random.key_data(actual_leaf), jax.random.key_data(expected_leaf)
      )
    else:
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-7)


def test_stp_adamw_m_v_and_bias_correction_recurrence():
  optimizer = STPAdamW(
      learning_rate=0.1,
      beta1=0.5,
      beta2=0.25,
      eps=0.1,
      scale_eps=0.5,
  )
  params = jnp.array([1.0, 2.0], dtype=jnp.float32)
  state = optimizer.init(params)
  first_gradient = jnp.array([2.0, 4.0], dtype=jnp.float32)
  updates, state = optimizer.update(first_gradient, state, params)
  np.testing.assert_array_equal(state.count, 1)
  np.testing.assert_allclose(state.m, [1.0, 2.0])
  np.testing.assert_allclose(state.v, [3.0, 12.0])
  expected_first = -0.1 * first_gradient / (jnp.abs(first_gradient) + 0.1)
  np.testing.assert_allclose(updates, expected_first)

  second_gradient = jnp.array([-1.0, 3.0], dtype=jnp.float32)
  updates, state = optimizer.update(second_gradient, state, params)
  expected_m = 0.5 * jnp.array([1.0, 2.0]) + 0.5 * second_gradient
  expected_v = 0.25 * jnp.array([3.0, 12.0]) + 0.75 * second_gradient**2
  np.testing.assert_array_equal(state.count, 2)
  np.testing.assert_allclose(state.m, expected_m)
  np.testing.assert_allclose(state.v, expected_v)
  expected = -0.1 * (
      (expected_m / (1.0 - 0.5**2))
      / (jnp.sqrt(expected_v / (1.0 - 0.25**2)) + 0.1)
  )
  np.testing.assert_allclose(updates, expected)


def test_t1_scaling_uses_zero_previous_v_hat():
  optimizer = STPAdamW(learning_rate=0.1, scale_eps=0.5)
  state = optimizer.init({"a": jnp.ones(2), "b": jnp.ones(1)})
  scale = optimizer.scale(state)
  _tree_allclose(scale, {"a": jnp.full(2, 2.0), "b": jnp.full(1, 2.0)})


def test_scaled_query_is_s_times_g_without_clipping():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {
      "scalar": jnp.array([2.0, -1.0]),
      "vector": jnp.array([[1.0, 2.0], [-2.0, 1.0]]),
  }
  scale = {"scalar": jnp.array(3.0), "vector": jnp.array([2.0, 4.0])}
  query = make_clipped_gradient_query(
      _loss,
      clip_norm=100.0,
      normalize_by=1.0,
      pre_clipping_transform=lambda gradient: jax.tree_util.tree_map(
          lambda factor, value: factor * value, scale, gradient
      ),
  )
  output = query(params, batch)
  expected = {
      "scalar": jnp.array(3.0),
      "vector": jnp.array([-2.0, 12.0]),
  }
  _tree_allclose(output, expected)


def test_scaled_gradient_global_l2_clipping_is_global():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {"scalar": jnp.array([3.0]), "vector": jnp.array([[0.0, 4.0]])}
  scale = {"scalar": jnp.array(2.0), "vector": jnp.ones(2)}
  query = make_clipped_gradient_query(
      _loss,
      clip_norm=4.0,
      normalize_by=1.0,
      pre_clipping_transform=lambda gradient: jax.tree_util.tree_map(
          lambda factor, value: factor * value, scale, gradient
      ),
  )
  output = query(params, batch)
  factor = 4.0 / np.sqrt(6.0**2 + 4.0**2)
  np.testing.assert_allclose(output["scalar"], 6.0 * factor)
  np.testing.assert_allclose(output["vector"], [0.0, 4.0 * factor])


def test_scale_one_query_matches_existing_clipping_path():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {"scalar": jnp.array([3.0, 1.0]), "vector": jnp.array([[0.0, 4.0], [2.0, -2.0]])}
  ordinary = make_clipped_gradient_query(
      _loss, clip_norm=4.0, normalize_by=5.0
  )(params, batch)
  scaled = make_clipped_gradient_query(
      _loss,
      clip_norm=4.0,
      normalize_by=5.0,
      pre_clipping_transform=lambda gradient: gradient,
  )(params, batch)
  _tree_allclose(scaled, ordinary)


def test_bandinv_noise_state_and_optimizer_step_remain_aligned():
  strategy, calibration, participation = _strategy_and_calibration(clip_norm=100.0)
  train_step, optimizer = make_nonamplified_bandinv_stp_dpadamw_train_step(
      _loss,
      strategy,
      calibration,
      participation,
      learning_rate=0.01,
      scale_eps=0.5,
  )
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  initial = init_nonamplified_bandinv_stp_dpadamw_state(
      params, strategy, jax.random.key(7), optimizer
  )
  batch = {"scalar": jnp.array([1.0]), "vector": jnp.array([[2.0, -1.0]])}
  actual = train_step(initial, batch)
  expected_noise, expected_noise_state, expected_key = sample_bandinv_noise(
      initial.rng_key,
      initial.noise_state,
      strategy.noising_coef,
      calibration.iid_noise_std,
  )
  del expected_noise
  _tree_allclose(actual.noise_state, expected_noise_state)
  _tree_allclose(actual.rng_key, expected_key)
  assert int(actual.step) == int(actual.noise_state.step) == int(actual.optimizer_state.count) == 1


def test_checkpoint_resume_preserves_stp_and_correlated_state(tmp_path, monkeypatch):
  strategy, calibration, participation = _strategy_and_calibration(clip_norm=100.0)
  real_sampler = stp_training.sample_bandinv_noise

  def zero_noise(key, state, coef, std):
    return real_sampler(key, state, coef, 0.0)

  monkeypatch.setattr(stp_training, "sample_bandinv_noise", zero_noise)
  train_step, optimizer = make_nonamplified_bandinv_stp_dpadamw_train_step(
      _loss,
      strategy,
      calibration,
      participation,
      learning_rate=0.01,
      scale_eps=0.5,
  )
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  initial = init_nonamplified_bandinv_stp_dpadamw_state(
      params, strategy, jax.random.key(8), optimizer
  )
  batches = [
      {"scalar": jnp.array([1.0]), "vector": jnp.array([[2.0, -1.0]])},
      {"scalar": jnp.array([-1.0]), "vector": jnp.array([[1.0, 3.0]])},
      {"scalar": jnp.array([2.0]), "vector": jnp.array([[-2.0, 1.0]])},
  ]
  uninterrupted = initial
  for batch in batches:
    uninterrupted = train_step(uninterrupted, batch)
  partial = train_step(train_step(initial, batches[0]), batches[1])
  checkpoint = tmp_path / "stp.pkl"
  save_checkpoint(
      checkpoint,
      state=partial,
      current_step=2,
      experiment_config={"algorithm": "stp"},
      artifact_identifiers={"strategy": "shared-prefix"},
  )
  resumed = load_checkpoint(checkpoint)["state"]
  assert isinstance(resumed.optimizer_state, STPAdamWState)
  np.testing.assert_array_equal(resumed.optimizer_state.count, 2)
  _tree_allclose(resumed.optimizer_state.m, partial.optimizer_state.m)
  _tree_allclose(resumed.optimizer_state.v, partial.optimizer_state.v)
  _tree_allclose(resumed.noise_state, partial.noise_state)
  resumed = train_step(resumed, batches[2])
  _tree_allclose(resumed, uninterrupted)


def test_jitted_stp_step_matches_eager():
  strategy, calibration, participation = _strategy_and_calibration(clip_norm=100.0)
  train_step, optimizer = make_nonamplified_bandinv_stp_dpadamw_train_step(
      _loss,
      strategy,
      calibration,
      participation,
      learning_rate=0.01,
      scale_eps=0.5,
  )
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  initial = init_nonamplified_bandinv_stp_dpadamw_state(
      params, strategy, jax.random.key(9), optimizer
  )
  batch = {"scalar": jnp.array([1.0]), "vector": jnp.array([[2.0, -1.0]])}
  _tree_allclose(jax.jit(train_step)(initial, batch), train_step(initial, batch))
