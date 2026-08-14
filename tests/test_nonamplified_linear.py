"""End-to-end tests for the non-amplified linear BandInvMF trainer."""

from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy, sample_bandinv_noise
from dp_muon.optim import (
    fixed_lr_nesterov_trajectory_workload_coef,
    nesterov_kernel_coef,
)
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
)
from dp_muon.training import (
    init_nonamplified_bandinv_state,
    make_nonamplified_bandinv_train_step,
    validate_nonamplified_bandinv_setup,
)
import dp_muon.training.nonamplified_linear as nonamplified_linear


def _scalar_linear_loss(params, batch):
  """A one-example loss whose gradient is the public scalar batch value."""
  return params * batch[0]


def _nested_linear_loss(params, batch):
  return (
      params["left"][0] * batch["left"][0][0]
      + jnp.vdot(params["left"][1], batch["left"][1][0])
      + jnp.vdot(params["right"]["matrix"], batch["right"]["matrix"][0])
  )


def _artifacts(
    *, horizon=6, momentum=0.6, learning_rate=0.15, noising_coef=(1.0, -0.35)
):
  noising_coef = jnp.asarray(noising_coef, dtype=jnp.float32)
  sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon,
      noising_coef=noising_coef,
      min_sep=1,
      max_participations=None,
  )
  strategy = BandInvMFStrategy(
      horizon=horizon,
      bandwidth=len(noising_coef),
      min_sep=1,
      max_participations=None,
      workload_coef=fixed_lr_nesterov_trajectory_workload_coef(
          horizon, momentum, learning_rate
      ),
      noising_coef=noising_coef,
      strategy_coef=jnp.array([1.0], dtype=jnp.float32),
      sensitivity_squared=sensitivity_squared,
      objective=jnp.array(0.0, dtype=jnp.float32),
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=2.0,
      normalize_by=1.0,
      adjacency="add_remove",
      sensitivity_squared=float(sensitivity_squared),
  )
  return strategy, calibration, ParticipationSpec(horizon, 1, None)


def _lower_toeplitz(coef, horizon):
  rows = jnp.arange(horizon)[:, None]
  cols = jnp.arange(horizon)[None, :]
  offsets = rows - cols
  return jnp.where(
      (offsets >= 0) & (offsets < len(coef)), coef[jnp.maximum(offsets, 0)], 0
  )


def _latent_sequence(key, *, horizon, std):
  """Independent dense oracle for M2's scalar one-leaf PRNG convention."""
  latent = []
  for _ in range(horizon):
    key, sample_key = jax.random.split(key)
    leaf_key = jax.random.split(sample_key, 1)[0]
    latent.append(jax.random.normal(leaf_key, (), dtype=jnp.float32) * std)
  return jnp.stack(latent)


def _tree_allclose(actual, expected):
  assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(actual),
      jax.tree_util.tree_leaves(expected),
      strict=True,
  ):
    if jnp.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(
          jax.random.key_data(actual_leaf), jax.random.key_data(expected_leaf)
      )
    else:
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-6)


def test_dense_multistep_oracle_matches_runtime_and_conditional_noise_path():
  horizon, beta, learning_rate = 6, 0.6, 0.15
  strategy, calibration, participation = _artifacts(
      horizon=horizon, momentum=beta, learning_rate=learning_rate
  )
  train_step = make_nonamplified_bandinv_train_step(
      _scalar_linear_loss, strategy, calibration, participation, beta, learning_rate
  )
  initial = jnp.array(1.25, dtype=jnp.float32)
  key = jax.random.key(29)
  state = init_nonamplified_bandinv_state(initial, strategy, key)
  query_values = jnp.array([0.5, -1.0, 1.5, 0.25, -0.75, 1.0], dtype=jnp.float32)
  runtime_trajectory = []
  for value in query_values:
    state = train_step(state, jnp.array([value], dtype=jnp.float32))
    runtime_trajectory.append(state.params)
  runtime_trajectory = jnp.stack(runtime_trajectory)

  d = _lower_toeplitz(strategy.noising_coef, horizon)
  h = _lower_toeplitz(nesterov_kernel_coef(horizon, beta), horizon)
  a = learning_rate * jnp.tril(jnp.ones((horizon, horizon))) @ h
  latent = _latent_sequence(key, horizon=horizon, std=calibration.iid_noise_std)
  correlated_noise = d @ latent
  clean_trajectory = initial - a @ query_values
  expected_trajectory = initial - a @ (query_values + correlated_noise)

  np.testing.assert_allclose(runtime_trajectory, expected_trajectory, rtol=1e-6, atol=1e-6)
  # This is conditional on the exact same q sequence, not an averaging claim.
  np.testing.assert_allclose(
      runtime_trajectory - clean_trajectory,
      -a @ d @ latent,
      rtol=1e-6,
      atol=1e-6,
  )


def test_momentum_zero_is_correlated_noise_fixed_lr_dp_sgd():
  horizon, learning_rate = 5, 0.2
  strategy, calibration, participation = _artifacts(
      horizon=horizon,
      momentum=0.0,
      learning_rate=learning_rate,
      noising_coef=(1.0, -0.2, 0.1),
  )
  train_step = make_nonamplified_bandinv_train_step(
      _scalar_linear_loss, strategy, calibration, participation, 0.0, learning_rate
  )
  initial, key = jnp.array(-0.5, dtype=jnp.float32), jax.random.key(7)
  state = init_nonamplified_bandinv_state(initial, strategy, key)
  query_values = jnp.array([0.2, 0.4, -0.6, 1.2, -0.8], dtype=jnp.float32)
  actual = []
  for value in query_values:
    state = train_step(state, jnp.array([value], dtype=jnp.float32))
    actual.append(state.params)

  d = _lower_toeplitz(strategy.noising_coef, horizon)
  latent = _latent_sequence(key, horizon=horizon, std=calibration.iid_noise_std)
  expected = initial - learning_rate * jnp.cumsum(query_values + d @ latent)
  np.testing.assert_allclose(jnp.stack(actual), expected, rtol=1e-6, atol=1e-6)


def test_zero_iid_noise_core_is_clipped_fixed_lr_nesterov_sgd(monkeypatch):
  """The zero-noise kernel has the exact clipped fixed-LR Nesterov limit.

  A forged public ``PrivacyCalibration(iid_noise_std=0)`` is intentionally
  rejected by integrity validation, so the M2 sampler is replaced only inside
  this runtime-limit test after a genuine calibration has been validated.
  """
  horizon, beta, learning_rate = 4, 0.5, 0.1
  strategy, calibration, participation = _artifacts(
      horizon=horizon, momentum=beta, learning_rate=learning_rate, noising_coef=(1.0,)
  )
  original_sample = sample_bandinv_noise

  def zero_noise(key, state, noising_coef, iid_noise_std):
    return original_sample(key, state, noising_coef, 0.0)

  monkeypatch.setattr(nonamplified_linear, "sample_bandinv_noise", zero_noise)
  train_step = make_nonamplified_bandinv_train_step(
      _scalar_linear_loss, strategy, calibration, participation, beta, learning_rate
  )
  initial = jnp.array(2.0, dtype=jnp.float32)
  state = init_nonamplified_bandinv_state(initial, strategy, jax.random.key(3))
  query_values = jnp.array([1.0, -0.5, 0.25, 1.5], dtype=jnp.float32)
  actual = []
  for value in query_values:
    state = train_step(state, jnp.array([value], dtype=jnp.float32))
    actual.append(state.params)
  a = learning_rate * jnp.tril(jnp.ones((horizon, horizon))) @ _lower_toeplitz(
      nesterov_kernel_coef(horizon, beta), horizon
  )
  np.testing.assert_allclose(jnp.stack(actual), initial - a @ query_values, rtol=1e-6, atol=1e-6)


def test_bandwidth_one_nested_pytree_and_full_step_jit_match_eager():
  strategy, calibration, participation = _artifacts(
      horizon=3, momentum=0.4, learning_rate=0.05, noising_coef=(1.0,)
  )
  train_step = make_nonamplified_bandinv_train_step(
      _nested_linear_loss, strategy, calibration, participation, 0.4, 0.05
  )
  params = {
      "left": (jnp.array(1.0, dtype=jnp.float32), jnp.array([2.0, -1.0])),
      "right": {"matrix": jnp.arange(4, dtype=jnp.float32).reshape(2, 2)},
  }
  batch = {
      "left": (jnp.array([0.25], dtype=jnp.float32), jnp.array([[1.0, -2.0]])),
      "right": {"matrix": jnp.array([[[0.5, -1.0], [2.0, 0.25]]])},
  }
  state = init_nonamplified_bandinv_state(params, strategy, jax.random.key(13))
  eager = train_step(state, batch)
  compiled = jax.jit(train_step)(state, batch)
  _tree_allclose(compiled, eager)
  assert eager.noise_state.bandwidth == 1
  assert jax.tree_util.tree_structure(eager.params) == jax.tree_util.tree_structure(params)


def test_checkpoint_resume_matches_uninterrupted_state_including_key():
  horizon, beta, learning_rate = 7, 0.7, 0.08
  strategy, calibration, participation = _artifacts(
      horizon=horizon, momentum=beta, learning_rate=learning_rate, noising_coef=(1.0, -0.3)
  )
  train_step = make_nonamplified_bandinv_train_step(
      _scalar_linear_loss, strategy, calibration, participation, beta, learning_rate
  )
  batches = [jnp.array([value], dtype=jnp.float32) for value in jnp.linspace(-1.0, 1.0, horizon)]

  uninterrupted = init_nonamplified_bandinv_state(jnp.array(0.0, dtype=jnp.float32), strategy, jax.random.key(41))
  for batch in batches:
    uninterrupted = train_step(uninterrupted, batch)

  resumed = init_nonamplified_bandinv_state(jnp.array(0.0, dtype=jnp.float32), strategy, jax.random.key(41))
  for batch in batches[:3]:
    resumed = train_step(resumed, batch)
  checkpoint = resumed
  for batch in batches[3:]:
    checkpoint = train_step(checkpoint, batch)

  _tree_allclose(checkpoint, uninterrupted)
  np.testing.assert_array_equal(
      jax.random.key_data(checkpoint.rng_key), jax.random.key_data(uninterrupted.rng_key)
  )


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_sensitivity", 0.0),
        ("mu", 0.0),
        ("noise_multiplier", 0.0),
        ("iid_noise_std", 0.0),
    ],
)
def test_forged_calibration_derived_fields_fail_fast(field, value):
  strategy, calibration, participation = _artifacts()
  forged = replace(calibration, **{field: value})
  with pytest.raises(ValueError, match=field):
    validate_nonamplified_bandinv_setup(
        strategy, forged, participation, momentum=0.6, learning_rate=0.15
    )


def test_mismatched_workload_and_participation_fail_fast():
  strategy, calibration, participation = _artifacts()
  with pytest.raises(ValueError, match="workload_coef"):
    validate_nonamplified_bandinv_setup(
        replace(strategy, workload_coef=jnp.ones(strategy.horizon)),
        calibration,
        participation,
        momentum=0.6,
        learning_rate=0.15,
    )
  with pytest.raises(ValueError, match="participation spec"):
    validate_nonamplified_bandinv_setup(
        strategy,
        calibration,
        ParticipationSpec(strategy.horizon, 2, None),
        momentum=0.6,
        learning_rate=0.15,
    )


@pytest.mark.parametrize(
    "forged_strategy",
    [
        lambda strategy: replace(
            strategy, noising_coef=jnp.array([1.0, 0.2], dtype=jnp.float32)
        ),
        lambda strategy: replace(
            strategy, sensitivity_squared=strategy.sensitivity_squared * 1.1
        ),
    ],
)
def test_forged_strategy_sensitivity_or_noising_coef_fails_fast(forged_strategy):
  strategy, calibration, participation = _artifacts()
  with pytest.raises(
      ValueError,
      match="strategy.sensitivity_squared must match strategy.noising_coef and participation metadata",
  ):
    validate_nonamplified_bandinv_setup(
        forged_strategy(strategy),
        calibration,
        participation,
        momentum=0.6,
        learning_rate=0.15,
    )


def test_strategy_with_jax_privacy_computed_sensitivity_passes_setup_validation():
  strategy, calibration, participation = _artifacts()
  validate_nonamplified_bandinv_setup(
      strategy, calibration, participation, momentum=0.6, learning_rate=0.15
  )


def test_train_step_exposes_only_one_batch_pytree_argument():
  signature = inspect.signature(make_nonamplified_bandinv_train_step)
  assert "batch_argnums" not in signature.parameters
  assert "keep_batch_dim" not in signature.parameters
  strategy, calibration, participation = _artifacts()
  with pytest.raises(TypeError, match="unexpected keyword"):
    make_nonamplified_bandinv_train_step(
        _scalar_linear_loss,
        strategy,
        calibration,
        participation,
        0.6,
        0.15,
        batch_argnums=1,
    )
