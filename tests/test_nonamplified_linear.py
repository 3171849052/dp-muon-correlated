"""Tests for the end-to-end non-amplified linear BandInvMF trainer."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dp_muon.bandinvmf import (
    BandInvMFStrategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.privacy import ParticipationSpec, PrivacyCalibration
from dp_muon.training import (
    init_nonamplified_bandinv_state,
    make_nonamplified_bandinv_train_step,
    validate_nonamplified_bandinv_setup,
)


def _loss(params, batch):
  return params * batch[0]


def _artifacts(*, horizon=3, momentum=0.5, learning_rate=0.2):
  strategy = BandInvMFStrategy(
      horizon=horizon,
      bandwidth=1,
      min_sep=1,
      max_participations=None,
      workload_coef=fixed_lr_nesterov_trajectory_workload_coef(
          horizon, momentum, learning_rate
      ),
      noising_coef=jnp.array([1.0], dtype=jnp.float32),
      strategy_coef=jnp.array([1.0], dtype=jnp.float32),
      sensitivity_squared=jnp.array(1.0, dtype=jnp.float32),
      objective=jnp.array(0.0, dtype=jnp.float32),
  )
  calibration = PrivacyCalibration(
      epsilon=1.0,
      delta=1e-5,
      adjacency="add_remove",
      clip_norm=2.0,
      normalize_by=4.0,
      query_sensitivity=0.5,
      matrix_sensitivity=1.0,
      total_sensitivity=0.5,
      mu=1.0,
      noise_multiplier=0.0,
      iid_noise_std=0.0,
  )
  return strategy, calibration, ParticipationSpec(horizon, 1, None)


def test_step_applies_noisy_nesterov_update_and_is_jittable():
  strategy, calibration, participation = _artifacts()
  calibration = replace(calibration, iid_noise_std=0.3)
  train_step = make_nonamplified_bandinv_train_step(
      _loss, strategy, calibration, participation, momentum=0.5, learning_rate=0.2
  )
  state = init_nonamplified_bandinv_state(jnp.array(1.0, dtype=jnp.float32), strategy, jax.random.key(1))
  batch = jnp.array([2.0, 4.0], dtype=jnp.float32)

  eager = train_step(state, batch)
  compiled = jax.jit(train_step)(state, batch)

  expected_noise, expected_noise_state, expected_key = sample_bandinv_noise(
      jax.random.key(1),
      init_bandinv_noise_state(jnp.array(1.0, dtype=jnp.float32), bandwidth=1),
      strategy.noising_coef,
      calibration.iid_noise_std,
  )
  # Each example clips to 2, hence q = (2 + 2) / 4 = 1.  At t=0,
  # U_0=(1-beta**2) * (q + R_0), with R_0 sampled by the M2 filter.
  expected_params = 1.0 - 0.2 * 0.75 * (1.0 + expected_noise)
  np.testing.assert_allclose(eager.params, expected_params)
  np.testing.assert_allclose(compiled.params, eager.params)
  np.testing.assert_array_equal(eager.nesterov_state.step, 1)
  np.testing.assert_array_equal(eager.noise_state.step, 1)
  np.testing.assert_array_equal(compiled.nesterov_state.step, compiled.noise_state.step)
  np.testing.assert_allclose(eager.noise_state.buffer, expected_noise_state.buffer)
  np.testing.assert_array_equal(
      jax.random.key_data(eager.rng_key), jax.random.key_data(expected_key)
  )
  assert eager.noise_state.bandwidth == len(strategy.noising_coef)


@pytest.mark.parametrize(
    "strategy_change,calibration_change,participation_change,message",
    [
        (
            lambda strategy: replace(strategy, workload_coef=jnp.ones(3)),
            lambda calibration: calibration,
            lambda participation: participation,
            "workload_coef",
        ),
        (
            lambda strategy: replace(strategy, sensitivity_squared=jnp.array(4.0)),
            lambda calibration: calibration,
            lambda participation: participation,
            "matrix_sensitivity",
        ),
        (
            lambda strategy: strategy,
            lambda calibration: replace(calibration, query_sensitivity=0.7),
            lambda participation: participation,
            "query_sensitivity",
        ),
        (
            lambda strategy: strategy,
            lambda calibration: calibration,
            lambda participation: ParticipationSpec(3, 2, None),
            "participation spec",
        ),
    ],
)
def test_setup_rejects_mismatched_fitted_artifacts(
    strategy_change, calibration_change, participation_change, message
):
  strategy, calibration, participation = _artifacts()
  with pytest.raises(ValueError, match=message):
    validate_nonamplified_bandinv_setup(
        strategy_change(strategy),
        calibration_change(calibration),
        participation_change(participation),
        momentum=0.5,
        learning_rate=0.2,
    )


def test_misaligned_checkpoint_steps_fail_before_updating():
  strategy, calibration, participation = _artifacts()
  train_step = make_nonamplified_bandinv_train_step(
      _loss, strategy, calibration, participation, momentum=0.5, learning_rate=0.2
  )
  state = init_nonamplified_bandinv_state(jnp.array(1.0, dtype=jnp.float32), strategy, jax.random.key(1))
  state = replace(state, noise_state=replace(state.noise_state, step=jnp.array(1, dtype=jnp.int32)))
  with pytest.raises(ValueError, match="nesterov_state.step"):
    train_step(state, jnp.array([2.0], dtype=jnp.float32))
