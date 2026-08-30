"""Algorithm, pairing, diagnostic, and smoke tests for Experiment 7b."""

import csv
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from exp7.core import init_exp7_train_state, make_exp7_train_step
from exp7b.core import (
    gamma_prime_from_ratio, make_exp7b_train_step, paper_bc_preconditioner,
    phi_infinity,
)
from exp7b.online_shadow import Exp7bWindowCollector
from exp7b.run import run_smoke


def _strategy(horizon=4, coef=(1.0, -0.25)):
  noising = jnp.asarray(coef, jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=noising, min_sep=1, max_participations=1
  )
  return BandInvMFStrategy(
      horizon=horizon, bandwidth=len(coef), min_sep=1, max_participations=1,
      workload_coef=jnp.ones(horizon), noising_coef=noising,
      strategy_coef=toeplitz.inverse_coef(noising, horizon),
      sensitivity_squared=sensitivity, objective=jnp.asarray(0.0),
  )


def _calibration(strategy):
  return calibrate_nonamplified_bandinv(
      epsilon=2.0, delta=1e-5, clip_norm=10.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )


def _loss(params, batch):
  prediction = jnp.dot(params["w"], batch["x"][0])
  return .5 * (prediction - batch["y"][0]) ** 2


def _batch(step=1):
  return {
      "x": jnp.asarray([[1.0, step / 10], [1.0, -step / 10]], jnp.float32),
      "y": jnp.asarray([.5, -.25], jnp.float32),
  }


def _assert_tree_allclose(left, right, **kwargs):
  for a, b in zip(
      jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
  ):
    if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
    else:
      np.testing.assert_allclose(a, b, **kwargs)


def _steps(strategy, calibration, algorithm, key, gamma_prime, horizon=4):
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  state = init_exp7_train_state(params, strategy, key)
  step_fn = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, algorithm=algorithm,
      learning_rate=.02, beta1=.9, beta2=.8, eps=1e-3,
      weight_decay=.01, gamma_prime=gamma_prime,
  ))
  output = []
  for index in range(horizon):
    state = step_fn(state, _batch(index + 1))
    output.append(state)
  return output


def test_gamma_prime_is_ratio_times_phi_infinity():
  strategy = _strategy(coef=(1.0, -.5, .25))
  sigma, ratio = .7, 1.25
  expected_phi = sigma ** 2 * (1.0 + .25 + .0625)
  phi_inf, gamma_prime = gamma_prime_from_ratio(strategy, sigma, ratio)
  assert float(phi_infinity(strategy, sigma)) == pytest.approx(expected_phi)
  assert float(phi_inf) == pytest.approx(expected_phi)
  assert float(gamma_prime) == pytest.approx(ratio * expected_phi)
  with pytest.raises(ValueError):
    gamma_prime_from_ratio(strategy, sigma, 0.0)


def test_correlated_phi_ema_is_identical_to_exp7():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  key = jax.random.key(7)
  old_state = init_exp7_train_state(params, strategy, key)
  new_state = init_exp7_train_state(params, strategy, key)
  old_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, algorithm="bc", learning_rate=.02,
      beta2=.8,
  ))
  _, gamma_prime = gamma_prime_from_ratio(strategy, calibration.iid_noise_std)
  new_step = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, algorithm="bc", learning_rate=.02,
      beta2=.8, gamma_prime=gamma_prime,
  ))
  for index in range(horizon):
    batch = _batch(index + 1)
    old_state, new_state = old_step(old_state, batch), new_step(new_state, batch)
    np.testing.assert_array_equal(old_state.phi_t, new_state.phi_t)
    np.testing.assert_array_equal(old_state.bias_v, new_state.bias_v)


def test_paper_preconditioner_is_exact_and_has_no_outer_adam_epsilon():
  corrected = jnp.asarray([-3.0, 0.25, 4.0], jnp.float32)
  gamma_prime = 1.0
  expected = 1.0 / np.sqrt(np.maximum(np.asarray(corrected), gamma_prime))
  actual = paper_bc_preconditioner(corrected, gamma_prime)
  np.testing.assert_array_equal(actual, expected.astype(np.float32))
  assert "+ eps" not in inspect.getsource(paper_bc_preconditioner)
  assert float(actual[0]) == pytest.approx(1.0 / np.sqrt(gamma_prime))
  assert np.all(np.asarray(actual) <= 1.0 / np.sqrt(gamma_prime))


def test_bc_parameter_update_uses_paper_preconditioner_exactly():
  strategy = _strategy(horizon=1, coef=(1.0,))
  calibration = _calibration(strategy)
  participation = ParticipationSpec(1, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  gamma_prime = .75
  beta1, beta2, lr, wd = .9, .8, .02, .01
  state = init_exp7_train_state(params, strategy, jax.random.key(11))
  step_fn = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, algorithm="bc",
      learning_rate=lr, beta1=beta1, beta2=beta2, eps=.25,
      weight_decay=wd, gamma_prime=gamma_prime,
  ))
  updated = step_fn(state, _batch())
  t = updated.step
  mhat = jax.tree_util.tree_map(lambda x: x / (1 - beta1 ** t), updated.dp_m)
  vhat = jax.tree_util.tree_map(lambda x: x / (1 - beta2 ** t), updated.v11)
  phi_hat = updated.bias_v / (1 - beta2 ** t)
  expected = jax.tree_util.tree_map(
      lambda old, m, v: (1 - lr * wd) * old
      - lr * m / jnp.sqrt(jnp.maximum(v - phi_hat, gamma_prime)),
      params, mhat, vhat,
  )
  _assert_tree_allclose(updated.params, expected, rtol=1e-7, atol=1e-7)


def test_floor_activation_fraction_matches_coordinate_definition():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  _, gamma_prime = gamma_prime_from_ratio(strategy, calibration.iid_noise_std)
  states = _steps(
      strategy, calibration, "bc", jax.random.key(3), float(gamma_prime), horizon
  )
  collector = Exp7bWindowCollector(
      {"w": jnp.asarray([.1, -.2], jnp.float32)}, seed=3, algorithm="bc",
      beta1=.9, beta2=.8, learning_rate=.02, weight_decay=.01, eps=1e-3,
      gamma_prime=float(gamma_prime), window_size=horizon, histogram_bins=64,
  )
  expected_nonpositive, expected_floor = [], []
  for index, state in enumerate(states, 1):
    collector.after_step(state, index)
    vhat = state.v11["w"] / (1 - .8 ** state.step)
    corrected = vhat - state.bias_v / (1 - .8 ** state.step)
    expected_nonpositive.append(float(jnp.mean(corrected <= 0)))
    expected_floor.append(float(jnp.mean(corrected <= gamma_prime)))
  row = collector.finalize()[0]
  assert row["corrected_v_nonpositive_fraction"] == pytest.approx(
      np.mean(expected_nonpositive)
  )
  assert row["floor_activation_fraction"] == pytest.approx(np.mean(expected_floor))
  assert row["p_bc_max"] <= 1.0 / np.sqrt(float(gamma_prime)) * (1 + 1e-6)


def test_baseline_trajectory_is_identical_to_exp7():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  key = jax.random.key(123)
  old_state = init_exp7_train_state(params, strategy, key)
  new_state = init_exp7_train_state(params, strategy, key)
  kwargs = dict(
      algorithm="baseline", learning_rate=.02, beta1=.9, beta2=.999,
      eps=1e-8, weight_decay=.01,
  )
  old_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, **kwargs
  ))
  new_step = jax.jit(make_exp7b_train_step(
      _loss, strategy, calibration, participation, gamma_prime=1.0, **kwargs
  ))
  for index in range(horizon):
    batch = _batch(index + 1)
    old_state, new_state = old_step(old_state, batch), new_step(new_state, batch)
    _assert_tree_allclose(old_state, new_state, rtol=0, atol=0)


def test_paired_branches_share_initialization_batches_and_latent_noise():
  horizon = 4
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  _, gamma_prime = gamma_prime_from_ratio(strategy, calibration.iid_noise_std)
  key = jax.random.key(99)
  baseline = _steps(strategy, calibration, "baseline", key, float(gamma_prime), horizon)
  bc = _steps(strategy, calibration, "bc", key, float(gamma_prime), horizon)
  for baseline_state, bc_state in zip(baseline, bc, strict=True):
    _assert_tree_allclose(
        baseline_state.last_latent_noise, bc_state.last_latent_noise, rtol=0, atol=0
    )
    _assert_tree_allclose(
        baseline_state.last_noise, bc_state.last_noise, rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        jax.random.key_data(baseline_state.rng_key),
        jax.random.key_data(bc_state.rng_key),
    )


def test_small_smoke_writes_all_required_outputs(tmp_path: Path):
  run_smoke(tmp_path, [0], gamma_prime_ratio=1.0)
  required = (
      "window_diagnostics_baseline_seed0.csv",
      "window_diagnostics_bc_seed0.csv",
      "window_summary.csv", "summary.json", "shadow_cancellation.png",
      "bc_preconditioner_floor_activation.png", "update_norm_diagnostics.png",
  )
  for name in required:
    assert (tmp_path / name).is_file(), name
  summary = json.loads((tmp_path / "summary.json").read_text())
  assert summary["smoke"] is True
  assert summary["gamma_prime_ratio"] == 1.0
  assert summary["gamma_prime"] == pytest.approx(summary["phi_infty"])
  assert summary["implied_p_max"] == pytest.approx(1 / np.sqrt(summary["gamma_prime"]))
  assert set(summary["per_seed"]["0"]) == {"baseline", "bc"}
  assert set(summary["per_seed"]["0"]["bc"]) >= {
      "final_test_loss", "final_test_accuracy",
      "early_steps_1_97", "late_steps_98_488",
  }
  with (tmp_path / "window_diagnostics_bc_seed0.csv").open(newline="") as stream:
    row = next(csv.DictReader(stream))
  assert set(row) >= {
      "corrected_v_nonpositive_fraction", "floor_activation_fraction",
      "p_bc_median", "p_bc_q99", "p_bc_q99_9", "p_bc_max",
      "raw_optimizer_update_l2_mean", "applied_parameter_update_l2_mean",
      "parameter_l2_mean",
  }
