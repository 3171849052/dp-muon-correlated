"""Unit and smoke tests for Experiment 7."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import (
    ParticipationSpec, calibrate_nonamplified_bandinv, make_clipped_gradient_query,
)
from exp7.core import (
    bandinv_marginal_variances, init_exp7_train_state, make_exp7_train_step,
    shadow_second_moment_inputs, update_bias_ema,
)
from exp7.diagnostics import factorial_effects, restoration_ratio
from exp7.run import run_smoke
from exp3.online_shadow import init_online_shadow_state, make_online_shadow_train_step
from exp6.online_shadow import WindowDiagnosticsCollector
from exp7.online_shadow import Exp7WindowCollector


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
  for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
    np.testing.assert_allclose(a, b, **kwargs)


def test_four_v_inputs_have_exact_factorial_identity():
  g = {"a": jnp.asarray([1.0, -2.0]), "b": jnp.asarray([[.5]])}
  noise = {"a": jnp.asarray([.25, 3.0]), "b": jnp.asarray([[-1.5]])}
  values = shadow_second_moment_inputs(g, noise)
  reconstructed = jax.tree_util.tree_map(
      lambda v10, v01, v00: v10 + v01 - v00,
      values["10"], values["01"], values["00"],
  )
  _assert_tree_allclose(reconstructed, values["11"], rtol=1e-6, atol=1e-6)


def test_v11_is_ema_of_actual_private_gradient_squared():
  strategy = _strategy(horizon=3)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(3, 1, 1)
  step_fn = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, algorithm="baseline",
      learning_rate=.01, beta2=.8,
  ))
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  state = init_exp7_train_state(params, strategy, jax.random.key(3))
  query = make_clipped_gradient_query(
      _loss, clip_norm=calibration.clip_norm, normalize_by=calibration.normalize_by,
      batch_argnums=1, keep_batch_dim=True,
  )
  expected = jax.tree_util.tree_map(jnp.zeros_like, params)
  for index in range(3):
    batch = _batch(index + 1)
    g = query(state.params, batch)
    state = step_fn(state, batch)
    private = jax.tree_util.tree_map(lambda x, z: x + z, g, state.last_noise)
    expected = jax.tree_util.tree_map(
        lambda old, value: .8 * old + .2 * value * value, expected, private
    )
    _assert_tree_allclose(state.v11, expected, rtol=2e-6, atol=2e-6)


def test_factorial_effect_formulas():
  effects = factorial_effects(1.0, 4.0, 6.0, 12.0)
  assert effects["E_cross"] == pytest.approx(.5 * ((4 - 1) + (12 - 6)))
  assert effects["E_square"] == pytest.approx(.5 * ((6 - 1) + (12 - 4)))
  assert effects["interaction"] == pytest.approx(12 - 4 - 6 + 1)


def test_bandinv_phi_matches_dense_operator_covariance_diagonal():
  strategy = _strategy(horizon=6, coef=(1.0, -.4, .2))
  sigma = .7
  rows = np.arange(strategy.horizon)[:, None]
  columns = np.arange(strategy.horizon)[None, :]
  offsets = rows - columns
  coef = np.asarray(strategy.noising_coef)
  dense = np.where(
      (offsets >= 0) & (offsets < len(coef)), coef[np.clip(offsets, 0, len(coef) - 1)], 0.0
  )
  covariance = sigma ** 2 * dense @ dense.T
  np.testing.assert_allclose(
      bandinv_marginal_variances(strategy, sigma), np.diag(covariance), rtol=1e-6, atol=1e-7
  )


def test_bc_bias_ema_and_debiasing_for_time_varying_phi():
  beta2, bias = .8, jnp.asarray(0.0)
  phis = [1.0, 2.5, .25, 4.0]
  weighted = 0.0
  for step, phi in enumerate(phis, 1):
    bias, debiased = update_bias_ema(bias, phi, beta2, step)
    weighted = beta2 * weighted + (1 - beta2) * phi
    assert float(bias) == pytest.approx(weighted)
    assert float(debiased) == pytest.approx(weighted / (1 - beta2 ** step))


def test_restoration_ratio_and_near_zero_denominator():
  assert restoration_ratio(2.0, 6.0, 2.0) == pytest.approx(1.0)
  assert restoration_ratio(2.0, 6.0, 6.0) == pytest.approx(0.0)
  assert restoration_ratio(2.0, 6.0, 4.0) == pytest.approx(.5)
  assert np.isnan(restoration_ratio(2.0, 2.0, 2.0))
  assert np.isnan(restoration_ratio(2.0, 2.0 + 1e-14, 2.0))


def test_paired_algorithms_use_identical_latent_and_correlated_noise():
  strategy = _strategy(horizon=4)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(4, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  key = jax.random.key(99)
  baseline = init_exp7_train_state(params, strategy, key)
  bc = init_exp7_train_state(params, strategy, key)
  baseline_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, algorithm="baseline", learning_rate=.02
  ))
  bc_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, algorithm="bc", learning_rate=.02
  ))
  for index in range(4):
    batch = _batch(index + 1)
    baseline, bc = baseline_step(baseline, batch), bc_step(bc, batch)
    _assert_tree_allclose(baseline.last_latent_noise, bc.last_latent_noise, rtol=0, atol=0)
    _assert_tree_allclose(baseline.last_noise, bc.last_noise, rtol=0, atol=0)
    np.testing.assert_array_equal(
        jax.random.key_data(baseline.rng_key), jax.random.key_data(bc.rng_key)
    )


def test_baseline_matches_exp3_adamw_and_exp6_named_paths():
  horizon = 16
  strategy = _strategy(horizon=horizon)
  calibration = _calibration(strategy)
  participation = ParticipationSpec(horizon, 1, 1)
  params = {"w": jnp.asarray([.1, -.2], jnp.float32)}
  key = jax.random.key(123)
  exp7_state = init_exp7_train_state(params, strategy, key)
  exp7_step = jax.jit(make_exp7_train_step(
      _loss, strategy, calibration, participation, algorithm="baseline",
      learning_rate=.02, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01,
  ))
  exp3_step, optimizer = make_online_shadow_train_step(
      _loss, strategy, calibration, participation, learning_rate=.02,
      beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01,
  )
  exp3_state = init_online_shadow_state(params, strategy, key, optimizer)
  exp3_step = jax.jit(exp3_step)
  collector7 = Exp7WindowCollector(
      params, seed=0, algorithm="baseline", beta1=.9, beta2=.999,
      learning_rate=.02, weight_decay=.01, eps=1e-8,
  )
  collector6 = WindowDiagnosticsCollector(
      params, seed=0, beta1=.9, beta2=.999, learning_rate=.02,
      weight_decay=.01, eps=1e-8,
  )
  for index in range(horizon):
    batch = _batch(index + 1)
    exp7_state, exp3_state = exp7_step(exp7_state, batch), exp3_step(exp3_state, batch)
    _assert_tree_allclose(exp7_state.params, exp3_state.params, rtol=2e-6, atol=2e-7)
    collector7.after_step(exp7_state, index + 1)
    collector6.after_step(exp7_state, index + 1)
  row7, row6 = collector7.finalize()[0], collector6.finalize()[0]
  assert row7["C_00"] == pytest.approx(row6["C_dynamic_clean_p"], rel=2e-6, abs=2e-7)
  assert row7["C_11"] == pytest.approx(row6["C_real_adamw"], rel=2e-6, abs=2e-7)


def test_small_smoke_path_writes_required_outputs(tmp_path: Path):
  run_smoke(tmp_path, [0])
  assert (tmp_path / "window_diagnostics_baseline_seed0.csv").is_file()
  assert (tmp_path / "window_diagnostics_bc_seed0.csv").is_file()
  assert (tmp_path / "window_summary.csv").is_file()
  summary = json.loads((tmp_path / "summary.json").read_text())
  assert summary["smoke"] is True
  assert set(summary["per_seed"]["0"]) >= {"baseline", "bc", "cancellation"}
