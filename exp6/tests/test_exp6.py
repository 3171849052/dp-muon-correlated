"""Unit tests for Experiment 6's local shadow diagnostics."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from exp3.online_shadow import init_online_shadow_state, make_online_shadow_train_step
from exp6.diagnostics import (
    cancellation_score,
    correlation_from_rows,
    stage_summary,
    window_ranges,
)
from exp6.online_shadow import WindowDiagnosticsCollector


def _simple_strategy(horizon=4):
  noising = jnp.asarray([1.0, -0.2], jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=noising, min_sep=1, max_participations=1
  )
  return BandInvMFStrategy(
      horizon=horizon,
      bandwidth=2,
      min_sep=1,
      max_participations=1,
      workload_coef=jnp.ones((horizon,), jnp.float32),
      noising_coef=noising,
      strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=sensitivity,
      objective=jnp.asarray(0.0, jnp.float32),
  )


def _fake_moment_state(step, *, beta1=.9, beta2=.9, noise_value=1.0):
  """Moments for a constant clean gradient and a prescribed linear noise."""
  clean_gradient = 2.0
  clean_m = (1.0 - beta1 ** step) / (1.0 - beta1) * clean_gradient
  clean_v = (1.0 - beta2 ** step) / (1.0 - beta2) * clean_gradient ** 2
  noise_m = (1.0 - beta1 ** step) / (1.0 - beta1) * noise_value
  tree = lambda value: {"w": jnp.asarray([value], jnp.float32)}
  return SimpleNamespace(
      clean_m=tree(clean_m),
      clean_v=tree(clean_v),
      dp_m=tree(clean_m),
      dp_v=tree(clean_v),
      noise_m=tree(noise_m),
  )


def test_fixed_p_makes_dynamic_and_frozen_scores_equal():
  collector = WindowDiagnosticsCollector(
      {"w": jnp.zeros((1,), jnp.float32)},
      seed=0,
      beta1=.9,
      beta2=.9,
      learning_rate=.1,
      weight_decay=.01,
      eps=1e-6,
  )
  for step in range(1, 17):
    collector.after_step(_fake_moment_state(step, noise_value=(-1.0) ** step), step)
  row = collector.rows[0]
  assert row["start_step"] == 1 and row["end_step"] == 16
  assert row["C_dynamic_clean_p"] == pytest.approx(row["C_frozen_p"], abs=1e-7)
  assert row["delta_p_cancellation"] == pytest.approx(0.0, abs=1e-7)


def test_two_step_cancellation_is_broken_by_changing_p():
  noise = [np.asarray([1.0]), np.asarray([-1.0])]
  same_p = [value * p for value, p in zip(noise, [1.0, 1.0], strict=True)]
  changing_p = [value * p for value, p in zip(noise, [1.0, 2.0], strict=True)]
  kept = cancellation_score(same_p, learning_rate=1.0, weight_decay=0.0)
  broken = cancellation_score(changing_p, learning_rate=1.0, weight_decay=0.0)
  assert kept == pytest.approx(0.0)
  assert broken > kept


def test_window_boundaries_reset_only_local_diagnostic_accumulators():
  params = {"w": jnp.zeros((1,), jnp.float32)}
  collector = WindowDiagnosticsCollector(
      params, seed=3, beta1=.9, beta2=.9, learning_rate=.1,
      weight_decay=.01, eps=1e-6, window_size=2,
  )
  for step in range(1, 3):
    collector.after_step(_fake_moment_state(step), step)
  state = collector.state
  assert int(state.count) == 0
  assert int(state.window_index) == 1
  assert float(state.denominator_dynamic_clean_p) == 0.0
  np.testing.assert_array_equal(state.weighted_dynamic_clean_p["w"], np.zeros(1))
  np.testing.assert_array_equal(state.frozen_p["w"], np.zeros(1))
  # p is retained solely to define the next step's relative change; the
  # window-local path and denominator are what get cleared.
  assert bool(state.has_previous_p)
  assert len(collector.rows) == 1


def test_diagnostic_callback_does_not_change_exp3_training_trajectory():
  strategy = _simple_strategy(4)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2.0, delta=1e-5, clip_norm=10.0, normalize_by=1.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(4, 1, 1)
  def loss(params, batch):
    return jnp.sum((params["w"] - batch["target"]) ** 2)

  online_step, online_optimizer = make_online_shadow_train_step(
      loss, strategy, calibration, participation, learning_rate=.01, weight_decay=.01,
  )
  standard_step, standard_optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      loss, strategy, calibration, participation, learning_rate=.01, weight_decay=.01,
  )
  params = {"w": jnp.asarray([1.0, -1.0], jnp.float32)}
  key = jax.random.key(7)
  online = init_online_shadow_state(params, strategy, key, online_optimizer)
  standard = init_nonamplified_bandinv_dpadamw_state(params, strategy, key, standard_optimizer)
  collector = WindowDiagnosticsCollector(
      params, seed=0, beta1=.9, beta2=.999, learning_rate=.01,
      weight_decay=.01, eps=1e-8, window_size=2,
  )
  batch = {"target": jnp.asarray([.25, -.5], jnp.float32)}
  online_compiled, standard_compiled = jax.jit(online_step), jax.jit(standard_step)
  for step in range(1, 5):
    online = online_compiled(online, batch)
    standard = standard_compiled(standard, batch)
    before = online
    collector.after_step(online, step)
    # The callback only reads the state.  Compare the actual state after the
    # callback against the object returned by the genuine update.
    for left, right in zip(jax.tree_util.tree_leaves(online), jax.tree_util.tree_leaves(before), strict=True):
      if jnp.issubdtype(jnp.asarray(left).dtype, jax.dtypes.prng_key):
        np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
      else:
        np.testing.assert_allclose(left, right)
    for left, right in zip(jax.tree_util.tree_leaves(online.params), jax.tree_util.tree_leaves(standard.params), strict=True):
      np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip(jax.tree_util.tree_leaves(online.optimizer_state), jax.tree_util.tree_leaves(standard.optimizer_state), strict=True):
      np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip(jax.tree_util.tree_leaves(online.noise_state), jax.tree_util.tree_leaves(standard.noise_state), strict=True):
      np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(jax.random.key_data(online.rng_key), jax.random.key_data(standard.rng_key))
  assert len(collector.finalize()) == 2


def test_window_ranges_keep_short_tail_and_stage_weighting_splits_boundary():
  assert window_ranges(488) == [(i, i * 16 + 1, min((i + 1) * 16, 488)) for i in range(31)]
  rows = [
      {"seed": 0, "window_index": 0, "start_step": 90, "end_step": 100, "delta_p_cancellation": 1.0,
       "mean_p_relative_change": 0.1},
      {"seed": 0, "window_index": 1, "start_step": 101, "end_step": 112, "delta_p_cancellation": 3.0,
       "mean_p_relative_change": 0.2},
  ]
  summary = stage_summary(rows, total_steps=112)
  assert summary["early_steps_1_97"]["covered_steps"] == 8
  assert summary["early_steps_1_97"]["mean_delta_p_cancellation"] == pytest.approx(1.0)
  assert summary["late_steps_98_488"]["covered_steps"] == 15
  assert summary["late_steps_98_488"]["mean_delta_p_cancellation"] == pytest.approx(2.6)


def test_spearman_correlation_handles_rank_order():
  rows = [
      {"mean_p_relative_change": value, "delta_p_cancellation": value}
      for value in (0.1, 0.4, 0.2, 0.8)
  ]
  assert correlation_from_rows(rows) == pytest.approx(1.0)
