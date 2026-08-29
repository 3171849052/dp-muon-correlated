"""Behavioral tests for segmented correlated DP-AdamW."""

from dataclasses import replace
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax_privacy.matrix_factorization import toeplitz
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from dp_muon.training.nonamplified_segmented_bandinv_dpadamw import (
    SegmentedBandInvDPAdamWState,
    SegmentedPlan,
    begin_segment,
    fit_segmented_plan,
    global_segmented_sensitivity_squared,
    init_segmented_bandinv_dpadamw_state,
    make_segmented_bandinv_dpadamw_train_step,
)
from exp4.segmented_strategy import (
    global_segmented_sensitivity_squared as exp4_global_segmented_sensitivity_squared,
)


def _strategy(n: int, *, min_sep: int, max_participations: int | None = 2):
  noising = jnp.asarray([1.0, -0.25][: min(2, n)], jnp.float32)
  strategy_coef = toeplitz.inverse_coef(noising, n)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=n,
      noising_coef=noising,
      min_sep=min_sep,
      max_participations=max_participations,
  )
  return BandInvMFStrategy(
      horizon=n,
      bandwidth=len(noising),
      min_sep=min_sep,
      max_participations=max_participations,
      workload_coef=jnp.ones(n),
      noising_coef=noising,
      strategy_coef=strategy_coef,
      sensitivity_squared=sensitivity,
      objective=jnp.array(1.0),
  )


def _plan(lengths=(2, 2, 1), *, epsilon=2.0):
  strategies = tuple(_strategy(length, min_sep=length) for length in lengths)
  sensitivity = global_segmented_sensitivity_squared(
      strategies, min_sep=2, max_participations=3
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=epsilon,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=1.0,
      adjacency="add_remove",
      sensitivity_squared=sensitivity,
  )
  return SegmentedPlan(
      "seg2",
      lengths,
      strategies,
      sensitivity,
      calibration,
      global_min_sep=2,
      max_participations=3,
      runtime_bandwidth=2,
  )


def _loss(params, batch):
  return 0.5 * (params["x"] * batch["x"][0]) ** 2


def _tree_equal(left, right):
  assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
  return all(
      bool(jnp.array_equal(a, b))
      for a, b in zip(
          jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
      )
  )


def _count(state):
  # optax.ScaleByAdamState.count is the only integer scalar in AdamW state.
  for leaf in jax.tree_util.tree_leaves(state.optimizer_state):
    array = jnp.asarray(leaf)
    if array.shape == () and jnp.issubdtype(array.dtype, jnp.integer):
      return int(array)
  raise AssertionError("AdamW count was not found")


def test_boundary_only_resets_noise_and_preserves_adamw_state():
  plan = _plan()
  step, optimizer = make_segmented_bandinv_dpadamw_train_step(
      _loss, plan, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
      weight_decay=0.01,
  )
  state = init_segmented_bandinv_dpadamw_state(
      {"x": jnp.array(1.0)}, plan, jax.random.key(7), optimizer
  )
  state = step(state, {"x": jnp.array([1.0])})
  state = step(state, {"x": jnp.array([1.0])})
  before = begin_segment(state, 2, plan)
  assert _tree_equal(before.params, state.params)
  assert _tree_equal(before.optimizer_state, state.optimizer_state)
  assert int(before.step) == int(state.step) == 2
  assert int(before.noise_state.step) == 0
  assert all(bool(jnp.all(leaf == 0)) for leaf in jax.tree_util.tree_leaves(before.noise_state.buffer))
  assert _count(before) == 2
  after = step(before, {"x": jnp.array([1.0])})
  assert _count(after) == 3
  assert int(after.noise_state.step) == 1
  assert int(after.segment_index) == 1


def test_segment_rng_streams_are_independent_and_reproducible():
  plan = _plan()
  optimizer = optax.adamw(0.01)
  initial = init_segmented_bandinv_dpadamw_state(
      {"x": jnp.array(1.0)}, plan, jax.random.key(12), optimizer
  )
  first = begin_segment(initial, 2, plan)
  second = begin_segment(initial, 2, plan)
  np.testing.assert_array_equal(jax.random.key_data(first.rng_key), jax.random.key_data(second.rng_key))
  np.testing.assert_array_equal(
      jax.random.key_data(first.rng_key), jax.random.key_data(jax.random.fold_in(initial.rng_root_key, 1))
  )
  assert not np.array_equal(
      np.asarray(jax.random.key_data(initial.rng_key)),
      np.asarray(jax.random.key_data(first.rng_key)),
  )


def test_workload_is_exactly_decayed_prefix_and_has_no_optimizer_inputs():
  calls = []

  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    calls.append((horizon, bandwidth, min_sep, set(kwargs)))
    return replace(
        _strategy(horizon, min_sep=min_sep, max_participations=3),
        workload_coef=jnp.asarray(kwargs["workload_coef"]),
    )

  plan = fit_segmented_plan(
      horizon=5,
      segment_length=2,
      bandwidth=4,
      min_sep=2,
      max_participations=3,
      max_optimizer_steps=1,
      reduction="mean",
      learning_rate=0.2,
      weight_decay=0.1,
      epsilon=2.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=1.0,
      adjacency="add_remove",
      fit_strategy=fake_fit,
  )
  assert [(call[0], call[1], call[2]) for call in calls] == [(1, 1, 1), (2, 2, 2)]
  assert all(call[3] == {"max_participations", "max_optimizer_steps", "reduction", "workload_coef"} for call in calls)
  for strategy in plan.strategies:
    np.testing.assert_array_equal(
        strategy.workload_coef,
        decayed_prefix_sum_workload_coef(strategy.horizon, 0.2, 0.1),
    )


def test_global_sensitivity_matches_block_diagonal_bruteforce():
  strategies = (_strategy(2, min_sep=2), _strategy(2, min_sep=2))
  actual = global_segmented_sensitivity_squared(
      strategies, min_sep=2, max_participations=2
  )
  matrices = []
  for strategy in strategies:
    coef = np.asarray(strategy.strategy_coef)
    n = strategy.horizon
    rows, columns = np.indices((n, n))
    matrices.append(np.where(rows >= columns, coef[rows - columns], 0.0))
  dense = np.zeros((4, 4))
  dense[:2, :2] = matrices[0]
  dense[2:, 2:] = matrices[1]
  expected = 0.0
  for size in range(3):
    for chosen in combinations(range(4), size):
      if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
        vector = np.zeros(4)
        vector[list(chosen)] = 1.0
        expected = max(expected, float(np.sum((dense @ vector) ** 2)))
  assert actual == expected
  assert actual == exp4_global_segmented_sensitivity_squared(
      strategies, min_sep=2, max_participations=2
  )


def test_final_calibration_uses_one_global_target():
  plan = _plan(epsilon=2.5)
  assert plan.calibration.epsilon == 2.5
  recovered_epsilon = float(gdp.eps_from_mu(mu=plan.calibration.mu, delta=plan.calibration.delta))
  assert recovered_epsilon == pytest.approx(plan.calibration.epsilon, rel=1e-10, abs=1e-10)


def test_segmented_state_checkpoint_resume_matches_uninterrupted(tmp_path):
  plan = _plan()
  step, optimizer = make_segmented_bandinv_dpadamw_train_step(
      _loss, plan, learning_rate=0.01, weight_decay=0.01
  )
  initial = init_segmented_bandinv_dpadamw_state(
      {"x": jnp.array(1.0)}, plan, jax.random.key(22), optimizer
  )
  batches = [{"x": jnp.array([value])} for value in (1.0, 2.0, -1.0, 0.5, 3.0)]
  uninterrupted = initial
  for batch in batches:
    uninterrupted = step(uninterrupted, batch)
  resumed = initial
  for batch in batches[:3]:
    resumed = step(resumed, batch)
  checkpoint = tmp_path / "segmented.pkl"
  save_checkpoint(
      checkpoint,
      state=resumed,
      current_step=3,
      experiment_config={"segmented": True},
      artifact_identifiers={"algorithm": "dp-adamw-correlated-segmented"},
  )
  resumed = load_checkpoint(checkpoint)["state"]
  for batch in batches[3:]:
    resumed = step(resumed, batch)
  assert _tree_equal(resumed.params, uninterrupted.params)
  assert _tree_equal(resumed.optimizer_state, uninterrupted.optimizer_state)
  assert _tree_equal(resumed.noise_state, uninterrupted.noise_state)
  assert _tree_equal(resumed.rng_key, uninterrupted.rng_key)


def test_jit_step_switches_segment_without_changing_global_counter():
  plan = _plan()
  step, optimizer = make_segmented_bandinv_dpadamw_train_step(
      _loss, plan, learning_rate=0.01, weight_decay=0.01
  )
  state = init_segmented_bandinv_dpadamw_state(
      {"x": jnp.array(1.0)}, plan, jax.random.key(27), optimizer
  )
  compiled_step = jax.jit(step)
  for index in range(plan.horizon):
    state = compiled_step(state, {"x": jnp.array([1.0 + index])})
  assert int(state.step) == plan.horizon
  assert int(state.segment_index) == len(plan.block_lengths) - 1
  assert int(state.noise_state.step) == plan.block_lengths[-1]


def test_segment_length_horizon_matches_continuous_naive():
  strategy = _strategy(2, min_sep=2, max_participations=2)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2.0, delta=1e-5, clip_norm=1.0, normalize_by=1.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )
  plan = SegmentedPlan("seg2", (2,), (strategy,), float(strategy.sensitivity_squared), calibration,
                       global_min_sep=2, max_participations=2, runtime_bandwidth=strategy.bandwidth)
  segmented_step, segmented_optimizer = make_segmented_bandinv_dpadamw_train_step(
      _loss, plan, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
  )
  naive_step, naive_optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      _loss, strategy, calibration,
      # The naive implementation has the same participation contract.
      ParticipationSpec(2, 2, 2),
      learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
  )
  params = {"x": jnp.array(1.0)}
  segmented = init_segmented_bandinv_dpadamw_state(params, plan, jax.random.key(31), segmented_optimizer)
  naive = init_nonamplified_bandinv_dpadamw_state(params, strategy, jax.random.key(31), naive_optimizer)
  for value in (1.0, -0.5):
    batch = {"x": jnp.array([value])}
    segmented = segmented_step(segmented, batch)
    naive = naive_step(naive, batch)
  np.testing.assert_array_equal(segmented.params["x"], naive.params["x"])
