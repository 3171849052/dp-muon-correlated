from dataclasses import replace
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.bandinvmf import BandInvMFStrategy, init_bandinv_noise_state
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.training.nonamplified_bandinv_dpadamw import NonAmplifiedBandInvDPAdamWState
from exp4.diagnostics import compute_p_tree, p_tree_statistics
from exp4.run import parse_args, run_smoke
from exp4.segmented_strategy import (
    SegmentedPlan, begin_segment, block_lengths, fit_segmented_plan,
    global_segmented_sensitivity_squared,
)


def _strategy(coef):
  coef = jnp.asarray(coef, jnp.float32)
  return BandInvMFStrategy(
      horizon=len(coef), bandwidth=1, min_sep=len(coef), max_participations=2,
      workload_coef=jnp.ones(len(coef)), noising_coef=jnp.ones(1),
      strategy_coef=coef, sensitivity_squared=jnp.array(1.), objective=jnp.array(1.))


def test_compute_p_tree_matches_manual_formula():
  optimizer = optax.adamw(.1, b2=.9, eps=1e-6)
  params = {"a": jnp.ones(2)}
  state = optimizer.init(params)
  _, state = optimizer.update({"a": jnp.array([2., 4.])}, state, params)
  actual = compute_p_tree(state, beta2=.9, eps=1e-6)
  np.testing.assert_allclose(actual["a"], 1. / (jnp.array([2., 4.]) + 1e-6))


def test_pytree_statistics_and_relative_change():
  tree = {"a": jnp.array([1., 2.]), "b": (jnp.array([3., 4.]),)}
  previous = jax.tree_util.tree_map(lambda value: value / 2, tree)
  row, _ = p_tree_statistics(tree, previous, step=2)
  values = np.arange(1., 5.)
  assert row.p_mean == np.mean(values)
  assert row.p_median == np.median(values)
  assert row.p_p25 == np.percentile(values, 25)
  assert row.p_rms == np.sqrt(np.mean(values ** 2))
  assert np.isclose(row.relative_change, 1.0)


def test_reading_diagnostics_does_not_change_adamw_result():
  optimizer = optax.adamw(.01)
  params = {"x": jnp.array([1., 2.])}; gradient = {"x": jnp.array([.2, -.3])}
  state = optimizer.init(params)
  updates, updated = optimizer.update(gradient, state, params)
  expected = optax.apply_updates(params, updates)
  compute_p_tree(updated, beta2=.999, eps=1e-8)
  updates2, updated2 = optimizer.update(gradient, state, params)
  actual = optax.apply_updates(params, updates2)
  np.testing.assert_array_equal(actual["x"], expected["x"])
  assert jax.tree_util.tree_all(jax.tree_util.tree_map(
      lambda a, b: jnp.array_equal(a, b), updated, updated2))


def test_block_lengths_required_cases_and_short_tail():
  assert block_lengths(488, 97) == (97, 97, 97, 97, 97, 3)
  assert block_lengths(488, 16) == (16,) * 30 + (8,)


def test_boundary_resets_only_noise_state():
  params = {"x": jnp.ones(2)}; optimizer = optax.adamw(.1)
  optimizer_state = optimizer.init(params)
  noise = init_bandinv_noise_state(params, 1)
  state = NonAmplifiedBandInvDPAdamWState(params, optimizer_state, noise, jax.random.key(1), jnp.array(2))
  strategies = (_strategy([1., .5]), _strategy([1., .5]))
  plan = SegmentedPlan("seg2", (2, 2), strategies, 1., None)  # calibration unused
  result = begin_segment(state, 2, plan)
  assert result.params is state.params
  assert result.optimizer_state is state.optimizer_state
  assert result.rng_key is state.rng_key
  assert int(result.step) == 2 and int(result.noise_state.step) == 0


def test_segment_fit_uses_decayed_prefix_workload():
  calls = []
  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    calls.append((horizon, np.asarray(kwargs["workload_coef"])))
    return _strategy(np.ones(horizon))
  fit_segmented_plan(
      horizon=5, block_size=3, bandwidth=1, min_sep=3, max_participations=2,
      max_optimizer_steps=1, reduction="mean", learning_rate=.2, weight_decay=.1,
      epsilon=3., delta=1e-5, clip_norm=1., normalize_by=1.,
      adjacency="add_remove", fit_strategy=fake_fit)
  for length, workload in calls:
    np.testing.assert_allclose(workload, decayed_prefix_sum_workload_coef(length, .2, .1))


def test_global_segmented_sensitivity_matches_dense_oracle():
  strategies = (_strategy([1., .5, .25]), _strategy([1., .4]))
  min_sep, cap = 3, 2
  actual = global_segmented_sensitivity_squared(
      strategies, min_sep=min_sep, max_participations=cap)
  blocks=[]
  for strategy in strategies:
    coef=np.asarray(strategy.strategy_coef); n=len(coef)
    blocks.append(np.fromfunction(lambda i,j: np.where(i>=j, coef[(i-j).astype(int)],0.),(n,n)))
  dense=np.zeros((5,5)); dense[:3,:3]=blocks[0]; dense[3:,3:]=blocks[1]
  expected=0.
  for size in range(cap+1):
    for chosen in combinations(range(5),size):
      if all(b-a>=min_sep for a,b in zip(chosen,chosen[1:])):
        vector=np.zeros(5); vector[list(chosen)]=1
        expected=max(expected,float(np.sum((dense@vector)**2)))
  assert np.isclose(actual,expected)


def test_segmented_calibration_uses_global_target():
  plan = fit_segmented_plan(
      horizon=5, block_size=3, bandwidth=1, min_sep=3, max_participations=2,
      max_optimizer_steps=1, reduction="mean", learning_rate=.1, weight_decay=.01,
      epsilon=2., delta=1e-5, clip_norm=1., normalize_by=2., adjacency="add_remove")
  assert plan.calibration.epsilon == 2. and plan.calibration.delta == 1e-5
  assert np.isclose(plan.calibration.matrix_sensitivity ** 2, plan.sensitivity_squared)


def test_cli_seeds_and_real_smoke_outputs(tmp_path):
  args = parse_args(["--seeds", "0", "1", "2"])
  assert args.seeds == [0, 1, 2]
  run_smoke(tmp_path, [0])
  required = ["comparison.csv", "summary.json", "comparison_final_accuracy.png",
              "diagnostics_continuous_seed0.csv", "p_over_steps_seed0.png",
              "p_relative_change_seed0.png", "p_median_summary.png"]
  assert all((tmp_path / name).stat().st_size > 0 for name in required)
  rows = (tmp_path / "diagnostics_continuous_seed0.csv").read_text().splitlines()
  assert len(rows) == 19
