from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer
from exp5.hybrid_optimizer import FrozenPAdamW, freeze_optax_adamw, p_star_from_optax
from exp5.hybrid_strategy import (
    block_lengths, hybrid_noising_matrix, hybrid_sensitivity_squared,
    hybrid_strategy_matrix, p_aware_hybrid_objective,
)
from exp5.run import _plans
from exp5.replay import (
    FullPyTreeReplayAccumulator, filter_latent_draws, paired_replay,
)
from exp5.run import parse_args, run_smoke, select_tau_star
from exp5.workload import apply_frozen_p_workload, frozen_p_time_workload


def _warm_state():
  params = jnp.array([1., -2.])
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=.05, beta1=.8, beta2=.9, eps=1e-6, weight_decay=.01)
  state = optimizer.init(params)
  for gradient in (jnp.array([.2, -.1]), jnp.array([.4, .3])):
    updates, state = optimizer.update(gradient, state, params)
    params = optax.apply_updates(params, updates)
  return params, state


def _strategy(c):
  c = np.asarray(c, dtype=np.float64)
  d = np.linalg.inv(np.where(
      np.arange(len(c))[:, None] >= np.arange(len(c))[None, :],
      np.pad(c, (0, len(c)))[:len(c)][np.maximum(
          np.arange(len(c))[:, None] - np.arange(len(c))[None, :], 0)], 0.))
  return BandInvMFStrategy(len(c), len(c), 1, 2, None, jnp.asarray(d[:, 0]),
                           jnp.asarray(c), jnp.array(1.), jnp.array(1.))


def test_p_star_reads_real_optax_nu_and_count():
  _, state = _warm_state()
  adam = state[0]
  expected = 1 / (jnp.sqrt(adam.nu / (1 - .9 ** int(adam.count))) + 1e-6)
  np.testing.assert_allclose(
      p_star_from_optax(state, beta2=.9, eps=1e-6), expected, rtol=3e-7)


def test_switch_preserves_count_mu_and_freezes_nu_p():
  params, state = _warm_state()
  frozen = freeze_optax_adamw(state, beta2=.9, eps=1e-6)
  old = frozen
  updates, frozen = FrozenPAdamW(.05, .8, .01).update(jnp.array([.3, -.4]), frozen, params)
  assert int(frozen.count) == int(old.count) + 1
  np.testing.assert_array_equal(frozen.frozen_nu, old.frozen_nu)
  np.testing.assert_array_equal(frozen.p_star, old.p_star)
  assert not np.allclose(frozen.mu, old.mu)
  assert np.all(np.isfinite(updates))


def test_frozen_update_matches_formula():
  params, state = _warm_state()
  old = freeze_optax_adamw(state, beta2=.9, eps=1e-6)
  gradient = jnp.array([.3, -.4])
  updates, new = FrozenPAdamW(.05, .8, .01).update(gradient, old, params)
  mu = .8 * old.mu + .2 * gradient
  expected = -.05 * (old.p_star * mu / (1 - .8 ** int(new.count)) + .01 * params)
  np.testing.assert_allclose(updates, expected, rtol=2e-6)


def test_workload_uses_global_bias_correction():
  global_a = frozen_p_time_workload(3, tau=4, beta1=.8, learning_rate=.05, weight_decay=.01)
  restarted = frozen_p_time_workload(3, tau=1, beta1=.8, learning_rate=.05, weight_decay=.01)
  assert not np.allclose(global_a, restarted)
  assert np.isclose(global_a[0, 0], -.05 * .2 / (1 - .8 ** 5))


def test_p_aware_workload_matches_impulse_optimizer():
  tau, horizon = 3, 4
  p = jnp.array([2., .5])
  noise = jnp.zeros((horizon, 2)).at[1].set(jnp.array([.3, -.2]))
  predicted = apply_frozen_p_workload(
      frozen_p_time_workload(horizon, tau=tau, beta1=.8, learning_rate=.05,
                             weight_decay=.01), noise, p)
  theta = jnp.zeros(2); mu = jnp.zeros(2); actual = []
  for step in range(horizon):
    mu = .8 * mu + .2 * noise[step]
    theta = (1 - .05 * .01) * theta - .05 * p * mu / (1 - .8 ** (tau + step + 1))
    actual.append(theta)
  np.testing.assert_allclose(predicted, np.asarray(actual), rtol=2e-6, atol=1e-8)


def test_frozen_replay_is_linear_dynamic_is_not():
  rng = np.random.default_rng(4)
  result = paired_replay(
      rng.normal(size=(6, 2)) * .1, rng.normal(size=(6, 2)) * .05,
      params=np.array([.2, -.1]), mu=np.array([.03, -.02]),
      nu=np.array([.01, .04]), count=5, beta1=.8, beta2=.9, eps=1e-6,
      learning_rate=.03, weight_decay=.02)
  assert result.g_frozen < 1e-24
  assert result.g_dynamic > result.g_frozen + 1e-8


def test_full_pytree_online_replay_uses_every_leaf():
  beta1, beta2, eps = .8, .9, 1e-6
  learning_rate, weight_decay = .03, .02
  params = {
      "largest": jnp.array([.2, -.1, .05, .3]),
      "other": jnp.array([-.4, .15]),
  }
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
      weight_decay=weight_decay)
  state = optimizer.init(params)
  warm_gradients = [
      {"largest": jnp.array([.2, -.1, .3, .05]),
       "other": jnp.array([.01, -.5])},
      {"largest": jnp.array([-.1, .4, .05, -.2]),
       "other": jnp.array([.6, .02])},
      {"largest": jnp.array([.3, .1, -.2, .4]),
       "other": jnp.array([-.4, .3])},
  ]
  for gradient in warm_gradients:
    updates, state = optimizer.update(gradient, state, params)
    params = optax.apply_updates(params, updates)
  frozen = freeze_optax_adamw(state, beta2=beta2, eps=eps)
  accumulator = FullPyTreeReplayAccumulator(
      params=params, dynamic_optimizer=optimizer, dynamic_state=state,
      frozen_state=frozen, learning_rate=learning_rate, beta1=beta1,
      weight_decay=weight_decay)
  clean = [
      {"largest": jnp.array([.1, -.2, .05, .3]),
       "other": jnp.array([.7, -.4])},
      {"largest": jnp.array([-.3, .1, .2, -.1]),
       "other": jnp.array([-.5, .8])},
      {"largest": jnp.array([.2, .05, -.1, .4]),
       "other": jnp.array([.9, -.6])},
  ]
  noise = [
      {"largest": jnp.array([.01, -.02, .015, -.01]),
       "other": jnp.array([.2, -.15])},
      {"largest": jnp.array([-.015, .01, .02, -.005]),
       "other": jnp.array([-.18, .25])},
      {"largest": jnp.array([.02, .005, -.01, .015]),
       "other": jnp.array([.3, -.2])},
  ]
  for gradient, perturbation in zip(clean, noise, strict=True):
    accumulator.update(gradient, perturbation)
  result = accumulator.result()
  assert result.g_frozen < 1e-10
  assert result.g_dynamic > result.g_frozen + 1e-8

  adam = state[0]
  leaf_results = {}
  for leaf in params:
    leaf_results[leaf] = paired_replay(
        np.asarray([gradient[leaf] for gradient in clean]),
        np.asarray([perturbation[leaf] for perturbation in noise]),
        params=np.asarray(params[leaf]), mu=np.asarray(adam.mu[leaf]),
        nu=np.asarray(adam.nu[leaf]), count=int(adam.count), beta1=beta1,
        beta2=beta2, eps=eps, learning_rate=learning_rate,
        weight_decay=weight_decay)
  assert np.isclose(
      result.denominator,
      sum(leaf.denominator for leaf in leaf_results.values()), rtol=2e-5)
  assert np.isclose(
      result.numerator_dynamic,
      sum(leaf.numerator_dynamic for leaf in leaf_results.values()), rtol=2e-4)
  assert not np.isclose(result.g_dynamic, leaf_results["largest"].g_dynamic,
                        rtol=1e-2)


def test_segment_filter_resets_only_noise():
  latent = np.arange(6., dtype=float)[:, None]
  d1 = np.array([[1., 0., 0.], [.5, 1., 0.], [.2, .5, 1.]])
  d2 = d1.copy()
  out = filter_latent_draws(latent, (d1, d2))
  assert out[3, 0] == latent[3, 0]
  assert out[2, 0] != latent[2, 0]


def test_last_partial_block():
  assert block_lengths(11, 4) == (4, 4, 3)


def test_hybrid_sensitivity_matches_bruteforce_oracle():
  c = np.array([[1., 0., 0., 0., 0.], [.2, 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.], [0., 0., .4, 1., 0.],
                [0., 0., .1, .4, 1.]])
  actual = hybrid_sensitivity_squared(c, min_sep=2, max_participations=2)
  oracle = 0.
  for count in (1, 2):
    for positions in itertools.combinations(range(5), count):
      if all(b - a >= 2 for a, b in zip(positions, positions[1:])):
        oracle = max(oracle, np.sum(np.sum(c[:, positions], axis=1) ** 2))
  assert np.isclose(actual, oracle)


def test_global_separation_crosses_hybrid_boundary():
  c = np.eye(6)
  assert hybrid_sensitivity_squared(c, min_sep=3, max_participations=3) == 2.


def test_hybrid_d_and_c_are_inverses_and_keep_partial_block():
  strategies = (_strategy([1., .2]), _strategy([1., -.1, .05]))
  c = hybrid_strategy_matrix(2, strategies)
  d = hybrid_noising_matrix(2, strategies)
  np.testing.assert_allclose(c @ d, np.eye(7), atol=1e-6)


def test_tau_selection_accuracy_then_loss():
  rows = [
      {"tau": 2, "final_test_accuracy": .8, "final_test_loss": .4},
      {"tau": 2, "final_test_accuracy": .6, "final_test_loss": .5},
      {"tau": 4, "final_test_accuracy": .7, "final_test_loss": .3},
      {"tau": 4, "final_test_accuracy": .7, "final_test_loss": .4},
  ]
  assert select_tau_star(rows) == 4


def test_cli_parses_seed_and_tau_lists():
  args = parse_args(["--seeds", "3", "5", "--tau-candidates", "2", "6", "--smoke"])
  assert args.seeds == [3, 5] and args.tau_candidates == [2, 6] and args.smoke


def test_smoke_writes_all_outputs(tmp_path: Path):
  tau = run_smoke(tmp_path, [0], [2, 4])
  assert tau in {2, 4}
  required = {"switch_comparison.csv", "switch_summary.json",
              "nonlinearity_comparison.csv", "nonlinearity_summary.json",
              "replay_results.csv", "replay_summary.json"}
  assert required.issubset({path.name for path in tmp_path.iterdir()})
  summary = json.loads((tmp_path / "switch_summary.json").read_text())
  assert summary["tau_star"] == tau
  with (tmp_path / "replay_results.csv").open() as stream:
    rows = list(csv.DictReader(stream))
  assert all(float(row["G_frozen"]) < 1e-12 for row in rows)
  assert {"numerator_dynamic", "numerator_frozen", "denominator"}.issubset(rows[0])
  nonlinear = json.loads((tmp_path / "nonlinearity_summary.json").read_text())
  assert nonlinear["paired_within_mechanism"] is True
  assert nonlinear["shared_base_gaussian_seeds_across_mechanisms"] is True
  assert nonlinear["exact_equal_final_privacy_target"] is True
  assert set(nonlinear["mechanisms"]) == {"cont", "seg97"}
  for mechanism in nonlinear["mechanisms"].values():
    assert mechanism["epsilon"] == 3.0
    assert mechanism["delta"] == 1e-5
    assert mechanism["sensitivity_squared"] > 0
    assert mechanism["iid_noise_std"] > 0


def test_smoke_privacy_is_single_final_calibration(tmp_path: Path):
  run_smoke(tmp_path, [1], [2, 4])
  summary = json.loads((tmp_path / "switch_summary.json").read_text())
  assert summary["privacy"] == {
      "epsilon": 3.0, "delta": 1e-5, "adjacency": "add_remove",
      "calibration_scope": "one full hybrid transcript per run"}


def test_mechanisms_are_independently_calibrated_to_equal_final_privacy():
  plans = {
      "cont": _plans(10, 2, segmented=False),
      "seg": _plans(10, 2, segmented=True),
  }
  assert plans["cont"].sensitivity_squared != plans["seg"].sensitivity_squared
  assert plans["cont"].calibration.iid_noise_std != plans["seg"].calibration.iid_noise_std
  assert plans["cont"].calibration.epsilon == plans["seg"].calibration.epsilon == 3.
  assert plans["cont"].calibration.delta == plans["seg"].calibration.delta == 1e-5


def test_complete_objective_is_explicitly_p_weighted():
  plan = _plans(10, 2, segmented=True)
  one = p_aware_hybrid_objective(
      plan, jnp.ones(3), beta1=.8, learning_rate=.04, weight_decay=.02)
  two = p_aware_hybrid_objective(
      plan, jnp.full(3, 2.), beta1=.8, learning_rate=.04, weight_decay=.02)
  assert np.isclose(two, 4 * one)
