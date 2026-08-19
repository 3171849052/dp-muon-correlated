"""Focused regression tests for Experiment 3's online shadow step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
from exp3.online_shadow import (
    aggregate_ratio,
    init_online_shadow_state,
    make_online_shadow_train_step,
)
from exp3.run import build_schedule_for_seed, paired_seed_aggregation


def _setup():
  horizon = 3
  coef = jnp.asarray([1.0, -0.2], jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=coef, min_sep=1, max_participations=1
  )
  strategy = BandInvMFStrategy(
      horizon=horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=jnp.ones((horizon,), jnp.float32), noising_coef=coef,
      strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=sensitivity, objective=jnp.array(0., jnp.float32),
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2., delta=1e-5, clip_norm=10., normalize_by=1.,
      adjacency="add_remove", sensitivity_squared=float(sensitivity),
  )
  participation = ParticipationSpec(horizon, 1, 1)
  def loss(params, batch):
    return jnp.sum((params["w"] - batch["target"]) ** 2)
  return strategy, calibration, participation, loss


def _leaves_equal(left, right):
  for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
    if jnp.issubdtype(jnp.asarray(a).dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
    else:
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_jit_step_and_standard_semantics_match():
  strategy, calibration, participation, loss = _setup()
  online_step, online_optimizer = make_online_shadow_train_step(
      loss, strategy, calibration, participation, learning_rate=.01, weight_decay=.01,
  )
  standard_step, standard_optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      loss, strategy, calibration, participation, learning_rate=.01, weight_decay=.01,
  )
  params = {"w": jnp.array([1., -1.], jnp.float32)}
  key = jax.random.key(7)
  online = init_online_shadow_state(params, strategy, key, online_optimizer)
  standard = init_nonamplified_bandinv_dpadamw_state(params, strategy, key, standard_optimizer)
  batch = {"target": jnp.array([.25, -.5], jnp.float32)}
  compiled_online = jax.jit(online_step)
  compiled_standard = jax.jit(standard_step)
  for _ in range(3):
    online = compiled_online(online, batch)
    standard = compiled_standard(standard, batch)
  _leaves_equal(online.params, standard.params)
  _leaves_equal(online.optimizer_state, standard.optimizer_state)
  _leaves_equal(online.noise_state, standard.noise_state)
  np.testing.assert_array_equal(jax.random.key_data(online.rng_key), jax.random.key_data(standard.rng_key))
  assert int(online.step) == int(standard.step) == 3


def test_shadow_checkpoint_roundtrip_and_resume(tmp_path: Path):
  strategy, calibration, participation, loss = _setup()
  step, optimizer = make_online_shadow_train_step(loss, strategy, calibration, participation, learning_rate=.01)
  initial = init_online_shadow_state({"w": jnp.array([1., -1.], jnp.float32)}, strategy, jax.random.key(9), optimizer)
  batch = {"target": jnp.array([.25, -.5], jnp.float32)}
  compiled = jax.jit(step)
  uninterrupted = compiled(compiled(initial, batch), batch)
  path = tmp_path / "shadow.pkl"
  save_checkpoint(path, state=uninterrupted, current_step=2, experiment_config={"x": 1}, artifact_identifiers={"a": "b"})
  loaded = load_checkpoint(path)["state"]
  _leaves_equal(loaded, uninterrupted)
  resumed = compiled(loaded, batch)
  expected = compiled(uninterrupted, batch)
  _leaves_equal(resumed, expected)


def test_aggregate_ratio_and_synthetic_recurrence():
  eta, rho = .5, .8
  responses = [jnp.array([1., 0.]), jnp.array([0., 2.]), jnp.array([1., 1.])]
  d = jnp.zeros((2,)); prefix_d = sum_j = sum_d = 0.
  expected_prefixes, expected_js = [], []
  for response in responses:
    x = -eta * response
    d = rho * d + x
    energy = float(jnp.sum(x * x)); prefix_d += energy
    current_j = float(jnp.sum(d * d)); sum_j += current_j; sum_d += prefix_d
    expected_prefixes.append(prefix_d); expected_js.append(current_j)
  assert expected_prefixes == pytest.approx([.25, 1.25, 1.75])
  assert sum_d == pytest.approx(3.25)
  assert sum_j == pytest.approx(sum(expected_js))
  assert float(aggregate_ratio(sum_j, sum_d)) == pytest.approx(sum_j / 3.25)
  assert float(aggregate_ratio(sum_j, sum(responses[0] ** 2) * eta ** 2 + sum(responses[1] ** 2) * eta ** 2 + sum(responses[2] ** 2) * eta ** 2)) != pytest.approx(sum_j / sum_d)


def test_participation_validation_is_applied():
  strategy, calibration, _, loss = _setup()
  with pytest.raises(ValueError):
    make_online_shadow_train_step(
        loss, strategy, calibration, ParticipationSpec(4, 1, 1), learning_rate=.01,
    )


def test_paired_multi_seed_aggregation_and_single_seed_std():
  results = []
  for seed, naive_r, aware_r, naive_acc, aware_acc, naive_loss, aware_loss in (
      (0, 1.0, .8, .20, .25, 2.0, 1.5),
      (1, .6, .9, .30, .28, 1.4, 1.6),
      (2, .5, .2, .40, .45, 1.2, .9),
  ):
    results.extend([
        {"seed": seed, "strategy": "decayed-prefix", "R_linear": naive_r,
         "R_adamw": naive_r + .1, "final_accuracy": naive_acc, "final_test_loss": naive_loss},
        {"seed": seed, "strategy": "adam-m-aware", "R_linear": aware_r,
         "R_adamw": aware_r + .2, "final_accuracy": aware_acc, "final_test_loss": aware_loss},
    ])
  paired, aggregate = paired_seed_aggregation(results)
  assert [row["seed"] for row in paired] == [0, 1, 2]
  assert [row["delta_R_linear"] for row in paired] == pytest.approx([-.2, .3, -.3])
  assert [row["delta_R_adamw"] for row in paired] == pytest.approx([-.1, .4, -.2])
  assert [row["gamma_R"] for row in paired] == pytest.approx([.1, .1, .1])
  assert [row["delta_accuracy"] for row in paired] == pytest.approx([.05, -.02, .05])
  assert [row["delta_test_loss"] for row in paired] == pytest.approx([.5, -.2, .3])
  assert aggregate["num_seeds"] == 3
  assert aggregate["delta_R_linear_mean"] == pytest.approx(-.0666666667)
  assert aggregate["delta_R_linear_std"] == pytest.approx(np.std([-.2, .3, -.3], ddof=1))
  one_paired, one_aggregate = paired_seed_aggregation(results[:2])
  assert one_aggregate["num_seeds"] == 1
  for key, value in one_aggregate.items():
    if key.endswith("_std"):
      assert value == 0.0


def test_schedule_is_paired_within_seed_and_changes_between_seeds():
  contract = SimpleNamespace(num_examples=10, batch_size=2, horizon=10, min_sep=5, max_participations=2)
  schedule_0_naive = build_schedule_for_seed(contract, 0)
  schedule_0_aware = build_schedule_for_seed(contract, 0)
  schedule_1 = build_schedule_for_seed(contract, 1)
  assert all(np.array_equal(a, b) for a, b in zip(schedule_0_naive, schedule_0_aware, strict=True))
  assert any(not np.array_equal(a, b) for a, b in zip(schedule_0_naive, schedule_1, strict=True))
  assert len(schedule_0_naive) == len(schedule_1) == contract.horizon
  assert all(np.unique(np.concatenate(schedule)).size == contract.num_examples for schedule in (schedule_0_naive, schedule_1))
