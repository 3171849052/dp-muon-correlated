"""Focused mechanics tests for Experiment 10."""

from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy, init_bandinv_noise_state
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from exp10.core import (
    COMPONENTS,
    adam_private_v_hat,
    bandinv_marginal_variances,
    component_identity_residual,
    component_metrics,
    init_exp10_train_state,
    make_exp10_train_step,
    paired_noise_from_innovation,
    second_moment_components,
)
from exp10.diagnostics import (
    Exp10Collector,
    histogram_checkpoint_steps,
    save_histograms,
)


def _strategy(horizon=6):
  coef = jnp.asarray([1.0, 0.5], jnp.float32)
  return BandInvMFStrategy(
      horizon=horizon,
      bandwidth=2,
      min_sep=1,
      max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(horizon, .01, .01),
      noising_coef=coef,
      strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=toeplitz.compute_banded_inverse_sensitivity_squared(
          n=horizon,
          noising_coef=coef,
          min_sep=1,
          max_participations=1,
      ),
      objective=jnp.asarray(1.0, jnp.float32),
  )


def _calibration(strategy):
  return calibrate_nonamplified_bandinv(
      epsilon=3.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=2.0,
      adjacency="add_remove",
      sensitivity_squared=float(strategy.sensitivity_squared),
  )


def _tree_allclose(left, right, **kwargs):
  for a, b in zip(
      jax.tree_util.tree_leaves(left),
      jax.tree_util.tree_leaves(right),
      strict=True,
  ):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), **kwargs)


def _fake_state(step: int = 1):
  g = {"w": jnp.asarray([1.0, 1.0], jnp.float32)}
  xi = {"w": jnp.asarray([-1.0, 0.25], jnp.float32)}
  instantaneous = {
      branch: second_moment_components(g, xi) for branch in ("mf", "iid")
  }
  ema = {
      branch: {
          "V_g": instantaneous[branch]["g2"],
          "V_g_cross": instantaneous[branch]["g2_cross"],
          "V_xi": instantaneous[branch]["xi2"],
      }
      for branch in ("mf", "iid")
  }
  metrics = {
      branch: component_metrics(instantaneous[branch])
      for branch in ("mf", "iid")
  }
  last = SimpleNamespace(
      instantaneous=instantaneous,
      ema=ema,
      metrics=metrics,
      decomposition_error_max_abs={
          "mf": jnp.asarray(0.0), "iid": jnp.asarray(0.0)
      },
      decomposition_error_rms={
          "mf": jnp.asarray(0.0), "iid": jnp.asarray(0.0)
      },
      phi_t=jnp.asarray(.5),
  )
  return SimpleNamespace(step=jnp.asarray(step), last_step=last)


def test_mf_and_iid_have_the_same_per_step_marginal_variance():
  strategy = _strategy(horizon=6)
  sigma = 2.0
  phi = np.asarray(bandinv_marginal_variances(strategy, sigma))
  expected = sigma**2 * np.asarray([1.0, 1.25, 1.25, 1.25, 1.25, 1.25])
  np.testing.assert_allclose(phi, expected)
  # IID construction uses exactly this variance for every coordinate of z_t.
  z = {"w": jnp.asarray([-.2, 1.3], jnp.float32)}
  state = init_bandinv_noise_state({"w": jnp.zeros((2,), jnp.float32)}, 2)
  _, iid, _ = paired_noise_from_innovation(
      state, z, strategy, sigma, phi[2]
  )
  np.testing.assert_allclose(
      np.asarray(iid["w"] ** 2), float(phi[2]) * np.asarray(z["w"] ** 2),
      atol=2e-6,
  )


def test_mf_and_iid_consume_the_same_current_latent_z():
  strategy = _strategy()
  params = {"w": jnp.zeros((2,), jnp.float32)}
  state = init_bandinv_noise_state(params, strategy.bandwidth)
  z = {"w": jnp.asarray([.25, -.75], jnp.float32)}
  sigma = .7
  phi = bandinv_marginal_variances(strategy, sigma)
  mf, iid, next_state = paired_noise_from_innovation(
      state, z, strategy, sigma, phi[0]
  )
  expected_state = init_bandinv_noise_state(params, strategy.bandwidth)
  expected_mf, _, expected_state = paired_noise_from_innovation(
      expected_state, z, strategy, sigma, phi[0]
  )
  _tree_allclose(mf, expected_mf)
  _tree_allclose(iid, {"w": jnp.sqrt(phi[0]) * z["w"]})
  assert int(next_state.step) == 1


def test_component_identity_holds_without_clipping_cross_term():
  g = {"w": jnp.asarray([2.0, -.5], jnp.float32)}
  xi = {"w": jnp.asarray([-3.0, .25], jnp.float32)}
  _tree_allclose(component_identity_residual(g, xi), {"w": jnp.zeros((2,))}, atol=1e-6)
  components = second_moment_components(g, xi)
  _tree_allclose(
      jax.tree_util.tree_map(
          lambda cross, noise2: cross + noise2,
          components["g2_cross"],
          components["xi2"],
      ),
      jax.tree_util.tree_map(lambda a, b: (a + b) ** 2, g, xi),
      atol=1e-6,
  )
  assert np.any(np.asarray(components["g2_cross"]["w"]) < 0)


def test_bias_corrected_ema_decomposes_actual_adam_second_moment():
  strategy = _strategy(horizon=6)
  calibration = _calibration(strategy)

  def loss(params, batch):
    return .5 * (jnp.dot(params["w"], batch["x"][0]) - batch["y"][0]) ** 2

  step_fn, optimizer = make_exp10_train_step(
      loss,
      strategy,
      calibration,
      ParticipationSpec(strategy.horizon, 1, 1),
      learning_rate=.01,
      beta1=.9,
      beta2=.8,
      eps=1e-6,
      weight_decay=.01,
  )
  state = init_exp10_train_state(
      {"w": jnp.asarray([.1, -.2], jnp.float32)},
      strategy,
      jax.random.key(4),
      optimizer,
  )
  batch = {
      "x": jnp.asarray([[1.0, .5], [1.0, -.5]], jnp.float32),
      "y": jnp.asarray([.2, -.1], jnp.float32),
  }
  state = step_fn(state, batch)
  actual = adam_private_v_hat(state.mf_optimizer_state, beta2=.8)
  expected = jax.tree_util.tree_map(
      lambda cross, noise2: cross + noise2,
      state.last_step.ema["mf"]["V_g_cross"],
      state.last_step.ema["mf"]["V_xi"],
  )
  _tree_allclose(actual, expected, atol=2e-6, rtol=2e-6)
  assert float(state.last_step.decomposition_error_max_abs["mf"]) < 2e-6
  assert float(state.last_step.decomposition_error_max_abs["iid"]) < 2e-6


def test_histogram_checkpoints_include_final_non_multiple():
  assert histogram_checkpoint_steps(32) == [16, 32]
  assert histogram_checkpoint_steps(33) == [16, 32, 33]
  collector = Exp10Collector(
      {"w": jnp.zeros((2,), jnp.float32)}, seed=3, horizon=33, histogram_bins=8
  )
  for step in range(1, 34):
    collector.after_step(_fake_state(step), step)
  assert [record["step"] for record in collector.histogram_records] == [16, 32, 33]
  assert len(collector.rows) == 33 * 2


def test_histogram_retains_negative_cross_values_and_shared_bins(tmp_path):
  collector = Exp10Collector(
      {"w": jnp.zeros((2,), jnp.float32)}, seed=0, horizon=16, histogram_bins=8
  )
  for step in range(1, 17):
    collector.after_step(_fake_state(step), step)
  record = collector.histogram_records[0]
  assert record["bin_edges"][0] < 0
  assert record["counts"].shape == (2, len(COMPONENTS), 8)
  np.testing.assert_allclose(record["relative_frequency"].sum(axis=-1), 1.0)
  output = tmp_path / "histograms.npz"
  save_histograms(output, collector.histogram_records, histogram_bins=8)
  with np.load(output, allow_pickle=False) as data:
    assert tuple(data["component_names"].astype(str)) == COMPONENTS
    assert data["bin_edges"].shape == (1, 9)


def test_smoke_run_writes_all_exp10_artifacts(tmp_path):
  from exp10.run import run_smoke

  run_smoke(tmp_path, [0], histogram_bins=8)
  expected = {
      "summary.json", "step_metrics.csv", "stage_metrics.csv",
      "histograms.npz", "histograms.png",
  }
  assert expected.issubset({path.name for path in tmp_path.iterdir()})
  summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
  assert summary["experiment"] == "exp10"
  assert summary["smoke"] is True
  with np.load(tmp_path / "histograms.npz", allow_pickle=False) as data:
    np.testing.assert_array_equal(data["steps"], [16, 20])
