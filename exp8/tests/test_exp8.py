"""Focused tests for Exp8's paired noise and layer decomposition."""

from types import SimpleNamespace
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import (
    BandInvMFStrategy,
    fit_bandinv_strategy,
    filter_latent_noise,
    init_bandinv_noise_state,
)
from jax_privacy.matrix_factorization import toeplitz
from dp_muon.optim import (
    adam_first_moment_workload_matrix,
    decayed_prefix_sum_workload_coef,
)
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from exp8.core import (
    BRANCHES,
    PATHS,
    Exp8DiagnosticStep,
    advance_diagnostic_shadow,
    bandinv_marginal_variances,
    init_exp8_train_state,
    init_diagnostic_shadow_state,
    make_exp8_train_step,
    paired_diagnostic_noise_from_innovation,
    sample_paired_diagnostic_noise,
)
from exp8.diagnostics import (
    cancellation_statistics,
    cross_seed_aggregate,
    paired_gains,
)
from exp8.online_shadow import Exp8WindowCollector


def _strategy(horizon=6):
  coef = jnp.asarray([1.0, .5], jnp.float32)
  return BandInvMFStrategy(
      horizon=horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(horizon, .01, .01),
      noising_coef=coef,
      strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=toeplitz.compute_banded_inverse_sensitivity_squared(
          n=horizon, noising_coef=coef, min_sep=1, max_participations=1
      ),
      objective=jnp.asarray(1.0, jnp.float32),
  )


def _calibration(strategy):
  return calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )


def _momentum_strategy(horizon=6):
  return fit_bandinv_strategy(
      horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_matrix=adam_first_moment_workload_matrix(
          horizon, .9, .01, .01
      ),
      max_optimizer_steps=3,
  )


def _tree_allclose(left, right, **kwargs):
  for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), **kwargs)


def test_correlated_and_matched_iid_theoretical_marginal_variance_match():
  strategy = _strategy()
  sigma = 2.0
  phi = np.asarray(bandinv_marginal_variances(strategy, sigma))
  expected = sigma**2 * np.cumsum(np.asarray(strategy.noising_coef) ** 2)
  expected = expected[np.minimum(np.arange(strategy.horizon), strategy.bandwidth - 1)]
  np.testing.assert_allclose(phi, expected)
  np.testing.assert_allclose(phi, np.asarray(np.sqrt(phi) ** 2))


def test_paired_noise_uses_same_z_and_only_corr_has_filter_state():
  strategy = _strategy()
  params = {"w": jnp.zeros((3,), jnp.float32)}
  state = init_bandinv_noise_state(params, strategy.bandwidth)
  phi = bandinv_marginal_variances(strategy, 2.0)
  corr, iid, z, next_state, next_key = sample_paired_diagnostic_noise(
      jax.random.key(4), state, strategy, 2.0, phi[0]
  )
  _tree_allclose(iid, jax.tree_util.tree_map(lambda value: jnp.sqrt(phi[0]) * value, z))
  _tree_allclose(corr, jax.tree_util.tree_map(
      lambda value: 2.0 * strategy.noising_coef[0] * value, z
  ))
  assert int(next_state.step) == 1
  # The returned API has no IID state, and independent calls draw distinct z_t.
  _, _, z2, _, _ = sample_paired_diagnostic_noise(next_key, next_state, strategy, 2.0, phi[1])
  assert not np.array_equal(np.asarray(z["w"]), np.asarray(z2["w"]))


def test_diagnostic_filter_matches_formal_filter_for_ten_deterministic_steps():
  strategy = _strategy(horizon=10)
  params = {"w": jnp.zeros((2,), jnp.float32)}
  actual_state = init_bandinv_noise_state(params, strategy.bandwidth)
  reference_state = init_bandinv_noise_state(params, strategy.bandwidth)
  sigma = .7
  phi = bandinv_marginal_variances(strategy, sigma)
  innovations = [
      {"w": jnp.asarray([step + .25, -2.0 * step - .5], jnp.float32)}
      for step in range(10)
  ]
  for step, z in enumerate(innovations):
    actual_corr, actual_iid, actual_state = paired_diagnostic_noise_from_innovation(
        actual_state, z, strategy, sigma, phi[step]
    )
    latent = jax.tree_util.tree_map(lambda value: sigma * value, z)
    expected_corr, reference_state = filter_latent_noise(
        reference_state, latent, strategy.noising_coef
    )
    _tree_allclose(actual_corr, expected_corr, atol=1e-6)
    _tree_allclose(
        actual_iid,
        jax.tree_util.tree_map(lambda value: jnp.sqrt(phi[step]) * value, z),
        atol=1e-6,
    )
    _tree_allclose(actual_state.buffer, reference_state.buffer, atol=1e-6)
    np.testing.assert_array_equal(actual_state.cursor, reference_state.cursor)
    np.testing.assert_array_equal(actual_state.step, reference_state.step)


def test_training_and_diagnostic_rng_streams_are_distinct_and_shadows_do_not_update_real_state():
  strategy = _strategy()
  calibration = _calibration(strategy)
  participation = ParticipationSpec(strategy.horizon, 1, 1)

  def loss(params, batch):
    return .5 * (jnp.dot(params["w"], batch["x"][0]) - batch["y"][0]) ** 2

  step_fn, optimizer = make_exp8_train_step(
      loss, strategy, calibration, participation,
      diagnostic_strategy=strategy, diagnostic_calibration=calibration,
      learning_rate=.01, beta1=.9, beta2=.9, eps=1e-6, weight_decay=.01,
  )
  params = {"w": jnp.asarray([.1, -.2, .3], jnp.float32)}
  training_key = jax.random.key(12)
  state1 = init_exp8_train_state(
      params, strategy, training_key, optimizer, jax.random.key(13),
      diagnostic_strategy=strategy,
  )
  state2 = init_exp8_train_state(
      params, strategy, training_key, optimizer, jax.random.key(14),
      diagnostic_strategy=strategy,
  )
  assert not np.array_equal(
      np.asarray(jax.random.key_data(state1.training_rng_key)),
      np.asarray(jax.random.key_data(state1.diagnostic_rng_key)),
  )
  batch = {"x": jnp.ones((2, 3), jnp.float32), "y": jnp.asarray([.2, .2], jnp.float32)}
  next1, next2 = step_fn(state1, batch), step_fn(state2, batch)
  _tree_allclose(next1.params, next2.params)
  _tree_allclose(next1.optimizer_state, next2.optimizer_state)
  # Diagnostic branches differ, proving they are not accidentally feeding the update.
  assert not np.array_equal(
      np.asarray(next1.last_step.z["w"]), np.asarray(next2.last_step.z["w"])
  )


def test_momentum_aware_diagnostic_strategy_is_separate_and_drives_phi():
  training_strategy = _strategy()
  diagnostic_strategy = _momentum_strategy()
  training_calibration = _calibration(training_strategy)
  diagnostic_calibration = _calibration(diagnostic_strategy)
  expected_workload = adam_first_moment_workload_matrix(6, .9, .01, .01)
  np.testing.assert_allclose(
      np.asarray(diagnostic_strategy.workload_matrix), np.asarray(expected_workload)
  )
  assert diagnostic_strategy.workload_matrix is not None

  def loss(params, batch):
    return .5 * (jnp.dot(params["w"], batch["x"][0]) - batch["y"][0]) ** 2

  participation = ParticipationSpec(6, 1, 1)
  step_fn, optimizer = make_exp8_train_step(
      loss, training_strategy, training_calibration, participation,
      diagnostic_strategy=diagnostic_strategy,
      diagnostic_calibration=diagnostic_calibration,
      learning_rate=.01, beta1=.9, beta2=.9, eps=1e-6, weight_decay=.01,
  )
  state = init_exp8_train_state(
      {"w": jnp.asarray([.1, -.2, .3], jnp.float32)}, training_strategy,
      jax.random.key(30), optimizer, jax.random.key(31),
      diagnostic_strategy=diagnostic_strategy,
  )
  batch = {
      "x": jnp.ones((2, 3), jnp.float32),
      "y": jnp.asarray([.2, .2], jnp.float32),
  }
  next_state = step_fn(state, batch)
  expected_phi = bandinv_marginal_variances(
      diagnostic_strategy, diagnostic_calibration.iid_noise_std
  )
  np.testing.assert_allclose(float(next_state.last_step.phi_t), float(expected_phi[0]))
  assert float(next_state.last_step.phi_t) != float(
      bandinv_marginal_variances(
          training_strategy, training_calibration.iid_noise_std
      )[0]
  )


def test_diagnostic_strategy_and_rng_cannot_change_real_training_update():
  training_strategy = _strategy()
  diagnostic_a = _momentum_strategy()
  diagnostic_b = _strategy()
  training_calibration = _calibration(training_strategy)
  diagnostic_calibration_a = _calibration(diagnostic_a)
  diagnostic_calibration_b = _calibration(diagnostic_b)
  participation = ParticipationSpec(6, 1, 1)

  def loss(params, batch):
    return .5 * (jnp.dot(params["w"], batch["x"][0]) - batch["y"][0]) ** 2

  step_a, optimizer_a = make_exp8_train_step(
      loss, training_strategy, training_calibration, participation,
      diagnostic_strategy=diagnostic_a,
      diagnostic_calibration=diagnostic_calibration_a,
      learning_rate=.01,
  )
  step_b, optimizer_b = make_exp8_train_step(
      loss, training_strategy, training_calibration, participation,
      diagnostic_strategy=diagnostic_b,
      diagnostic_calibration=diagnostic_calibration_b,
      learning_rate=.01,
  )
  params = {"w": jnp.asarray([.1, -.2, .3], jnp.float32)}
  state_a = init_exp8_train_state(
      params, training_strategy, jax.random.key(40), optimizer_a, jax.random.key(41),
      diagnostic_strategy=diagnostic_a,
  )
  state_b = init_exp8_train_state(
      params, training_strategy, jax.random.key(40), optimizer_b, jax.random.key(99),
      diagnostic_strategy=diagnostic_b,
  )
  batch = {
      "x": jnp.ones((2, 3), jnp.float32),
      "y": jnp.asarray([.2, .2], jnp.float32),
  }
  next_a, next_b = step_a(state_a, batch), step_b(state_b, batch)
  _tree_allclose(next_a.params, next_b.params)
  _tree_allclose(next_a.optimizer_state, next_b.optimizer_state)
  _tree_allclose(next_a.training_noise_state.buffer, next_b.training_noise_state.buffer)
  np.testing.assert_array_equal(
      jax.random.key_data(next_a.training_rng_key),
      jax.random.key_data(next_b.training_rng_key),
  )


def test_train_step_invokes_the_single_clipped_query_once(monkeypatch):
  strategy = _strategy()
  calibration = _calibration(strategy)
  calls = []

  def fake_query_factory(*args, **kwargs):
    del args, kwargs
    def query(params, batch):
      del params, batch
      calls.append("query")
      return {"w": jnp.asarray([.1, .1, .1], jnp.float32)}
    return query

  import exp8.core as core
  monkeypatch.setattr(core, "make_clipped_gradient_query", fake_query_factory)
  step_fn, optimizer = core.make_exp8_train_step(
      lambda params, batch: jnp.asarray(0.0), strategy, calibration,
      ParticipationSpec(strategy.horizon, 1, 1), learning_rate=.01,
      diagnostic_strategy=strategy, diagnostic_calibration=calibration,
  )
  state = core.init_exp8_train_state(
      {"w": jnp.zeros((3,), jnp.float32)}, strategy, jax.random.key(20),
      optimizer, jax.random.key(21), diagnostic_strategy=strategy,
  )
  step_fn(state, {"unused": jnp.asarray([0.0])})
  assert calls == ["query"]


def test_r_is_bias_corrected_pure_noise_momentum_response_and_phi_ema():
  params = {"w": jnp.zeros((1,), jnp.float32)}
  state = init_diagnostic_shadow_state(params)
  beta1, beta2 = .5, .8
  xi1 = {"w": jnp.asarray([2.0], jnp.float32)}
  xi2 = {"w": jnp.asarray([-1.0], jnp.float32)}
  g = {"w": jnp.asarray([3.0], jnp.float32)}
  state, first = advance_diagnostic_shadow(
      state, g, xi1, xi1, .25, beta1=beta1, beta2=beta2,
      learning_rate=.1, eps=1e-6,
  )
  expected1 = (1 - beta1) * 2.0 / (1 - beta1)
  np.testing.assert_allclose(np.asarray(first.r["corr"]["w"]), [expected1])
  np.testing.assert_allclose(np.asarray(first.r["iid"]["w"]), [expected1])
  np.testing.assert_allclose(float(first.Phi_t), (1 - beta2) * .25 / (1 - beta2))
  state, second = advance_diagnostic_shadow(
      state, g, xi2, xi2, .25, beta1=beta1, beta2=beta2,
      learning_rate=.1, eps=1e-6,
  )
  expected2 = (beta1 * (1 - beta1) * 2.0 + (1 - beta1) * -1.0) / (1 - beta1**2)
  np.testing.assert_allclose(np.asarray(second.r["corr"]["w"]), [expected2])
  np.testing.assert_allclose(np.asarray(second.r["iid"]["w"]), [expected2])
  denominator = 1.0 - beta1**2
  corr_difference = state.corr_m["w"] / denominator - state.clean_m["w"] / denominator
  iid_difference = state.iid_m["w"] / denominator - state.clean_m["w"] / denominator
  np.testing.assert_allclose(np.asarray(corr_difference), np.asarray(second.r["corr"]["w"]), atol=1e-6)
  np.testing.assert_allclose(np.asarray(iid_difference), np.asarray(second.r["iid"]["w"]), atol=1e-6)
  expected_phi2 = (beta2 * (1 - beta2) * .25 + (1 - beta2) * .25) / (1 - beta2**2)
  np.testing.assert_allclose(float(second.Phi_t), expected_phi2, atol=1e-6)


def test_pure_noise_ema_and_private_difference_match_for_eight_steps_both_branches():
  params = {"w": jnp.zeros((2,), jnp.float32)}
  state = init_diagnostic_shadow_state(params)
  beta1, beta2 = .7, .85
  clean_m = np.zeros(2, np.float32)
  noise_m = {branch: np.zeros(2, np.float32) for branch in BRANCHES}
  for step in range(1, 9):
    g = {"w": jnp.asarray([1.0 + step, -2.0 + .5 * step], jnp.float32)}
    xi = {
        "corr": {"w": jnp.asarray([.25 * step, -1.0], jnp.float32)},
        "iid": {"w": jnp.asarray([-0.5, .15 * step], jnp.float32)},
    }
    state, out = advance_diagnostic_shadow(
        state, g, xi["corr"], xi["iid"], .4,
        beta1=beta1, beta2=beta2, learning_rate=.03, eps=1e-6,
    )
    clean_m = beta1 * clean_m + (1 - beta1) * np.asarray(g["w"])
    for branch in BRANCHES:
      noise_m[branch] = beta1 * noise_m[branch] + (1 - beta1) * np.asarray(xi[branch]["w"])
      expected = noise_m[branch] / (1 - beta1**step)
      np.testing.assert_allclose(np.asarray(out.r[branch]["w"]), expected, atol=2e-6)
      private_m = state.corr_m["w"] if branch == "corr" else state.iid_m["w"]
      difference = np.asarray(private_m) / (1 - beta1**step) - clean_m / (1 - beta1**step)
      np.testing.assert_allclose(difference, expected, atol=2e-6)
    assert float(out.r_difference_error_corr) < 2e-6
    assert float(out.r_difference_error_iid) < 2e-6
def test_p0_p1_p2_p3_formulas_and_p2_is_deterministic_bias_only():
  params = {"w": jnp.asarray([0.0], jnp.float32)}
  state = init_diagnostic_shadow_state(params)
  g = {"w": jnp.asarray([1.0], jnp.float32)}
  xi_corr = {"w": jnp.asarray([.5], jnp.float32)}
  xi_iid = {"w": jnp.asarray([-.25], jnp.float32)}
  _, out = advance_diagnostic_shadow(
      state, g, xi_corr, xi_iid, .25, beta1=.5, beta2=.5,
      learning_rate=.1, eps=0.0,
  )
  # t=1: mhat_c=1, r_corr=.5, vhat_c=1, Phi=.25, vhat_p=2.25.
  np.testing.assert_allclose(np.asarray(out.x["corr"]["P0"]["w"]), [-.05])
  np.testing.assert_allclose(np.asarray(out.x["corr"]["P1"]["w"]), [-.05])
  np.testing.assert_allclose(
      np.asarray(out.x["corr"]["P2"]["w"]), [-.05 / np.sqrt(1.25)], atol=1e-6
  )
  np.testing.assert_allclose(np.asarray(out.x["corr"]["P3"]["w"]), [-.05 / 1.5])
  # P2's scale is shared and contains no realization-specific 2*g*xi or xi^2.
  p2_corr_over_r = np.asarray(out.x["corr"]["P2"]["w"] / out.r["corr"]["w"])
  p2_iid_over_r = np.asarray(out.x["iid"]["P2"]["w"] / out.r["iid"]["w"])
  np.testing.assert_allclose(p2_corr_over_r, p2_iid_over_r)
  for field in ("A", "B", "I"):
    value = getattr(out, field)
    assert value is not None
  _tree_allclose(out.dq, jax.tree_util.tree_map(
      lambda a, b, c: a + b + c, out.A, out.B, out.I
  ), atol=1e-6)
  assert float(out.reconstruction_error) <= 1e-6


def test_hand_cancellation_and_gains():
  result = cancellation_statistics(
      np.asarray([[1.0], [2.0]]), weight_decay=.0, learning_rate=.1
  )
  # x is already scaled, so use a=1 and d_e=3.
  np.testing.assert_allclose(result["J"], 9.0)
  np.testing.assert_allclose(result["D"], 5.0)
  np.testing.assert_allclose(result["C"], 1.8)
  gains = paired_gains({"C": 2.0, "J": 3.0}, {"C": 4.0, "J": 6.0})
  assert gains == {"G_C": .5, "G_J": .5}
  assert cancellation_statistics(
      np.zeros((2, 1)), weight_decay=.01, learning_rate=.1
  ) == {"J": 0.0, "D": 0.0, "C": 0.0}


def _fake_step(value: float) -> Exp8DiagnosticStep:
  zero = {"w": jnp.asarray([0.0], jnp.float32)}
  x = {branch: {path: {"w": jnp.asarray([0.0], jnp.float32)} for path in PATHS} for branch in BRANCHES}
  x["corr"]["P0"] = {"w": jnp.asarray([value], jnp.float32)}
  return Exp8DiagnosticStep(
      r={branch: zero for branch in BRANCHES}, x=x,
      A=zero, B=zero, I=zero, dq=zero,
      xi={branch: zero for branch in BRANCHES}, z=zero,
      phi_t=jnp.asarray(0.0), Phi_t=jnp.asarray(0.0),
      reconstruction_error=jnp.asarray(0.0),
      r_difference_error_corr=jnp.asarray(0.0),
      r_difference_error_iid=jnp.asarray(0.0),
  )


def test_exact_stage_boundaries_are_not_window_weighted_averages():
  params = {"w": jnp.asarray([0.0], jnp.float32)}
  collector = Exp8WindowCollector(
      params, seed=0, learning_rate=0.0, weight_decay=0.0, horizon=100,
  )
  for step in range(1, 101):
    fake = SimpleNamespace(step=jnp.asarray(step), last_step=_fake_step(1.0 if step <= 97 else 2.0))
    collector.after_step(fake, step)
  stages = collector.stage_summaries()
  assert stages["early"]["start_step"] == 1
  assert stages["early"]["end_step"] == 97
  assert stages["early"]["num_steps"] == 97
  assert stages["late"]["start_step"] == 98
  assert stages["late"]["num_steps"] == 3
  assert stages["full"]["num_steps"] == 100
  # With a=1, the exact endpoints are 97, 6, and 103; no 16-step averaging.
  np.testing.assert_allclose(stages["early"]["metrics"]["corr"]["P0"]["J"], 97**2)
  np.testing.assert_allclose(stages["late"]["metrics"]["corr"]["P0"]["J"], 6**2)
  np.testing.assert_allclose(stages["full"]["metrics"]["corr"]["P0"]["J"], 103**2)


def _fake_stage_payload(offset: float):
  paths = {}
  for path_index, path in enumerate(PATHS):
    paths[path] = {
        "C_corr": offset + path_index,
        "C_iid": offset + path_index + 1,
        "J_corr": 2 * offset + path_index,
        "J_iid": 2 * offset + path_index + 2,
        "D_corr": 3 * offset + path_index,
        "D_iid": 3 * offset + path_index + 3,
        "G_C": .1 * offset + path_index,
        "G_J": .2 * offset + path_index,
    }
  decomposition = {
      "A_energy": offset + 1, "B_energy": offset + 2, "I_energy": offset + 3,
      "AB_dot": offset + 4, "AI_dot": offset + 5, "BI_dot": offset + 6,
      "reconstruction_error": offset / 100,
  }
  degradation = {
      gain: {field: offset + index for index, field in enumerate(
          ("delta_clean", "delta_bias", "delta_nonlinear")
      )}
      for gain in ("G_C", "G_J")
  }
  return {"paths": paths, "decomposition": decomposition, "degradation": degradation}


def test_cross_seed_aggregate_mean_std_includes_decomposition():
  aggregate = cross_seed_aggregate({
      "0": {"early": _fake_stage_payload(1.0)},
      "1": {"early": _fake_stage_payload(3.0)},
      "2": {"early": _fake_stage_payload(5.0)},
  })
  early = aggregate["early"]
  np.testing.assert_allclose(early["paths"]["P0"]["G_C_mean"], 0.3)
  np.testing.assert_allclose(early["paths"]["P0"]["G_C_std"], .2)
  np.testing.assert_allclose(early["paths"]["P3"]["G_J_mean"], 3.6)
  np.testing.assert_allclose(early["decomposition"]["A_energy"]["mean"], 4.0)
  np.testing.assert_allclose(early["decomposition"]["A_energy"]["std"], 2.0)
  np.testing.assert_allclose(early["decomposition_flat"]["BI_dot_std"], 2.0)


def test_plot_helpers_consume_cross_seed_means_and_uncertainty(tmp_path, monkeypatch):
  from matplotlib.axes import Axes
  from exp8 import plotting

  aggregate = cross_seed_aggregate({
      "0": {"early": _fake_stage_payload(1.0)},
      "1": {"early": _fake_stage_payload(3.0)},
  })
  captured = []
  original = Axes.errorbar

  def record(self, x, y, yerr=None, *args, **kwargs):
    captured.append((list(y), list(yerr)))
    return original(self, x, y, yerr=yerr, *args, **kwargs)

  monkeypatch.setattr(Axes, "errorbar", record)
  plotting.plot_path_gain_summary(aggregate, tmp_path / "path.png")
  plotting.plot_decomposition(aggregate, tmp_path / "decomp.png")
  assert captured
  # The first path summary point is the cross-seed mean (0.2), not seed 0 (0.1).
  np.testing.assert_allclose(captured[0][0][0], .2)
  np.testing.assert_allclose(captured[0][1][0], np.sqrt(.02))
  # Decomposition A mean is 3, with sample std sqrt(2).
  np.testing.assert_allclose(captured[2][0][0], 3.0)
  np.testing.assert_allclose(captured[2][1][0], np.sqrt(2.0))


def test_smoke_outputs_are_finite(tmp_path):
  from exp8.run import run_smoke

  run_smoke(tmp_path, [0])
  expected = {
      "window_diagnostics_seed0.csv", "window_summary.csv", "summary.json",
      "correlation_gain_over_steps.png", "endpoint_gain_over_steps.png",
      "path_gain_summary.png", "privacy_clean_decomposition.png",
  }
  assert expected.issubset({path.name for path in tmp_path.iterdir()})
  data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
  assert data["training_bandinv_strategy"]["workload_type"] == "decayed-prefix-sum"
  diagnostic_metadata = data["diagnostic_bandinv_strategy"]
  assert diagnostic_metadata["workload_type"] == "adam-first-moment-aware"
  assert diagnostic_metadata["workload_representation"] == "matrix"
  assert data["training_privacy_calibration"] != data["diagnostic_privacy_calibration"]
  assert len(data["phi_t"]) == diagnostic_metadata["horizon"]

  def visit(value):
    if isinstance(value, float):
      assert math.isfinite(value)
    elif isinstance(value, dict):
      for item in value.values():
        visit(item)
    elif isinstance(value, list):
      for item in value:
        visit(item)

  visit(data)
