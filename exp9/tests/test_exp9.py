"""Focused regression tests for Experiment 9's nonlinear decomposition."""

from types import SimpleNamespace
import json
import pytest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import (
    BandInvMFStrategy, filter_latent_noise, init_bandinv_noise_state,
)
from dp_muon.optim import (
    classic_nesterov_momentum, fixed_lr_nesterov_decayed_trajectory_workload_coef,
    muon_q_stages, nesterov_kernel_coef,
)
from exp9.core import (
    BRANCHES,
    PRIMARY_STAGES,
    PATHS,
    advance_exp9_diagnostic,
    bandinv_marginal_variances,
    classic_nesterov_frontend,
    estimate_output_bias,
    estimate_output_bias_replicates,
    extract_muon_blocks,
    init_exp9_shadow_state,
    muon_parameter_paths,
    nonlinear_response_decomposition,
    paired_diagnostic_noise_from_innovation,
    pre_q_marginal_variances,
    init_exp9_train_state,
    linear_frontend,
    make_exp9_train_step,
    smooth_muon_q,
)
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from exp9.diagnostics import (
    cancellation_statistics, cancellation_metrics_from_jd, cross_seed_aggregate,
    degradation, safe_ratio,
)
from exp9.online_shadow import Exp9WindowCollector
from exp9.run import _json_safe, _stage_payload


def _strategy(horizon=8):
  coef = jnp.asarray([1.0, .5], jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=coef, min_sep=1, max_participations=1
  )
  return BandInvMFStrategy(
      horizon=horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=jnp.ones((horizon,), jnp.float32), noising_coef=coef,
      strategy_coef=jnp.ones((horizon,), jnp.float32), sensitivity_squared=sensitivity,
      objective=jnp.asarray(1.),
  )


def _matrix(value=0.0):
  return jnp.asarray([[1.2 + value, .2], [.1, 1.7 - value]], jnp.float32)


def test_frontend_matches_production_classic_nesterov():
  gradient = _matrix()
  previous = jnp.zeros_like(gradient)
  _, expected = classic_nesterov_frontend(previous, gradient, .95)
  transform = classic_nesterov_momentum(.95)
  update, _ = transform.update(gradient, transform.init(gradient))
  np.testing.assert_allclose(expected, update, rtol=0, atol=2e-7)


def test_paired_variance_and_pre_q_variance_match_construction():
  strategy = _strategy()
  raw = np.asarray(bandinv_marginal_variances(strategy, 2.0))
  expected_raw = 4.0 * np.asarray([1.0, 1.25])[np.minimum(np.arange(8), 1)]
  np.testing.assert_allclose(raw, expected_raw)
  values = pre_q_marginal_variances(strategy, 2.0, .5)
  assert set(values) == {"raw_corr", "raw_iid", "pre_q_corr", "pre_q_iid"}
  assert np.all(np.asarray(values["pre_q_corr"]) >= 0)
  assert np.all(np.asarray(values["pre_q_iid"]) >= 0)


def test_jvp_finite_difference_and_antithetic_odd_sign():
  x, r = _matrix(), jnp.asarray([[.04, -.02], [.01, .03]], jnp.float32)
  result = nonlinear_response_decomposition(x, r, ns_steps=3, consistent_rms=.2)
  def F(matrix):
    return smooth_muon_q(matrix, ns_steps=3, consistent_rms=.2)
  _, jvp = jax.jvp(F, (x,), (r,))
  np.testing.assert_allclose(result["P1"], jvp, rtol=0, atol=0)
  central_errors = []
  for delta in (1e-2, 3e-3, 1e-3):
    f_plus = F(x + delta * r)
    f_minus = F(x - delta * r)
    central = (f_plus - f_minus) / (2.0 * delta)
    central_errors.append(float(np.linalg.norm(np.asarray(result["P1"] - central))))
  assert np.all(np.isfinite(central_errors))
  assert max(central_errors) < 1e-2
  opposite = nonlinear_response_decomposition(x, -r, ns_steps=3, consistent_rms=.2)
  np.testing.assert_allclose(result["P2"], -opposite["P2"], rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(result["Y"], result["P2"] + result["even"], rtol=0, atol=2e-6)


def test_bias_probe_is_zero_for_zero_rho_and_configurable():
  x = _matrix()
  zero = estimate_output_bias(x, 0.0, jax.random.key(10), probes=2, ns_steps=3)
  np.testing.assert_allclose(zero, 0.0, atol=0)
  one = estimate_output_bias(x, .1, jax.random.key(10), probes=1, ns_steps=3)
  eight = estimate_output_bias(x, .1, jax.random.key(10), probes=8, ns_steps=3)
  assert np.all(np.isfinite(np.asarray(one)))
  assert np.all(np.isfinite(np.asarray(eight)))


def test_bias_replicates_report_disagreement_and_share_rho():
  x = _matrix()
  estimate, bias_a, bias_b = estimate_output_bias_replicates(
      x, .1, jax.random.key(10), probes=4, ns_steps=3
  )
  np.testing.assert_allclose(estimate, (bias_a + bias_b) / 2, rtol=0, atol=0)
  disagreement = np.linalg.norm(np.asarray(bias_a - bias_b))
  assert np.isfinite(disagreement) and disagreement >= 0.0


def test_same_z_is_used_for_correlated_and_iid_controls():
  strategy = _strategy()
  state = init_bandinv_noise_state({"block": jnp.zeros((2, 2), jnp.float32)}, 2)
  z = {"block": jnp.asarray([[1.0, -2.0], [.5, 3.0]], jnp.float32)}
  corr, iid, next_state = paired_diagnostic_noise_from_innovation(
      state, z, strategy, iid_noise_std=2.0, raw_marginal_variance=4.0
  )
  np.testing.assert_allclose(np.asarray(iid["block"]), np.asarray(2.0 * z["block"]))
  np.testing.assert_allclose(np.asarray(corr["block"]), np.asarray(2.0 * z["block"]))
  assert int(next_state.step) == 1


def test_correlated_and_iid_raw_noise_marginals_match_empirically():
  horizon, samples = 8, 20000
  strategy = _strategy(horizon)
  sigma = 1.7
  coef = np.asarray(strategy.noising_coef)
  rng = np.random.default_rng(9)
  z = rng.standard_normal((samples, horizon))
  corr = np.zeros_like(z)
  for t in range(horizon):
    for lag in range(min(t + 1, len(coef))):
      corr[:, t] += sigma * coef[lag] * z[:, t - lag]
  raw = np.asarray(bandinv_marginal_variances(strategy, sigma))
  iid = np.sqrt(raw)[None, :] * z
  np.testing.assert_allclose(np.var(corr, axis=0), raw, rtol=.06, atol=.06)
  np.testing.assert_allclose(np.var(iid, axis=0), raw, rtol=.06, atol=.06)


def test_pre_q_marginals_match_empirical_frontend_filter():
  horizon, samples = 8, 30000
  strategy = _strategy(horizon)
  sigma, momentum = 1.3, .6
  theoretical = pre_q_marginal_variances(strategy, sigma, momentum)
  coef = np.asarray(strategy.noising_coef)
  h = np.asarray(nesterov_kernel_coef(horizon, momentum))
  rng = np.random.default_rng(12)
  z = rng.standard_normal((samples, horizon))
  raw_corr = np.zeros_like(z)
  for t in range(horizon):
    for lag in range(min(t + 1, len(coef))):
      raw_corr[:, t] += sigma * coef[lag] * z[:, t - lag]
  raw_iid = np.sqrt(np.asarray(theoretical["raw_iid"]))[None, :] * z
  corr_pre = np.zeros_like(z)
  iid_pre = np.zeros_like(z)
  for t in range(horizon):
    for lag in range(t + 1):
      corr_pre[:, t] += h[lag] * raw_corr[:, t - lag]
      iid_pre[:, t] += h[lag] * raw_iid[:, t - lag]
  np.testing.assert_allclose(np.var(corr_pre, axis=0), np.asarray(theoretical["pre_q_corr"]), rtol=.08, atol=.08)
  np.testing.assert_allclose(np.var(iid_pre, axis=0), np.asarray(theoretical["pre_q_iid"]), rtol=.08, atol=.08)


def test_nesterov_impulse_and_workload_are_exact():
  horizon, momentum, eta, decay = 9, .7, .03, .04
  impulse = [jnp.asarray(1.0)] + [jnp.asarray(0.0)] * (horizon - 1)
  _, frontend = linear_frontend(impulse, momentum)
  h = np.asarray(nesterov_kernel_coef(horizon, momentum))
  np.testing.assert_allclose(np.asarray(frontend).reshape(-1), h, rtol=0, atol=2e-7)
  rho = 1.0 - eta * decay
  trajectory = []
  value = 0.0
  for update in h:
    value = rho * value - eta * update
    trajectory.append(value)
  workload = np.asarray(fixed_lr_nesterov_decayed_trajectory_workload_coef(
      horizon, momentum, eta, decay
  ))
  np.testing.assert_allclose(-np.asarray(trajectory), workload, rtol=0, atol=2e-7)


def test_primary_blocks_follow_vit_labels_only():
  params = {
      "blocks": ({"attention": {"query": {"kernel": _matrix()}}},),
      "head": {"kernel": _matrix(.3)},
  }
  paths = muon_parameter_paths(params)
  blocks = extract_muon_blocks(params, paths)
  assert list(blocks) == ["blocks/0/attention/query/kernel"]


def test_aggregation_stays_muon_only_even_when_params_have_head():
  params = {
      "blocks": ({"attention": {"query": {"kernel": _matrix()}}},),
      "head": {"kernel": _matrix(.3)},
  }
  blocks = extract_muon_blocks(params, muon_parameter_paths(params))
  zeros = {key: jnp.zeros_like(value) for key, value in blocks.items()}
  _, step = advance_exp9_diagnostic(
      init_exp9_shadow_state(blocks), blocks, zeros, zeros,
      {"corr": 0.0, "iid": 0.0}, jax.random.key(31), momentum=.9,
      learning_rate=.01, probes=2, secondary_use_bf16_ns=False,
  )
  collector = Exp9WindowCollector(
      params, seed=0, learning_rate=.01, weight_decay=0.0, horizon=1,
  )
  collector.after_step(SimpleNamespace(step=jnp.asarray(1), last_step=step), 1)
  row = collector.finalize()[0]
  assert all("head" not in key for key in row)


def test_rng_streams_for_training_diagnostic_and_bias_are_independent():
  params = {"blocks": ({"attention": {"query": {"kernel": _matrix()}}},)}
  strategy = _strategy()
  state_a = init_exp9_train_state(
      params, strategy, jax.random.key(1), optax.sgd(.1), jax.random.key(2),
      bias_probe_rng_key=jax.random.key(3), diagnostic_strategy=strategy,
  )
  state_b = init_exp9_train_state(
      params, strategy, jax.random.key(4), optax.sgd(.1), jax.random.key(2),
      bias_probe_rng_key=jax.random.key(3), diagnostic_strategy=strategy,
  )
  np.testing.assert_array_equal(
      jax.random.key_data(state_a.diagnostic_rng_key),
      jax.random.key_data(state_b.diagnostic_rng_key),
  )
  np.testing.assert_array_equal(
      jax.random.key_data(state_a.bias_probe_rng_key),
      jax.random.key_data(state_b.bias_probe_rng_key),
  )
  assert not np.array_equal(
      np.asarray(jax.random.key_data(state_a.training_rng_key)),
      np.asarray(jax.random.key_data(state_a.diagnostic_rng_key)),
  )
  assert not np.array_equal(
      np.asarray(jax.random.key_data(state_a.training_rng_key)),
      np.asarray(jax.random.key_data(state_a.bias_probe_rng_key)),
  )


def _tree_allclose(actual, expected):
  assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
  ):
    try:
      actual_key = jax.random.key_data(actual_leaf)
      expected_key = jax.random.key_data(expected_leaf)
    except (TypeError, ValueError):
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-7)
    else:
      np.testing.assert_array_equal(actual_key, expected_key)


def test_diagnostic_and_bias_rngs_cannot_change_three_step_real_trajectory():
  horizon = 4
  strategy = _strategy(horizon)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=1.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(horizon, 1, 1)

  def loss(params, batch):
    matrix = params["blocks"][0]["attention"]["query"]["kernel"]
    return jnp.sum(matrix) * jnp.sum(jnp.asarray(batch["scale"]))

  train_step, optimizer = make_exp9_train_step(
      loss, strategy, calibration, participation,
      diagnostic_strategy=strategy, diagnostic_calibration=calibration,
      muon_learning_rate=.01, muon_weight_decay=0.0, momentum=.9,
      ns_steps=2, consistent_rms=.2, adamw_learning_rate=.01,
      adamw_beta1=.9, adamw_beta2=.99, adamw_eps=1e-6,
      adamw_weight_decay=0.0, probes=2, secondary_use_bf16_ns=False,
  )
  params = {"blocks": ({"attention": {"query": {"kernel": _matrix()}}},)}

  def initial(diagnostic_seed, bias_seed):
    return init_exp9_train_state(
        params, strategy, jax.random.key(101), optimizer,
        jax.random.key(diagnostic_seed), bias_probe_rng_key=jax.random.key(bias_seed),
        diagnostic_strategy=strategy,
    )

  states = [initial(202, 303), initial(404, 303), initial(202, 505)]
  batches = [
      {"scale": jnp.asarray([[1.0 + .1 * step]], jnp.float32)}
      for step in range(3)
  ]
  for batch in batches:
    states = [train_step(state, batch) for state in states]
    baseline, diagnostic_changed, bias_changed = states
    for candidate in (diagnostic_changed, bias_changed):
      _tree_allclose(candidate.params, baseline.params)
      _tree_allclose(candidate.optimizer_state, baseline.optimizer_state)
      _tree_allclose(candidate.training_noise_state, baseline.training_noise_state)
      _tree_allclose(candidate.training_rng_key, baseline.training_rng_key)
      _tree_allclose(candidate.step, baseline.step)


def test_p0_uses_fixed_muon_block_scale():
  blocks = {"wide": jnp.ones((2, 3), jnp.float32), "tall": jnp.ones((4, 1), jnp.float32)}
  state = init_exp9_shadow_state(blocks)
  noise = {"wide": jnp.full((2, 3), .1), "tall": jnp.full((4, 1), -.2)}
  _, step = advance_exp9_diagnostic(
      state, blocks, noise, noise, {"corr": .04, "iid": .04}, jax.random.key(7),
      momentum=0.0, learning_rate=.05, consistent_rms=.3, ns_steps=2, probes=2,
      secondary_use_bf16_ns=False,
  )
  np.testing.assert_allclose(
      np.asarray(step.x["corr"]["P0"]["wide"] / -.05),
      .3 * np.sqrt(3.0) * np.asarray(noise["wide"]), rtol=0, atol=1e-7,
  )
  np.testing.assert_allclose(
      np.asarray(step.x["corr"]["P0"]["tall"] / -.05),
      .3 * np.sqrt(4.0) * np.asarray(noise["tall"]), rtol=0, atol=1e-7,
  )


def test_stage_attribution_uses_fixed_scale_for_each_block():
  blocks = {
      "wide": jnp.asarray([[1.0, .2, -.4], [.3, 1.5, .7]], jnp.float32),
      "tall": jnp.asarray([[1.2], [-.3], [.8], [1.7]], jnp.float32),
  }
  noise = {
      "wide": jnp.asarray([[.1, -.2, .05], [.03, .04, -.1]], jnp.float32),
      "tall": jnp.asarray([[.2], [-.05], [.1], [-.15]], jnp.float32),
  }
  _, step = advance_exp9_diagnostic(
      init_exp9_shadow_state(blocks), blocks, noise, noise,
      {"corr": .04, "iid": .04}, jax.random.key(71), momentum=0.0,
      learning_rate=.05, consistent_rms=.3, ns_steps=3, probes=2,
      secondary_use_bf16_ns=False,
  )
  for key, value in noise.items():
    scale = .3 * np.sqrt(max(value.shape))
    plus = muon_q_stages(
        blocks[key] + value, ns_steps=3, consistent_rms=.3,
        use_bf16_ns=False,
    )
    minus = muon_q_stages(
        blocks[key] - value, ns_steps=3, consistent_rms=.3,
        use_bf16_ns=False,
    )
    odd = {stage: (plus[stage] - minus[stage]) * .5 for stage in plus}
    np.testing.assert_allclose(
        np.asarray(step.stage_odd["corr"]["linear"][key]),
        scale * np.asarray(value), rtol=0, atol=1e-6,
    )
    for stage in ("norm", "ns"):
      np.testing.assert_allclose(
          np.asarray(step.stage_odd["corr"][stage][key]),
          scale * np.asarray(odd[stage]), rtol=0, atol=1e-6,
      )
    np.testing.assert_allclose(
        np.asarray(step.stage_odd["corr"]["scale"][key]),
        np.asarray(odd["scale"]), rtol=0, atol=1e-6,
    )


def test_stage_attribution_is_invariant_to_single_block_scale():
  block = {"block": jnp.asarray([[1.0, -.25], [.5, 1.4]], jnp.float32)}
  noises = [
      {"block": jnp.asarray([[.1, -.2], [.03, .05]], jnp.float32)},
      {"block": jnp.asarray([[-.04, .08], [.06, -.02]], jnp.float32)},
  ]

  def responses(consistent_rms):
    state = init_exp9_shadow_state(block)
    values = {stage: [] for stage in PRIMARY_STAGES}
    for index, noise in enumerate(noises):
      state, step = advance_exp9_diagnostic(
          state, block, noise, noise, {"corr": .04, "iid": .04},
          jax.random.key(73 + index), momentum=.3, learning_rate=.05,
          consistent_rms=consistent_rms, ns_steps=3, probes=2,
          secondary_use_bf16_ns=False,
      )
      for stage in PRIMARY_STAGES:
        values[stage].append(np.asarray(step.stage_odd["corr"][stage]["block"]))
    return {stage: np.stack(stage_values) for stage, stage_values in values.items()}

  low, high = responses(.2), responses(.4)
  for stage in PRIMARY_STAGES:
    low_stats = cancellation_statistics(low[stage], weight_decay=.07, learning_rate=.05)
    high_stats = cancellation_statistics(high[stage], weight_decay=.07, learning_rate=.05)
    np.testing.assert_allclose(high_stats["C"], low_stats["C"], rtol=2e-5, atol=2e-5)


def test_secondary_bf16_stage_attribution_uses_same_block_scale():
  blocks = {"block": jnp.asarray([[1.0, .2, -.4], [.3, 1.5, .7]], jnp.float32)}
  noise = {"block": jnp.asarray([[.1, -.2, .05], [.03, .04, -.1]], jnp.float32)}
  _, step = advance_exp9_diagnostic(
      init_exp9_shadow_state(blocks), blocks, noise, noise,
      {"corr": .04, "iid": .04}, jax.random.key(74), momentum=0.0,
      learning_rate=.05, consistent_rms=.3, ns_steps=3, probes=2,
      secondary_use_bf16_ns=True,
  )
  plus = muon_q_stages(
      blocks["block"] + noise["block"], ns_steps=3, consistent_rms=.3,
      use_bf16_ns=True,
  )
  minus = muon_q_stages(
      blocks["block"] - noise["block"], ns_steps=3, consistent_rms=.3,
      use_bf16_ns=True,
  )
  odd = {stage: (plus[stage] - minus[stage]) * .5 for stage in plus}
  scale = .3 * np.sqrt(3.0)
  for stage in ("linear", "norm", "ns"):
    np.testing.assert_allclose(
        np.asarray(step.secondary_stage_odd["corr"][stage]["block"]),
        scale * np.asarray(odd[stage]), rtol=0, atol=1e-6,
    )
  np.testing.assert_allclose(
      np.asarray(step.secondary_stage_odd["corr"]["scale"]["block"]),
      np.asarray(odd["scale"]), rtol=0, atol=1e-6,
  )


def test_stage_scale_and_ns_match_when_final_scale_is_one():
  block = {"block": jnp.asarray([[1.0, -.25], [.5, 1.4]], jnp.float32)}
  noise = {"block": jnp.asarray([[.1, -.2], [.03, .05]], jnp.float32)}
  _, step = advance_exp9_diagnostic(
      init_exp9_shadow_state(block), block, noise, noise,
      {"corr": .04, "iid": .04}, jax.random.key(72), momentum=0.0,
      learning_rate=.05, consistent_rms=1 / np.sqrt(2), ns_steps=3,
      probes=2, secondary_use_bf16_ns=False,
  )
  np.testing.assert_allclose(
      np.asarray(step.stage_odd["corr"]["ns"]["block"]),
      np.asarray(step.stage_odd["corr"]["scale"]["block"]),
      rtol=2e-5, atol=2e-5,
  )


def test_same_probe_same_rho_gives_same_bias_hat_for_both_branches():
  blocks = {"block": _matrix()}
  state = init_exp9_shadow_state(blocks)
  corr = {"block": jnp.asarray([[.2, -.1], [.04, .08]], jnp.float32)}
  iid = {"block": jnp.asarray([[-.3, .1], [.02, -.07]], jnp.float32)}
  _, step = advance_exp9_diagnostic(
      state, blocks, corr, iid, {"corr": .09, "iid": .09}, jax.random.key(8),
      momentum=.4, learning_rate=.01, ns_steps=3, probes=4,
      secondary_use_bf16_ns=False,
  )
  np.testing.assert_array_equal(
      np.asarray(step.bias["corr"]["block"]), np.asarray(step.bias["iid"]["block"])
  )
  np.testing.assert_array_equal(
      np.asarray(step.bias_A["corr"]["block"]), np.asarray(step.bias_A["iid"]["block"])
  )


def test_zero_noise_has_zero_all_decomposition_paths():
  blocks = {"block": _matrix()}
  zeros = {"block": jnp.zeros_like(blocks["block"])}
  _, step = advance_exp9_diagnostic(
      init_exp9_shadow_state(blocks), blocks, zeros, zeros,
      {"corr": 0.0, "iid": 0.0}, jax.random.key(22), momentum=.9,
      learning_rate=.01, probes=2, secondary_use_bf16_ns=False,
  )
  for branch in BRANCHES:
    for path in PATHS:
      np.testing.assert_allclose(np.asarray(step.x[branch][path]["block"]), 0.0, atol=0)
    for stage in ("linear", "norm", "ns", "scale"):
      np.testing.assert_allclose(np.asarray(step.stage_odd[branch][stage]["block"]), 0.0, atol=0)


def test_advance_zero_noise_returns_zero_primary_paths():
  blocks = {"block": _matrix()}
  state = init_exp9_shadow_state(blocks)
  zeros = {"block": jnp.zeros_like(blocks["block"])}
  state, step = advance_exp9_diagnostic(
      state, blocks, zeros, zeros, {"corr": 0.0, "iid": 0.0}, jax.random.key(2),
      momentum=.9, learning_rate=.01, ns_steps=3, probes=2,
      secondary_use_bf16_ns=False,
  )
  for branch in BRANCHES:
    for path in PATHS:
      np.testing.assert_allclose(np.asarray(step.x[branch][path]["block"]), 0.0, atol=0)
    np.testing.assert_allclose(np.asarray(step.bias[branch]["block"]), 0.0, atol=0)


def test_collector_uses_exact_stage_endpoints():
  block = {"block": np.zeros((1,), np.float32)}
  def fake_step(value):
    x = {branch: {path: {"block": np.asarray([0.0], np.float32)
                         for _ in [0]} for path in PATHS} for branch in BRANCHES}
    x["corr"]["P0"]["block"] = np.asarray([value], np.float32)
    empty = {"block": np.zeros((1,), np.float32)}
    empty_branch = {branch: {"block": np.zeros((1,), np.float32)} for branch in BRANCHES}
    stages = {branch: {stage: {"block": np.zeros((1,), np.float32)}
                       for stage in ("linear", "bf16", "norm", "ns", "scale")}
              for branch in BRANCHES}
    return SimpleNamespace(
        x=x, clean_pre_q=empty, noise_pre_q=empty_branch, raw_response=empty_branch,
        bias=empty_branch, even_response=empty_branch, stage_odd=stages,
        secondary_stage_odd=stages, probe_disagreement=empty_branch,
        block_ratio_mean={branch: 0.0 for branch in BRANCHES},
        block_ratio_max={branch: 0.0 for branch in BRANCHES},
        global_noise_signal_ratio={branch: 0.0 for branch in BRANCHES},
        clean_pre_q_norm={"block": 1.0}, clean_pre_q_norm_min=1.0,
        odd_reconstruction_error={branch: 0.0 for branch in BRANCHES},
    )
  collector = Exp9WindowCollector(block, seed=0, learning_rate=0.0,
                                   weight_decay=0.0, horizon=100)
  for step in range(1, 101):
    collector.after_step(SimpleNamespace(step=jnp.asarray(step), last_step=fake_step(1.0 if step <= 97 else 2.0)), step)
  stages = collector.stage_summaries()
  assert stages["early"]["num_steps"] == 97
  assert stages["late"]["num_steps"] == 3
  assert stages["full"]["num_steps"] == 100
  np.testing.assert_allclose(stages["early"]["metrics"]["corr"]["P0"]["J"], 97**2)
  np.testing.assert_allclose(stages["late"]["metrics"]["corr"]["P0"]["J"], 6**2)
  np.testing.assert_allclose(stages["full"]["metrics"]["corr"]["P0"]["J"], 103**2)
  assert set(stages["full"]["stage_metrics"]["corr"]) == set(PRIMARY_STAGES)
  assert stages["full"]["bias"]["P3_reliable_corr"] is False
  assert "global_noise_signal_ratio_mean_corr" in stages["full"]["bias"]


def _synthetic_reliability_step(
    p3_signal, delta_b, bias=0.0, *, iid_p3_signal=None,
    iid_delta_b=None, iid_bias=None,
):
  x = {
      branch: {path: {"block": np.asarray([0.0], np.float32)} for path in PATHS}
      for branch in BRANCHES
  }
  branch_p3 = {"corr": p3_signal, "iid": p3_signal if iid_p3_signal is None else iid_p3_signal}
  branch_delta = {"corr": delta_b, "iid": delta_b if iid_delta_b is None else iid_delta_b}
  branch_bias = {"corr": bias, "iid": bias if iid_bias is None else iid_bias}
  for branch in BRANCHES:
    x[branch]["P3"]["block"] = np.asarray([branch_p3[branch]], np.float32)
  empty = {"block": np.zeros((1,), np.float32)}
  empty_branch = {branch: {"block": np.zeros((1,), np.float32)} for branch in BRANCHES}
  stages = {
      branch: {
          stage: {"block": np.zeros((1,), np.float32)}
          for stage in ("linear", "bf16", "norm", "ns", "scale")
      } for branch in BRANCHES
  }
  probe = {
      branch: {"block": np.asarray([branch_delta[branch]], np.float32)}
      for branch in BRANCHES
  }
  bias_tree = {
      branch: {"block": np.asarray([branch_bias[branch]], np.float32)}
      for branch in BRANCHES
  }
  return SimpleNamespace(
      x=x, clean_pre_q=empty, noise_pre_q=empty_branch,
      raw_response=empty_branch, bias=bias_tree, even_response=empty_branch,
      stage_odd=stages, secondary_stage_odd=stages,
      probe_disagreement=probe,
      block_ratio_mean={branch: 0.0 for branch in BRANCHES},
      block_ratio_max={branch: 0.0 for branch in BRANCHES},
      global_noise_signal_ratio={branch: 0.0 for branch in BRANCHES},
      clean_pre_q_norm={"block": 1.0}, clean_pre_q_norm_min=1.0,
      odd_reconstruction_error={branch: 0.0 for branch in BRANCHES},
  )


def _collect_reliability_stage(
    p3_signal, delta_b, bias=0.0, *, weight_decay=0.0,
    iid_p3_signal=None, iid_delta_b=None, iid_bias=None, horizon=1,
):
  collector = Exp9WindowCollector(
      {"block": np.zeros((1,), np.float32)}, seed=0,
      learning_rate=.1, weight_decay=weight_decay, horizon=horizon,
  )
  for step_index in range(1, horizon + 1):
    step = _synthetic_reliability_step(
        p3_signal[step_index - 1] if isinstance(p3_signal, (list, tuple)) else p3_signal,
        delta_b[step_index - 1] if isinstance(delta_b, (list, tuple)) else delta_b,
        bias, iid_p3_signal=(
            iid_p3_signal[step_index - 1]
            if isinstance(iid_p3_signal, (list, tuple)) else iid_p3_signal
        ), iid_delta_b=(
            iid_delta_b[step_index - 1]
            if isinstance(iid_delta_b, (list, tuple)) else iid_delta_b
        ), iid_bias=iid_bias,
    )
    collector.after_step(
        SimpleNamespace(step=jnp.asarray(step_index), last_step=step), step_index
    )
  return collector.stage_summaries()["full"]


def _collect_reliability(p3_signal, delta_b, bias=0.0, **kwargs):
  return _collect_reliability_stage(p3_signal, delta_b, bias, **kwargs)["bias"]


def test_probe_error_energy_uses_weight_decay_recurrence():
  p3_values = [2.0, 3.0, 4.0]
  delta_values = [.2, .3, .4]
  eta, weight_decay = .1, .5
  a = 1.0 - eta * weight_decay
  expected_energy = 0.0
  old_no_decay_energy = 0.0
  p3_D = 0.0
  for p3, delta in zip(p3_values, delta_values, strict=True):
    expected_energy = a * a * expected_energy + .25 * eta * eta * delta * delta
    old_no_decay_energy += .25 * eta * eta * delta * delta
    p3_D = a * a * p3_D + p3 * p3
  result = _collect_reliability(
      p3_values, delta_values, weight_decay=weight_decay, horizon=3,
  )
  expected_ratio = expected_energy / p3_D
  old_ratio = old_no_decay_energy / p3_D
  np.testing.assert_allclose(result["probe_error_to_P3_D_corr"], expected_ratio)
  assert not np.isclose(result["probe_error_to_P3_D_corr"], old_ratio)


@pytest.mark.parametrize(
    ("corr_ok", "iid_ok"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_p3_reliability_has_paired_boolean_for_all_branch_combinations(corr_ok, iid_ok):
  good = (10.0, .01)
  bad = (.01, 10.0)
  corr_values = good if corr_ok else bad
  iid_values = good if iid_ok else bad
  result = _collect_reliability(
      corr_values[0], corr_values[1], iid_p3_signal=iid_values[0],
      iid_delta_b=iid_values[1],
  )
  assert result["P3_reliable_corr"] is corr_ok
  assert result["P3_reliable_iid"] is iid_ok
  assert result["P3_reliable_paired"] is (corr_ok and iid_ok)


def test_delta_even_interpretation_and_cross_seed_count_use_paired_reliability():
  reliable_stage = _stage_payload(_collect_reliability_stage(10.0, .01))
  mixed_stage = _stage_payload(_collect_reliability_stage(
      10.0, .01, iid_p3_signal=.01, iid_delta_b=10.0
  ))
  assert reliable_stage["P3_reliable"] == {
      "corr": True, "iid": True, "paired": True,
      "rule": "probe_error_to_P3_D <= 0.1 AND probe_error_to_P3_endpoint <= 0.1",
  }
  assert reliable_stage["delta_even_interpretation"] == "reliable"
  assert mixed_stage["P3_reliable"]["paired"] is False
  assert mixed_stage["delta_even_interpretation"].startswith("unreliable due to")
  aggregate = cross_seed_aggregate({
      "0": {"full": reliable_stage}, "1": {"full": mixed_stage},
  })
  assert aggregate["full"]["paired_reliability"] == {
      "paired_reliable_seed_count": 1, "paired_reliable_fraction": .5,
  }


def test_p3_reliability_uses_probe_error_relative_to_p3_signal():
  zero = _collect_reliability(1.0, 0.0)
  assert zero["probe_error_to_P3_D_corr"] == 0.0
  assert zero["probe_error_to_P3_endpoint_corr"] == 0.0
  assert zero["P3_reliable_corr"] is True
  assert zero["P3_reliable_paired"] is True

  small = _collect_reliability(10.0, .01)
  assert small["probe_error_to_P3_D_corr"] < .1
  assert small["probe_error_to_P3_endpoint_corr"] < .1
  assert small["P3_reliable_corr"] is True
  assert small["P3_reliable_paired"] is True

  large = _collect_reliability(.01, 10.0)
  assert large["probe_error_to_P3_D_corr"] > .1
  assert large["probe_error_to_P3_endpoint_corr"] > .1
  assert large["P3_reliable_corr"] is False
  assert large["P3_reliable_paired"] is False


def test_large_relative_to_bias_disagreement_does_not_override_p3_signal_rule():
  result = _collect_reliability(10.0, 1.0, bias=1e-10)
  assert result["probe_disagreement_relative_to_bias_corr"] > 1e6
  assert result["P3_reliable_corr"] is True
  assert result["P3_reliable_paired"] is True


def test_p3_zero_signal_makes_reliability_invalid_and_false():
  result = _collect_reliability(0.0, 0.0)
  assert result["probe_error_to_P3_D_corr"] is None
  assert result["probe_error_to_P3_endpoint_corr"] is None
  assert result["P3_reliable_corr"] is False
  assert result["P3_reliable_iid"] is False
  assert result["P3_reliable_paired"] is False


def test_invalid_p3_denominator_in_either_branch_forces_paired_false():
  corr_invalid = _collect_reliability(
      0.0, 0.0, iid_p3_signal=10.0, iid_delta_b=.01,
  )
  iid_invalid = _collect_reliability(
      10.0, .01, iid_p3_signal=0.0, iid_delta_b=0.0,
  )
  assert corr_invalid["P3_reliable_corr"] is False
  assert corr_invalid["P3_reliable_iid"] is True
  assert corr_invalid["P3_reliable_paired"] is False
  assert iid_invalid["P3_reliable_corr"] is True
  assert iid_invalid["P3_reliable_iid"] is False
  assert iid_invalid["P3_reliable_paired"] is False


def test_degradation_names_and_reference_statistics():
  np.testing.assert_allclose(
      list(degradation({"P0": .8, "P1": .6, "P2": .5, "P3": .2}).values()),
      [.2, .1, .3],
  )
  np.testing.assert_allclose(
      cancellation_statistics(np.asarray([[1.], [2.]]), weight_decay=0., learning_rate=.1)["J"], 9.
  )


def test_invalid_ratios_are_explicitly_invalid_not_zero():
  assert safe_ratio(0.0, 0.0) is None
  assert cancellation_metrics_from_jd(0.0, 0.0)["valid"] is False
  with pytest.raises(ValueError, match="non-finite"):
    safe_ratio(1.0, np.nan)


def test_json_serialization_uses_null_for_invalid_reliability_values():
  safe = _json_safe({"probe_error_to_P3_D": float("nan"), "valid": False})
  assert safe == {"probe_error_to_P3_D": None, "valid": False}
  assert json.dumps(safe, allow_nan=False) == (
      '{"probe_error_to_P3_D": null, "valid": false}'
  )
