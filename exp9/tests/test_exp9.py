"""Focused regression tests for Experiment 9's nonlinear decomposition."""

from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import classic_nesterov_momentum
from exp9.core import (
    BRANCHES,
    PATHS,
    advance_exp9_diagnostic,
    bandinv_marginal_variances,
    classic_nesterov_frontend,
    estimate_output_bias,
    extract_muon_blocks,
    init_exp9_shadow_state,
    muon_parameter_paths,
    nonlinear_response_decomposition,
    pre_q_marginal_variances,
)
from exp9.diagnostics import cancellation_statistics, cross_seed_aggregate, degradation
from exp9.online_shadow import Exp9WindowCollector


def _strategy(horizon=8):
  coef = jnp.asarray([1.0, .5], jnp.float32)
  return BandInvMFStrategy(
      horizon=horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=jnp.ones((horizon,), jnp.float32), noising_coef=coef,
      strategy_coef=jnp.ones((horizon,), jnp.float32), sensitivity_squared=jnp.asarray(1.),
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
  delta = 1e-3
  plus = nonlinear_response_decomposition(x, delta * r, ns_steps=3, consistent_rms=.2)
  np.testing.assert_allclose(result["P1"], plus["P1"] / delta, rtol=2e-3, atol=2e-3)
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


def test_primary_blocks_follow_vit_labels_only():
  params = {
      "blocks": ({"attention": {"query": {"kernel": _matrix()}}},),
      "head": {"kernel": _matrix(.3)},
  }
  paths = muon_parameter_paths(params)
  blocks = extract_muon_blocks(params, paths)
  assert list(blocks) == ["blocks/0/attention/query/kernel"]


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
        bias=empty_branch, even_response=empty_branch, stage_odd=stages, secondary_stage_odd=stages,
        noise_signal_ratio={branch: {"block": 0.0} for branch in BRANCHES},
        clean_pre_q_norm={"block": 1.0}, normalization_boundary_margin={"block": 1.0},
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


def test_degradation_names_and_reference_statistics():
  np.testing.assert_allclose(
      list(degradation({"P0": .8, "P1": .6, "P2": .5, "P3": .2}).values()),
      [.2, .1, .3],
  )
  np.testing.assert_allclose(
      cancellation_statistics(np.asarray([[1.], [2.]]), weight_decay=0., learning_rate=.1)["J"], 9.
  )
