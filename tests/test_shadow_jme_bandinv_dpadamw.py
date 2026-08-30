"""Behavioral tests for correlated warmup plus shadow-JME AdamW."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import (
    frozen_p_adamw_segment_workload_matrix,
    freeze_optax_adamw,
    shadow_jme_second_moment_endpoint_workload_coef,
    shadow_jme_second_moment_endpoint_workload_matrix,
)
from dp_muon.privacy import (
    aggregate_square_sensitivities,
    calibrate_shadow_jme,
    epsilon_spent_for_shadow_jme_prefix,
    jme_gamma_and_joint_sensitivity,
)
from dp_muon.training import nonamplified_shadow_jme_bandinv_dpadamw as shadow_train
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
from dp_muon.training.bandinvmf_strategy_manager import (
    ShadowJMEFirstBandInvMFFitRequest,
    ShadowJMESecondBandInvMFFitRequest,
    fit_shadow_jme_first_strategy,
    fit_shadow_jme_second_strategy,
)


def _strategy(n: int, value: float = 1.0) -> BandInvMFStrategy:
  return BandInvMFStrategy(
      horizon=n,
      bandwidth=1,
      min_sep=n,
      max_participations=1,
      workload_coef=jnp.ones(n),
      noising_coef=jnp.asarray([value], dtype=jnp.float32),
      strategy_coef=jnp.ones(n, dtype=jnp.float32),
      sensitivity_squared=jnp.asarray(1.0),
      objective=jnp.asarray(1.0),
  )


def _plan() -> shadow_train.ShadowJMEPlan:
  warmup = _strategy(2, 3.0)
  first = (_strategy(2, 1.0), _strategy(2, 1.0))
  second = (_strategy(2, 2.0), _strategy(2, 2.0))
  calibration = calibrate_shadow_jme(
      epsilon=2.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=1.0,
      adjacency="add_remove",
      warmup_strategy=warmup,
      first_strategies=first,
      second_strategies=second,
  )
  return shadow_train.ShadowJMEPlan(
      condition="shadow-jme-t2-seg2",
      warmup_steps=2,
      segment_lengths=(2, 2),
      warmup_strategy=warmup,
      first_strategies=first,
      second_strategies=second,
      calibration=calibration,
      runtime_bandwidth=1,
      beta1=0.8,
      beta2=0.9,
      learning_rate=0.01,
      eps=1e-6,
      weight_decay=0.01,
      v_floor=1e-5,
      min_sep=2,
      max_participations=1,
  )


def _loss(params, batch):
  return 0.5 * (params["x"] * batch["x"][0]) ** 2


def _fake_noise_factory():
  calls = []

  def fake_noise(key, state, coef, std):
    del key, std
    value = jnp.asarray(coef[0]) / 10.0
    calls.append(value)
    noise = jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf[0], value), state.buffer
    )
    return noise, replace(
        state,
        cursor=jnp.mod(state.cursor + 1, state.bandwidth),
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    ), jax.random.key(0)

  return fake_noise, calls


def _make():
  plan = _plan()
  step, optimizer = shadow_train.make_nonamplified_shadow_jme_bandinv_dpadamw_train_step(
      _loss, plan
  )
  state = shadow_train.init_nonamplified_shadow_jme_bandinv_dpadamw_state(
      {"x": jnp.array(1.0)}, plan, jax.random.key(17), optimizer
  )
  return plan, step, state


def test_warmup_is_dynamic_adamw_and_switch_p_matches_last_adam_state(monkeypatch):
  plan, step, state = _make()
  fake_noise, calls = _fake_noise_factory()
  monkeypatch.setattr(shadow_train, "sample_bandinv_noise", fake_noise)
  state = step(state, {"x": jnp.array([0.4])})
  state = step(state, {"x": jnp.array([0.4])})
  expected = freeze_optax_adamw(state.optimizer_state, beta2=plan.beta2, eps=plan.eps)
  np.testing.assert_array_equal(state.frozen_state.p_star, expected.p_star)
  np.testing.assert_array_equal(state.frozen_state.mu, expected.mu)
  np.testing.assert_array_equal(state.v_shadow, expected.frozen_nu)
  assert int(state.phase) == 1
  assert int(state.step) == plan.warmup_steps
  assert int(state.noise_state_m.step) == int(state.noise_state_v.step) == 0
  assert calls


def test_frozen_p_and_shadow_change_independently_inside_a_segment(monkeypatch):
  plan, step, state = _make()
  fake_noise, calls = _fake_noise_factory()
  monkeypatch.setattr(shadow_train, "sample_bandinv_noise", fake_noise)
  for _ in range(2):
    state = step(state, {"x": jnp.array([0.4])})
  p_at_start = state.frozen_state.p_star
  shadow_at_start = state.v_shadow
  state = step(state, {"x": jnp.array([0.4])})
  # Calls 3 and 4 are independent first/second channel noises.  The expected
  # shadow value uses x*x and the second channel, not (x + Z_m)**2.
  x = 0.16
  q_private = x * x + 0.2 / np.sqrt(float(state.gamma))
  expected_shadow = plan.beta2 * shadow_at_start["x"] + (1 - plan.beta2) * q_private
  np.testing.assert_allclose(state.v_shadow["x"], expected_shadow, rtol=3e-3, atol=2e-4)
  np.testing.assert_array_equal(state.frozen_state.p_star, p_at_start)
  assert not np.isclose(float(state.v_shadow["x"]), (x + 0.3) ** 2)
  assert calls


def test_aggregate_first_sensitivity_is_clip_over_normalization():
  delta1, delta2 = aggregate_square_sensitivities(
      clip_norm=3.0, normalize_by=12.0, adjacency="add_remove"
  )
  assert delta1 == pytest.approx(3.0 / 12.0)
  assert _plan().calibration.query_sensitivity == pytest.approx(1.0)


def test_aggregate_square_sensitivity_uses_c_squared_over_b_not_b_squared():
  delta1, delta2 = aggregate_square_sensitivities(
      clip_norm=3.0, normalize_by=12.0, adjacency="add_remove"
  )
  assert delta2 == pytest.approx(2.0 * 3.0**2 / 12.0)
  assert delta2 != pytest.approx(2.0 * 3.0**2 / 12.0**2)
  assert delta2 == pytest.approx(2.0 * 3.0 * delta1)


def test_replace_one_scales_both_aggregate_query_bounds_by_existing_factor_two():
  add_remove = aggregate_square_sensitivities(
      clip_norm=3.0, normalize_by=12.0, adjacency="add_remove"
  )
  replace_one = aggregate_square_sensitivities(
      clip_norm=3.0, normalize_by=12.0, adjacency="replace_one"
  )
  np.testing.assert_allclose(replace_one, 2.0 * np.asarray(add_remove))


def test_calibration_can_reserve_an_explicit_segment_sensitivity_envelope():
  baseline = _plan().calibration
  reserved = calibrate_shadow_jme(
      epsilon=baseline.epsilon,
      delta=baseline.delta,
      clip_norm=baseline.clip_norm,
      normalize_by=baseline.normalize_by,
      adjacency=baseline.adjacency,
      warmup_strategy=_strategy(2, 3.0),
      first_strategies=(_strategy(2), _strategy(2)),
      second_strategies=(_strategy(2, 2.0), _strategy(2, 2.0)),
      segment_sensitivity_upper_bounds=(5.0, 6.0),
  )
  assert reserved.segment_sensitivity_squared == (5.0, 6.0)
  assert reserved.aggregate_delta1 == pytest.approx(1.0)
  assert reserved.aggregate_delta2 == pytest.approx(2.0)


def test_plan_construction_calibrates_with_explicit_surrogate_envelope():
  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    del bandwidth, min_sep, kwargs
    return _strategy(horizon)

  plan = shadow_train.fit_shadow_jme_plan(
      horizon=4,
      warmup_steps=2,
      segment_length=2,
      min_sep=2,
      max_participations=1,
      bandwidth=1,
      reduction="mean",
      max_optimizer_steps=1,
      learning_rate=0.01,
      beta1=0.8,
      beta2=0.9,
      eps=1e-6,
      weight_decay=0.01,
      epsilon=2.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=1.0,
      adjacency="add_remove",
      fit_strategy=fake_fit,
  )
  # A unit-coefficient length-2 strategy has ||C||_{1->2}^2 = 2, hence the
  # raw joint bound is 4 and the plan reserves the 1.25 operational envelope.
  assert plan.calibration.segment_sensitivity_squared == pytest.approx((5.0,))


def test_joint_bound_and_gamma_use_both_record_level_query_sensitivities():
  first = replace(_strategy(2), strategy_coef=jnp.asarray([2.0, 0.5]))
  second = replace(_strategy(2), strategy_coef=jnp.asarray([1.0, 0.25]))
  gamma, sensitivity = jme_gamma_and_joint_sensitivity(
      first,
      second,
      clip_norm=4.0,
      normalize_by=8.0,
      adjacency="add_remove",
  )
  first_norm = 2.0**2 + 0.5**2
  second_norm = 1.0**2 + 0.25**2
  delta1 = 4.0 / 8.0
  delta2 = 2.0 * 4.0**2 / 8.0
  assert gamma == pytest.approx(delta1**2 * first_norm / (delta2**2 * second_norm))
  assert sensitivity**2 == pytest.approx(2.0 * delta1**2 * first_norm)


def test_boundary_uses_bias_corrected_shadow_p_and_resets_both_streams(monkeypatch):
  plan, step, state = _make()
  fake_noise, _ = _fake_noise_factory()
  monkeypatch.setattr(shadow_train, "sample_bandinv_noise", fake_noise)
  for _ in range(4):
    state = step(state, {"x": jnp.array([0.4])})
  corrected = max(float(state.v_shadow["x"]) / (1 - plan.beta2 ** int(state.v_shadow_count)), plan.v_floor)
  expected_p = 1.0 / (np.sqrt(corrected) + plan.eps)
  np.testing.assert_allclose(state.frozen_state.p_star["x"], expected_p, rtol=1e-6)
  assert int(state.segment_index) == 1
  assert int(state.segment_start) == 4
  assert int(state.noise_state_m.step) == int(state.noise_state_v.step) == 0
  assert all(bool(jnp.all(leaf == 0)) for leaf in jax.tree_util.tree_leaves(state.noise_state_m.buffer))
  assert all(bool(jnp.all(leaf == 0)) for leaf in jax.tree_util.tree_leaves(state.noise_state_v.buffer))
  assert int(state.frozen_state.count) == 4


def test_endpoint_workload_is_exact_beta2_weighting():
  actual = shadow_jme_second_moment_endpoint_workload_coef(4, 0.9)
  expected = (1 - 0.9) * 0.9 ** np.arange(3, -1, -1)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_endpoint_workload_matrix_has_only_the_final_query_row():
  actual = np.asarray(shadow_jme_second_moment_endpoint_workload_matrix(4, 0.9))
  expected = (1 - 0.9) * 0.9 ** np.arange(3, -1, -1)
  assert actual.shape == (4, 4)
  np.testing.assert_array_equal(actual[:-1], np.zeros((3, 4)))
  np.testing.assert_allclose(actual[-1], expected, rtol=1e-6, atol=1e-7)


def test_strategy_helpers_use_fixed_p_and_endpoint_workloads():
  captured = []

  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    captured.append((horizon, bandwidth, min_sep, kwargs))
    return _strategy(horizon)

  first_request = ShadowJMEFirstBandInvMFFitRequest(
      segment_length=3,
      segment_start_step=2,
      min_sep=3,
      max_participations=1,
      bandwidth=2,
      beta1=0.8,
      learning_rate=0.1,
      weight_decay=0.01,
      frozen_preconditioner=0.25,
      reduction="mean",
      max_optimizer_steps=1,
  )
  fit_shadow_jme_first_strategy(first_request, fit_strategy=fake_fit)
  np.testing.assert_allclose(
      captured[0][3]["workload_matrix"],
      np.abs(np.asarray(frozen_p_adamw_segment_workload_matrix(
          3,
          beta1=0.8,
          learning_rate=0.1,
          weight_decay=0.01,
          frozen_preconditioner=0.25,
          first_moment_start_step=2,
      ))),
  )
  second_request = ShadowJMESecondBandInvMFFitRequest(
      segment_length=3,
      min_sep=3,
      max_participations=1,
      bandwidth=2,
      beta2=0.9,
      reduction="mean",
      max_optimizer_steps=1,
  )
  fit_shadow_jme_second_strategy(second_request, fit_strategy=fake_fit)
  np.testing.assert_allclose(
      captured[1][3]["workload_matrix"],
      shadow_jme_second_moment_endpoint_workload_matrix(3, 0.9),
  )
  assert "workload_coef" not in captured[1][3]


def test_runtime_refit_guard_rejects_a_pair_above_calibrated_bound():
  plan, _, state = _make()
  step, _ = shadow_train.make_nonamplified_shadow_jme_bandinv_dpadamw_train_step(
      _loss, plan
  )
  for _ in range(plan.warmup_steps):
    state = step(state, {"x": jnp.array([0.4])})
  too_sensitive = replace(
      plan.first_strategies[0],
      strategy_coef=2.0 * plan.first_strategies[0].strategy_coef,
  )
  with pytest.raises(RuntimeError, match="exceeds its calibrated"):
    shadow_train.begin_shadow_jme_segment(
        state,
        plan,
        segment_index=0,
        first_strategy=too_sensitive,
        second_strategy=plan.second_strategies[0],
    )


def test_runtime_refit_guard_accepts_a_pair_at_or_below_calibrated_bound():
  plan, _, state = _make()
  step, _ = shadow_train.make_nonamplified_shadow_jme_bandinv_dpadamw_train_step(
      _loss, plan
  )
  for _ in range(plan.warmup_steps):
    state = step(state, {"x": jnp.array([0.4])})
  smaller = replace(
      plan.first_strategies[0],
      strategy_coef=0.5 * plan.first_strategies[0].strategy_coef,
  )
  installed = shadow_train.begin_shadow_jme_segment(
      state,
      plan,
      segment_index=0,
      first_strategy=smaller,
      second_strategy=plan.second_strategies[0],
  )
  assert int(installed.phase) == 1
  np.testing.assert_allclose(installed.first_noising_coef[:2], smaller.noising_coef)


def test_jme_checkpoint_resume_matches_uninterrupted(tmp_path):
  plan, step, initial = _make()
  batches = [{"x": jnp.array([value])} for value in (0.4, 0.2, -0.3, 0.8, 0.1, -0.6)]
  uninterrupted = initial
  for batch in batches:
    uninterrupted = step(uninterrupted, batch)
  resumed = initial
  for batch in batches[:3]:
    resumed = step(resumed, batch)
  path = tmp_path / "shadow-jme.pkl"
  save_checkpoint(
      path,
      state=resumed,
      current_step=3,
      experiment_config={"algorithm": "shadow-jme"},
      artifact_identifiers={"algorithm": "shadow-jme"},
  )
  resumed = load_checkpoint(path)["state"]
  for batch in batches[3:]:
    resumed = step(resumed, batch)
  for left, right in zip(
      jax.tree_util.tree_leaves(resumed),
      jax.tree_util.tree_leaves(uninterrupted),
      strict=True,
  ):
    if str(jnp.asarray(left).dtype).startswith("key<"):
      np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
    else:
      np.testing.assert_array_equal(left, right)


def test_jitted_state_machine_switches_and_resets_both_channels():
  plan, step, state = _make()
  compiled_step = jax.jit(step)
  for _ in range(plan.horizon):
    state = compiled_step(state, {"x": jnp.array([0.4])})
  assert int(state.step) == plan.horizon
  assert int(state.segment_index) == len(plan.segment_lengths)
  assert int(state.noise_state_m.step) == int(state.noise_state_v.step) == 0
  assert int(state.v_shadow_count) == plan.horizon


def test_final_gdp_composition_stays_within_global_epsilon():
  plan = _plan()
  spent = epsilon_spent_for_shadow_jme_prefix(
      prefix_steps=plan.horizon,
      warmup_steps=plan.warmup_steps,
      segment_lengths=plan.segment_lengths,
      calibration=plan.calibration,
  )
  assert spent <= plan.calibration.epsilon + 1e-10
