"""Unit coverage for the formal frozen-p continuous algorithm."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import (
    FrozenPAdamW,
    freeze_optax_adamw,
    frozen_p_time_workload,
    p_star_from_optax,
)
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
    continuous_hybrid_sensitivity_squared,
    epsilon_spent_for_continuous_hybrid_prefix,
)
from dp_muon.training import (
    init_nonamplified_frozen_p_bandinv_dpadamw_state,
    make_nonamplified_frozen_p_bandinv_dpadamw_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
from dp_muon.training import nonamplified_frozen_p_bandinv_dpadamw as hybrid_train
from dp_muon.training.bandinvmf_strategy_manager import (
    FrozenPBandInvMFFitRequest,
    get_or_fit_frozen_p_strategy_snapshot,
)
from dp_muon.training.cifar10_driver import (
    Cifar10FrozenPBandInvDPAdamWTrainConfig,
    FROZEN_P_BANDINV_DPADAMW_ALGORITHM,
)
from dp_muon.training.cifar10_frozen_p_bandinv_dpadamw_experiment import (
    load_cifar10_frozen_p_bandinv_dpadamw_config,
)
from scripts.run_cifar10 import _config_algorithm


CONFIG = Path("config/cifar10_bandinv_dpadamw_frozen_p.yaml")


def _strategy(*, horizon: int = 3, min_sep: int = 1, max_participations: int = 3):
  return BandInvMFStrategy(
      horizon=horizon,
      bandwidth=2,
      min_sep=min(min_sep, horizon),
      max_participations=max_participations,
      workload_coef=None,
      workload_matrix=jnp.asarray(
          np.tril(np.ones((horizon, horizon), dtype=np.float32))
      ),
      noising_coef=jnp.asarray([1.0, 0.25], dtype=jnp.float32),
      strategy_coef=jnp.ones((horizon,), dtype=jnp.float32),
      sensitivity_squared=jnp.asarray(1.0, dtype=jnp.float32),
      objective=jnp.asarray(1.0, dtype=jnp.float32),
  )


def _warm_optax_state():
  params = jnp.array([1.0, -2.0])
  optimizer = optax.adamw(
      learning_rate=0.05, b1=0.8, b2=0.9, eps=1e-6, weight_decay=0.01
  )
  state = optimizer.init(params)
  for gradient in (jnp.array([0.2, -0.1]), jnp.array([0.4, 0.3])):
    updates, state = optimizer.update(gradient, state, params)
    params = optax.apply_updates(params, updates)
  return params, state


def _calibration(strategy, *, horizon=5, tau=2, min_sep=1, max_participations=3):
  sensitivity = continuous_hybrid_sensitivity_squared(
      tau, strategy, min_sep=min_sep, max_participations=max_participations
  )
  return calibrate_nonamplified_bandinv(
      epsilon=3.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=1.0,
      adjacency="add_remove",
      sensitivity_squared=sensitivity,
  )


def test_yaml_tau_parses_and_dispatches_without_exp5():
  config = load_cifar10_frozen_p_bandinv_dpadamw_config(CONFIG)
  assert config.algorithm == FROZEN_P_BANDINV_DPADAMW_ALGORITHM
  assert config.switch_step == 32
  assert config.learning_rate > 0
  assert config.warmup_learning_rate > 0
  assert config.learning_rate != config.warmup_learning_rate
  assert _config_algorithm(str(CONFIG)) == FROZEN_P_BANDINV_DPADAMW_ALGORITHM


def test_train_config_validates_tau_against_horizon():
  with pytest.raises(ValueError, match="switch_step"):
    Cifar10FrozenPBandInvDPAdamWTrainConfig(
        strategy="strategy.npz", pretrained="vit.npz", data_dir="data",
        batch_size=4, microbatch_size=2, clip_norm=1.0, epsilon=3.0,
        delta=1e-5, learning_rate=.05, beta1=.8, beta2=.9, eps=1e-6,
        weight_decay=.01, seed=0, checkpoint_dir="checkpoints", eval_every=1,
        horizon=4, min_sep=1, max_participations=2, switch_step=4,
    )


def test_p_star_uses_real_optax_nu_and_count_and_freeze_is_exact():
  _, state = _warm_optax_state()
  expected = jax.tree_util.tree_map(
      lambda nu: 1.0 / (jnp.sqrt(nu / (1.0 - 0.9 ** int(state[0].count))) + 1e-6),
      state[0].nu,
  )
  frozen = freeze_optax_adamw(state, beta2=.9, eps=1e-6)
  np.testing.assert_allclose(
      p_star_from_optax(state, beta2=.9, eps=1e-6), expected, rtol=3e-6
  )
  assert int(frozen.count) == 2
  np.testing.assert_array_equal(frozen.frozen_nu, state[0].nu)


def test_frozen_update_matches_formula_and_preserves_p_and_nu():
  params, state = _warm_optax_state()
  frozen = freeze_optax_adamw(state, beta2=.9, eps=1e-6)
  gradient = jnp.array([.3, -.4])
  updates, new_state = FrozenPAdamW(.05, .8, .01).update(gradient, frozen, params)
  expected_mu = .8 * frozen.mu + .2 * gradient
  expected = -.05 * (
      frozen.p_star * expected_mu / (1.0 - .8 ** int(new_state.count)) + .01 * params
  )
  np.testing.assert_allclose(updates, expected, rtol=2e-6)
  np.testing.assert_array_equal(new_state.frozen_nu, frozen.frozen_nu)
  np.testing.assert_array_equal(new_state.p_star, frozen.p_star)
  assert int(new_state.count) == 3


def test_frozen_workload_keeps_global_bias_correction():
  actual = frozen_p_time_workload(
      3, tau=4, beta1=.8, learning_rate=.05, weight_decay=.01
  )
  restarted = frozen_p_time_workload(
      3, tau=1, beta1=.8, learning_rate=.05, weight_decay=.01
  )
  assert not np.allclose(actual, restarted)
  assert np.isclose(actual[0, 0], -.05 * .2 / (1.0 - .8**5))


def test_formal_trainer_has_iid_warmup_and_one_continuous_phase_stream(monkeypatch):
  tau, horizon = 2, 5
  strategy = _strategy(horizon=horizon - tau, max_participations=3)
  calibration = _calibration(strategy, horizon=horizon, tau=tau)
  participation = ParticipationSpec(horizon, 1, 3)
  calls = {"iid": 0, "band": 0}

  def iid_noise(key, template, std):
    calls["iid"] += 1
    return jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * .1, template), key

  def band_noise(key, state, coef, std):
    calls["band"] += 1
    return (
        jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * .2, state.buffer[0]),
        replace(state, step=state.step + 1, cursor=(state.cursor + 1) % state.bandwidth),
        key,
    )

  monkeypatch.setattr(hybrid_train, "_sample_iid_gaussian_noise", iid_noise)
  monkeypatch.setattr(hybrid_train, "sample_bandinv_noise", band_noise)
  step, optimizer = make_nonamplified_frozen_p_bandinv_dpadamw_train_step(
      lambda params, batch: jnp.sum(params * batch["x"]),
      strategy, calibration, participation,
      switch_step=tau, learning_rate=.05, beta1=.8, beta2=.9,
      eps=1e-6, weight_decay=.01, microbatch_size=1,
  )
  state = init_nonamplified_frozen_p_bandinv_dpadamw_state(
      jnp.array([1., -2.]), strategy, jax.random.key(0), optimizer,
      switch_step=tau,
  )
  compiled_step = jax.jit(step)
  p_before = None
  nu_before = None
  for current in range(1, horizon + 1):
    state = compiled_step(state, {"x": jnp.array([.4, -.2])})
    if current == tau:
      p_before = state.frozen_state.p_star
      nu_before = state.frozen_state.frozen_nu
  # Both lax.cond branches are traced once; runtime selection still gives two
  # IID updates followed by three continuous-BandInvMF updates.
  assert calls == {"iid": 1, "band": 1}
  assert int(state.frozen_state.count) == horizon
  assert int(state.step) == horizon
  assert int(state.noise_state.step) == horizon - tau
  np.testing.assert_array_equal(state.frozen_state.p_star, p_before)
  np.testing.assert_array_equal(state.frozen_state.frozen_nu, nu_before)


def test_warmup_and_frozen_phase_use_separate_learning_rates(monkeypatch):
  tau, horizon = 1, 2
  strategy = replace(
      _strategy(horizon=1, max_participations=1),
      bandwidth=1,
      noising_coef=jnp.asarray([1.0], dtype=jnp.float32),
      strategy_coef=jnp.ones((1,), dtype=jnp.float32),
  )
  calibration = _calibration(
      strategy, horizon=horizon, tau=tau, max_participations=1
  )

  def zero_iid_noise(key, template, std):
    del std
    return jax.tree_util.tree_map(jnp.zeros_like, template), key

  def zero_band_noise(key, noise_state, noising_coef, std):
    del noising_coef, std
    noise = jax.tree_util.tree_map(jnp.zeros_like, noise_state.buffer[0])
    return (
        noise,
        replace(
            noise_state,
            step=noise_state.step + 1,
            cursor=(noise_state.cursor + 1) % noise_state.bandwidth,
        ),
        key,
    )

  monkeypatch.setattr(hybrid_train, "_sample_iid_gaussian_noise", zero_iid_noise)
  monkeypatch.setattr(hybrid_train, "sample_bandinv_noise", zero_band_noise)
  train_step, optimizer = make_nonamplified_frozen_p_bandinv_dpadamw_train_step(
      lambda params, batch: jnp.sum(params * batch["x"][0]),
      strategy,
      calibration,
      ParticipationSpec(horizon, 1, 1),
      switch_step=tau,
      learning_rate=0.01,
      warmup_learning_rate=0.1,
      beta1=0.8,
      beta2=0.9,
      eps=1e-6,
      weight_decay=0.0,
      microbatch_size=1,
  )
  params = jnp.array([1.0, -2.0])
  state = init_nonamplified_frozen_p_bandinv_dpadamw_state(
      params, strategy, jax.random.key(0), optimizer, switch_step=tau
  )
  batch = {"x": jnp.array([[0.4, -0.2]])}

  warm_state = train_step(state, batch)
  warm_optimizer = optax.adamw(
      learning_rate=0.1, b1=0.8, b2=0.9, eps=1e-6, weight_decay=0.0
  )
  warm_updates, _ = warm_optimizer.update(
      batch["x"][0], warm_optimizer.init(params), params
  )
  np.testing.assert_allclose(
      warm_state.params, optax.apply_updates(params, warm_updates), rtol=2e-6
  )

  phase_state = train_step(warm_state, batch)
  phase_count = warm_state.frozen_state.count + 1
  phase_mu = 0.8 * warm_state.frozen_state.mu + 0.2 * batch["x"][0]
  phase_updates = -0.01 * (
      warm_state.frozen_state.p_star * phase_mu / (1.0 - 0.8 ** int(phase_count))
  )
  np.testing.assert_allclose(
      phase_state.params,
      warm_state.params + phase_updates,
      rtol=2e-6,
  )


def test_full_hybrid_privacy_uses_one_final_calibration():
  tau, horizon = 2, 6
  strategy = _strategy(horizon=horizon - tau, max_participations=3)
  sensitivity = continuous_hybrid_sensitivity_squared(
      tau, strategy, min_sep=1, max_participations=3
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=1.0,
      adjacency="add_remove", sensitivity_squared=sensitivity,
  )
  final = epsilon_spent_for_continuous_hybrid_prefix(
      prefix_steps=horizon, tau=tau, phase_strategy=strategy,
      min_sep=1, max_participations=3, calibration=calibration,
      full_sensitivity_squared=sensitivity,
  )
  assert final == pytest.approx(3.0, rel=1e-6)


def test_continuous_strategy_manager_fits_full_frozen_workload_once(tmp_path):
  request = FrozenPBandInvMFFitRequest(
      horizon=8, switch_step=2, min_sep=2, max_participations=3,
      bandwidth=2, beta1=.8, learning_rate=.05, weight_decay=.01,
      reduction="mean", max_optimizer_steps=2, strategy_dir=tmp_path,
      force_refit=False,
  )
  captured = {}

  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    captured.update(horizon=horizon, bandwidth=bandwidth, min_sep=min_sep, **kwargs)
    workload = np.asarray(kwargs["workload_matrix"])
    return BandInvMFStrategy(
        horizon=horizon, bandwidth=bandwidth, min_sep=min_sep,
        max_participations=request.max_participations, workload_coef=None,
        workload_matrix=jnp.asarray(workload),
        noising_coef=jnp.asarray([1.0, .2]), strategy_coef=jnp.ones((horizon,)),
        sensitivity_squared=jnp.asarray(1.0), objective=jnp.asarray(1.0),
    )

  snapshot, action = get_or_fit_frozen_p_strategy_snapshot(
      request, fit_strategy=fake_fit
  )
  assert action == "fit"
  assert captured["horizon"] == request.horizon - request.switch_step
  np.testing.assert_allclose(
      captured["workload_matrix"],
      np.abs(frozen_p_time_workload(6, tau=2, beta1=.8, learning_rate=.05, weight_decay=.01)),
  )
  snapshot2, action2 = get_or_fit_frozen_p_strategy_snapshot(
      request, fit_strategy=lambda *args, **kwargs: pytest.fail("must reuse")
  )
  assert action2 == "reuse"
  assert snapshot.path == snapshot2.path


def test_frozen_hybrid_checkpoint_validates_global_and_phase_steps(tmp_path):
  tau, horizon = 2, 4
  strategy = _strategy(horizon=horizon - tau, max_participations=3)
  calibration = _calibration(strategy, horizon=horizon, tau=tau)
  step, optimizer = make_nonamplified_frozen_p_bandinv_dpadamw_train_step(
      lambda params, batch: jnp.sum(params * batch["x"]),
      strategy, calibration, ParticipationSpec(horizon, 1, 3),
      switch_step=tau, learning_rate=.05, beta1=.8, beta2=.9,
      eps=1e-6, weight_decay=.01, microbatch_size=1,
  )
  state = init_nonamplified_frozen_p_bandinv_dpadamw_state(
      jnp.array([1., -2.]), strategy, jax.random.key(9), optimizer,
      switch_step=tau,
  )
  for _ in range(horizon):
    state = step(state, {"x": jnp.array([.4, -.2])})
  checkpoint = tmp_path / "frozen.pkl"
  save_checkpoint(
      checkpoint, state=state, current_step=horizon,
      experiment_config={"switch_step": tau},
      artifact_identifiers={"algorithm": FROZEN_P_BANDINV_DPADAMW_ALGORITHM},
  )
  restored = load_checkpoint(checkpoint)["state"]
  assert int(restored.step) == horizon
  assert int(restored.noise_state.step) == horizon - tau
  assert int(restored.frozen_state.count) == horizon
