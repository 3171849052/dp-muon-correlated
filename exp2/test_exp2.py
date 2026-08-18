"""Focused correctness tests for the isolated Experiment 2 replay."""

import numpy as np
import pytest
import jax.numpy as jnp

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import adam_first_moment_workload_matrix, decayed_prefix_sum_workload_coef
import exp2.run as exp2_run
from exp2.run import adamw_perturbations, adamw_q, cancellation_statistics, run_replay
from exp2.strategies import ADAM_M_AWARE, DECAYED_PREFIX, StrategySpec, workload_for


def _trajectory():
  return {"g": np.array([[[0.5, -0.2]], [[-0.1, 0.7]], [[0.3, 0.4]]], dtype=np.float32),
          "parameter_name": "x", "start_step": 0, "learning_rate": 0.1,
          "beta1": 0.8, "beta2": 0.9, "eps": 1e-6, "weight_decay": 0.2}


def _strategy(name):
  matrix = np.eye(3, dtype=np.float32)
  coef = np.array([1.0], dtype=np.float32)
  return BandInvMFStrategy(horizon=3, bandwidth=1, min_sep=1, max_participations=None,
      workload_coef=None if name == ADAM_M_AWARE else jnp.ones(3),
      workload_matrix=matrix if name == ADAM_M_AWARE else None,
      noising_coef=jnp.asarray(coef), strategy_coef=jnp.ones(3),
      sensitivity_squared=jnp.asarray(1.0), objective=jnp.asarray(1.0))


def test_full_adamw_replay_matches_scalar_recurrence():
  trajectory = _trajectory()
  noise = np.array([[[[0.1, -0.1]], [[0.2, 0.0]], [[-0.1, 0.2]]]], dtype=np.float64)
  delta_q, delta_theta = adamw_perturbations(trajectory["g"], noise, **{
      key: trajectory[key] for key in ("learning_rate", "beta1", "beta2", "eps", "weight_decay")})
  m = np.zeros((1, 2)); v = np.zeros_like(m)
  expected_q = []
  for t, grad in enumerate(trajectory["g"] + noise[0]):
    m = trajectory["beta1"] * m + (1 - trajectory["beta1"]) * grad
    v = trajectory["beta2"] * v + (1 - trajectory["beta2"]) * grad * grad
    expected_q.append((m / (1 - trajectory["beta1"] ** (t + 1))) /
                      (np.sqrt(v / (1 - trajectory["beta2"] ** (t + 1))) + trajectory["eps"]))
  clean_q = adamw_q(trajectory["g"][None, ...], beta1=trajectory["beta1"],
                    beta2=trajectory["beta2"], eps=trajectory["eps"])
  np.testing.assert_allclose(delta_q[0] + clean_q[0], np.asarray(expected_q), rtol=1e-10, atol=1e-10)
  previous = np.zeros((1, 2)); expected_theta = []
  for value in delta_q[0]:
    previous = (1 - trajectory["learning_rate"] * trajectory["weight_decay"]) * previous - trajectory["learning_rate"] * value[0]
    expected_theta.append(previous.copy())
  np.testing.assert_allclose(delta_theta[0, :, 0], np.asarray(expected_theta)[:, 0])


def test_zero_noise_has_zero_perturbations():
  trajectory = _trajectory()
  dq, dt = adamw_perturbations(trajectory["g"], np.zeros((2, 3, 1, 2)), **{
      key: trajectory[key] for key in ("learning_rate", "beta1", "beta2", "eps", "weight_decay")})
  np.testing.assert_array_equal(dq, 0)
  np.testing.assert_array_equal(dt, 0)


def test_paired_latent_draws_and_global_scaling_are_deterministic(monkeypatch):
  trajectory = _trajectory()
  seen_latents = []
  seen_noises = []
  real_noise = exp2_run._strategy_noise
  real_perturbations = exp2_run.adamw_perturbations

  def capture_noise(latent, strategy):
    seen_latents.append(latent.copy())
    return real_noise(latent, strategy)

  def capture_perturbations(clean_gradients, noise, **kwargs):
    seen_noises.append(noise.copy())
    return real_perturbations(clean_gradients, noise, **kwargs)

  monkeypatch.setattr(exp2_run, "_strategy_noise", capture_noise)
  monkeypatch.setattr(exp2_run, "adamw_perturbations", capture_perturbations)
  result = run_replay(trajectory, strategies={DECAYED_PREFIX: _strategy(DECAYED_PREFIX), ADAM_M_AWARE: _strategy(ADAM_M_AWARE)},
      samples=4, seed=12, target_relative_noise=[0.1], sample_batch_size=2)
  summary = result[2]
  assert summary["paired_latent_draws"] is True
  # Both calibration and formal replay invoke strategies adjacently per batch.
  assert len(seen_latents) == 8
  for left, right in zip(seen_latents[::2], seen_latents[1::2], strict=True):
    np.testing.assert_array_equal(left, right)
  # The helper strategies are identity C^-1, so the same global scalar must
  # multiply every sample and every step of each recorded formal transcript.
  formal_latents = seen_latents[4:]
  for noise, latent in zip(seen_noises, formal_latents, strict=True):
    np.testing.assert_allclose(noise / latent, noise.flat[0] / latent.flat[0])
  for name in (DECAYED_PREFIX, ADAM_M_AWARE):
    assert summary["targets"]["0.1"]["strategies"][name]["actual_median_relative_noise_ratio"] > 0
    assert summary["targets"]["0.1"]["strategies"][name]["global_scalar"] > 0


def test_aggregate_is_ratio_of_sums():
  stats = cancellation_statistics(np.array([[[[2.0]], [[-2.0]]]]), np.array([[[[1.0]], [[0.0]]]]), 1.0)
  assert stats["R"] == stats["J"] / stats["D"]


def test_workloads_use_required_representations():
  spec_a = StrategySpec(DECAYED_PREFIX, 4, 2, 1, None, 0.1, 0.8, 0.2)
  spec_m = StrategySpec(ADAM_M_AWARE, 4, 2, 1, None, 0.1, 0.8, 0.2)
  assert "workload_coef" in workload_for(spec_a)
  assert "workload_matrix" in workload_for(spec_m)
  np.testing.assert_allclose(workload_for(spec_a)["workload_coef"], decayed_prefix_sum_workload_coef(4, .1, .2))
  np.testing.assert_allclose(workload_for(spec_m)["workload_matrix"], adam_first_moment_workload_matrix(4, .8, .1, .2))
