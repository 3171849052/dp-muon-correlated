import numpy as np
import jax.numpy as jnp
import pytest

from dp_muon.analysis import cancellation_statistics
from dp_muon.bandinvmf import BandInvMFArtifactMetadata, BandInvMFStrategy
from dp_muon.optim import MUON_Q_STAGES
from exp1 import run as exp1_run
from exp1.run import run_replay, validate_exp1_strategy


def _trajectory(*, start_step: int = 0):
  return {
      "u": np.array([
          [[1.0, -0.5], [0.25, 0.75]],
          [[-0.25, 1.5], [0.5, -1.0]],
      ], dtype=np.float32),
      "learning_rates": np.array([0.1, 0.1]),
      "parameter_name": "test/matrix",
      "start_step": start_step,
      "momentum": 0.7,
      "ns_steps": 2,
      "consistent_rms": 1 / np.sqrt(2),
      "use_bf16_ns": False,
  }


def _strategy() -> BandInvMFStrategy:
  return BandInvMFStrategy(
      horizon=2, bandwidth=2, min_sep=1, max_participations=None,
      workload_coef=jnp.ones(2), noising_coef=jnp.asarray([1.0, -0.5]),
      strategy_coef=jnp.asarray([1.0, 0.5]), sensitivity_squared=jnp.asarray(1.0),
      objective=jnp.asarray(1.0),
  )


def _metadata(**overrides) -> BandInvMFArtifactMetadata:
  values = {
      "reduction": "mean", "workload_type": "nesterov-trajectory",
      "momentum": 0.7, "learning_rate": 0.1, "max_optimizer_steps": 1,
  }
  values.update(overrides)
  return BandInvMFArtifactMetadata(**values)


def test_two_step_opposite_noise_identifies_prefix_cancellation():
  e = np.array([[[[2.0]], [[-2.0]]]], dtype=np.float64)
  stats = cancellation_statistics(e, np.array([1.0, 1.0]))
  np.testing.assert_allclose(stats["J"], [4.0, 0.0])
  np.testing.assert_allclose(stats["D"], [4.0, 8.0])
  np.testing.assert_allclose(stats["R"], [1.0, 0.0])
  assert stats["aggregate_R"] < 1.0


def test_replay_is_reproducible_with_fixed_seed():
  trajectory = _trajectory()
  kwargs = dict(
      noising_coef=np.array([1.0, -0.5]), samples=6, seed=17,
      target_median_r=[0.1], sample_batch_size=2,
  )
  first = run_replay(trajectory, **kwargs)
  second = run_replay(trajectory, **kwargs)
  assert first == second


def test_exp1_rejects_prefix_strategy_workload():
  with pytest.raises(ValueError, match="workload_type='nesterov-trajectory'"):
    validate_exp1_strategy(_strategy(), _metadata(workload_type="prefix"), _trajectory())


def test_exp1_rejects_strategy_momentum_mismatch():
  with pytest.raises(ValueError, match="momentum does not match"):
    validate_exp1_strategy(_strategy(), _metadata(momentum=0.6), _trajectory())


def test_exp1_rejects_strategy_learning_rate_mismatch():
  with pytest.raises(ValueError, match="learning_rate does not match"):
    validate_exp1_strategy(_strategy(), _metadata(learning_rate=0.2), _trajectory())


@pytest.mark.parametrize("field", ["momentum", "learning_rate"])
def test_exp1_rejects_missing_strategy_temporal_metadata(field):
  with pytest.raises(ValueError, match=f"missing {field}"):
    validate_exp1_strategy(_strategy(), _metadata(**{field: None}), _trajectory())


def test_replay_rejects_mid_trajectory_without_history():
  with pytest.raises(ValueError, match="start_step == 0"):
    run_replay(
        _trajectory(start_step=1), noising_coef=np.array([1.0, -0.5]),
        samples=2, seed=0, target_median_r=[0.1], sample_batch_size=1,
    )


def test_replay_uses_one_global_scalar_and_hits_target_median(monkeypatch):
  trajectory = _trajectory()
  observed_noise = []

  def capture_deltas(u, noise, clean_q, trajectory):
    del u, clean_q, trajectory
    observed_noise.append(noise.copy())
    return {stage: noise for stage in MUON_Q_STAGES}

  monkeypatch.setattr(exp1_run, "_q_deltas", capture_deltas)
  _, _, summary = run_replay(
      trajectory, noising_coef=np.array([1.0, -0.5]), samples=6, seed=23,
      target_median_r=[0.2], sample_batch_size=2,
  )
  target_summary = summary["targets"]["0.2"]
  scalar = target_summary["global_scalar"]
  operator = exp1_run.make_causal_noise_operator(
      np.array([1.0, -0.5]), horizon=2, momentum=trajectory["momentum"]
  )
  rng = np.random.default_rng(23)
  expected = []
  for _ in range(3):
    latent = rng.standard_normal((2, *trajectory["u"].shape), dtype=np.float32)
    raw = np.einsum("ts,bsij->btij", operator.total, latent, optimize=True).astype(np.float32)
    expected.append((scalar * raw).astype(np.float32))
  np.testing.assert_array_equal(np.concatenate(observed_noise), np.concatenate(expected))
  np.testing.assert_allclose(
      target_summary["actual_median_relative_noise_ratio"], 0.2, rtol=1e-5, atol=1e-6
  )
