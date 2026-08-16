import numpy as np

from dp_muon.analysis import cancellation_statistics
from exp1.run import run_replay


def test_two_step_opposite_noise_identifies_prefix_cancellation():
  e = np.array([[[[2.0]], [[-2.0]]]], dtype=np.float64)
  stats = cancellation_statistics(e, np.array([1.0, 1.0]))
  np.testing.assert_allclose(stats["J"], [4.0, 0.0])
  np.testing.assert_allclose(stats["D"], [4.0, 8.0])
  np.testing.assert_allclose(stats["R"], [1.0, 0.0])
  assert stats["aggregate_R"] < 1.0


def test_replay_is_reproducible_with_fixed_seed():
  trajectory = {
      "u": np.array([
          [[1.0, -0.5], [0.25, 0.75]],
          [[-0.25, 1.5], [0.5, -1.0]],
      ], dtype=np.float32),
      "learning_rates": np.array([0.1, 0.1]),
      "parameter_name": "test/matrix",
      "start_step": 0,
      "momentum": 0.7,
      "ns_steps": 2,
      "consistent_rms": 1 / np.sqrt(2),
      "use_bf16_ns": False,
  }
  kwargs = dict(
      noising_coef=np.array([1.0, -0.5]), samples=6, seed=17,
      target_median_r=[0.1], sample_batch_size=2,
  )
  first = run_replay(trajectory, **kwargs)
  second = run_replay(trajectory, **kwargs)
  assert first == second
