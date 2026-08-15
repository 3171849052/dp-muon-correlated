import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training import init_nonamplified_bandinv_state, make_nonamplified_bandinv_train_step
from dp_muon.training.cifar10_driver import run_training


def _setup():
  horizon, momentum, learning_rate = 3, 0.0, 0.1
  coef = jnp.ones((1,), jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(n=horizon, noising_coef=coef, min_sep=1, max_participations=None)
  strategy = BandInvMFStrategy(horizon, 1, 1, None, fixed_lr_nesterov_trajectory_workload_coef(horizon, momentum, learning_rate), coef, jnp.ones((horizon,)), sensitivity, jnp.array(0.0))
  calibration = calibrate_nonamplified_bandinv(epsilon=2.0, delta=1e-5, clip_norm=1.0, normalize_by=1.0, adjacency="add_remove", sensitivity_squared=float(sensitivity))
  step = make_nonamplified_bandinv_train_step(lambda p, b: p * b["x"][0], strategy, calibration, ParticipationSpec(horizon, 1, None), momentum, learning_rate, microbatch_size=1)
  return strategy, step


def test_driver_uses_exact_horizon_and_checkpoint_resume(tmp_path):
  strategy, step = _setup()
  batches = [{"x": jnp.array([value], jnp.float32)} for value in (1.0, 2.0, 3.0)]
  initial = init_nonamplified_bandinv_state(jnp.array(0.0), strategy, jax.random.key(4))
  common = dict(horizon=3, experiment_config={"synthetic": True}, artifact_identifiers={"strategy": "test"}, eval_every=1)
  uninterrupted, _ = run_training(initial_state=initial, train_step=step, logical_batches=batches, **common)
  checkpoint = tmp_path / "state.pkl"
  # Stop at the first logical step, then resume from the exact same batch stream.
  first, _ = run_training(initial_state=initial, train_step=step, logical_batches=batches[:1], horizon=1, experiment_config={"synthetic": True}, artifact_identifiers={"strategy": "test"}, checkpoint_path=checkpoint)
  # The synthetic checkpoint's state is valid at step one; complete with the remaining stream through M6.
  from dp_muon.training.checkpoint import save_checkpoint
  save_checkpoint(checkpoint, state=first, current_step=1, experiment_config={"synthetic": True}, artifact_identifiers={"strategy": "test"})
  resumed, _ = run_training(initial_state=initial, train_step=step, logical_batches=batches, checkpoint_path=checkpoint, resume_checkpoint=checkpoint, **common)
  np.testing.assert_allclose(resumed.params, uninterrupted.params)
  assert int(resumed.nesterov_state.step) == int(resumed.noise_state.step) == strategy.horizon


def test_driver_prints_eval_progress_and_returns_history(capsys):
  strategy, step = _setup()
  batches = [{"x": jnp.array([value], jnp.float32)} for value in (1.0, 2.0, 3.0)]
  initial = init_nonamplified_bandinv_state(jnp.array(0.0), strategy, jax.random.key(5))
  _, history = run_training(
      initial_state=initial,
      train_step=step,
      logical_batches=batches,
      horizon=3,
      experiment_config={"synthetic": True},
      artifact_identifiers={"strategy": "test"},
      eval_every=2,
  )
  assert history == [{"step": 2}, {"step": 3}]
  assert capsys.readouterr().out.splitlines() == ["{'step': 2}", "{'step': 3}"]


def test_microbatch_size_must_divide_logical_batch_size():
  from dp_muon.training.cifar10_driver import Cifar10TrainConfig

  with pytest.raises(ValueError, match="divisible"):
    Cifar10TrainConfig(
        strategy="strategy.npz", pretrained="vit.npz", data_dir="data", batch_size=5,
        microbatch_size=2, clip_norm=1.0, epsilon=1.0, delta=1e-5,
        momentum=0.0, learning_rate=0.1, seed=0, checkpoint_dir="checkpoints", eval_every=1,
    )
