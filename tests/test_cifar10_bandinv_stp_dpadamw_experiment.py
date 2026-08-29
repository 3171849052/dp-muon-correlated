"""Configuration and launch routing tests for STP correlated DP-AdamW."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training import (
    init_nonamplified_bandinv_stp_dpadamw_state,
    make_nonamplified_bandinv_stp_dpadamw_train_step,
)
from dp_muon.training.bandinvmf_strategy_manager import LoadedStrategySnapshot
from dp_muon.training.cifar10_driver import (
    STP_BANDINV_DPADAMW_ALGORITHM,
    Cifar10BandInvSTPDPAdamWTrainConfig,
    run_training,
)
from dp_muon.training import cifar10_bandinv_stp_dpadamw_experiment as experiment
from dp_muon.training import cifar10_experiment
from scripts import run_cifar10


CONFIG = Path("config/cifar10_bandinv_dpadamw_stp.yaml")


def _config(tmp_path, **changes):
  defaults = {
      "strategy_dir": str(tmp_path / "strategies"),
      "checkpoint_dir": str(tmp_path / "checkpoints"),
      "log_dir": str(tmp_path / "logs"),
  }
  defaults.update(changes)
  return replace(
      experiment.load_cifar10_bandinv_stp_dpadamw_config(CONFIG), **defaults
  )


def test_stp_yaml_parses_two_distinct_epsilons_and_maps_trainer_fields(tmp_path):
  config = _config(tmp_path)
  assert config.algorithm == STP_BANDINV_DPADAMW_ALGORITHM
  assert config.scale_eps == 1e-8
  assert config.eps == 1e-8
  train_config = experiment._train_config(config, tmp_path / "strategy.npz")
  assert isinstance(train_config, Cifar10BandInvSTPDPAdamWTrainConfig)
  assert train_config.scale_eps == config.scale_eps
  assert train_config.eps == config.eps
  for field in asdict(train_config):
    if field != "strategy":
      assert getattr(train_config, field) == getattr(config, field)


def test_stp_uses_the_same_prefix_sum_strategy_artifact_shape(tmp_path):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(
      50_000, config.epochs, config.batch_size
  )
  path = experiment.strategy_artifact_path(config, participation)
  assert "decayed-prefix-sum" in path.name
  assert "stp" not in path.name
  assert f"_lr{config.learning_rate}" in path.name
  assert f"_wd{config.weight_decay}" in path.name


def test_stp_yaml_rejects_missing_stp_section(tmp_path):
  config_path = tmp_path / "missing.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("stp:\n  scale_eps: 1.0e-8\n", ""),
      encoding="utf-8",
  )
  with pytest.raises(ValueError, match="config sections must be exactly"):
    experiment.load_cifar10_bandinv_stp_dpadamw_config(config_path)


def test_cli_routes_stp_prepare_and_resume_to_existing_launcher(tmp_path, monkeypatch, capsys):
  calls = []

  def fake_prepare(path):
    calls.append(("prepare", path))
    return type("Paths", (), {"directory": tmp_path / "run"})()

  def fake_run(path, *, resume_checkpoint=None, run_dir=None):
    calls.append(("run", path, resume_checkpoint, run_dir))

  monkeypatch.setattr(run_cifar10, "prepare_cifar10_bandinv_stp_dpadamw_run", fake_prepare)
  monkeypatch.setattr(run_cifar10, "run_cifar10_bandinv_stp_dpadamw", fake_run)
  monkeypatch.setattr(run_cifar10, "_config_algorithm", lambda _: STP_BANDINV_DPADAMW_ALGORITHM)
  monkeypatch.setattr(run_cifar10, "load_cifar10_bandinv_stp_dpadamw_config", lambda _: _config(tmp_path))
  monkeypatch.setattr(run_cifar10, "resolve_bandinv_stp_dpadamw_log_dir", lambda _: tmp_path / "logs")

  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", "stp.yaml", "--prepare-run"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out == f"{tmp_path / 'run'}\n"
  monkeypatch.setattr(
      sys,
      "argv",
      ["run_cifar10.py", "--config", "stp.yaml", "--resume-checkpoint", "checkpoint.pkl"],
  )
  run_cifar10.main()
  assert calls == [
      ("prepare", "stp.yaml"),
      ("run", "stp.yaml", "checkpoint.pkl", None),
  ]


def test_tiny_yaml_runs_multiple_steps_through_existing_launcher(tmp_path, monkeypatch):
  config_path = tmp_path / "stp.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8")
      .replace("epochs: 5", "epochs: 2")
      .replace("logical_batch_size: 512", "logical_batch_size: 1")
      .replace("microbatch_size: 16", "microbatch_size: 1")
      .replace("bandwidth: 4", "bandwidth: 1")
      .replace("strategy_dir: artifacts/strategies", f"strategy_dir: {tmp_path / 'strategies'}")
      .replace("checkpoint_dir: checkpoints", f"checkpoint_dir: {tmp_path / 'checkpoints'}")
      .replace("log_dir: logs", f"log_dir: {tmp_path / 'logs'}"),
      encoding="utf-8",
  )
  images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
  labels = np.array([0, 1], dtype=np.int32)
  monkeypatch.setattr(experiment, "load_cifar10", lambda *args, **kwargs: (images, labels))
  captured = {}

  def fake_get_or_fit(config, participation):
    noising_coef = jnp.ones((1,), dtype=jnp.float32)
    sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
        n=participation.horizon,
        noising_coef=noising_coef,
        min_sep=participation.min_sep,
        max_participations=participation.max_participations,
    )
    strategy = BandInvMFStrategy(
        horizon=participation.horizon,
        bandwidth=1,
        min_sep=participation.min_sep,
        max_participations=participation.max_participations,
        workload_coef=jnp.ones((participation.horizon,), dtype=jnp.float32),
        noising_coef=noising_coef,
        strategy_coef=jnp.ones((participation.horizon,), dtype=jnp.float32),
        sensitivity_squared=sensitivity_squared,
        objective=jnp.array(0.0, dtype=jnp.float32),
    )
    snapshot = LoadedStrategySnapshot(tmp_path / "strategy.npz", strategy, "smoke-sha")
    captured["snapshot"] = snapshot
    return snapshot, "fit"

  monkeypatch.setattr(experiment, "get_or_fit_strategy_snapshot", fake_get_or_fit)

  def fake_train(train_config, **kwargs):
    strategy = captured["snapshot"].strategy
    calibration = calibrate_nonamplified_bandinv(
        epsilon=train_config.epsilon,
        delta=train_config.delta,
        clip_norm=train_config.clip_norm,
        normalize_by=float(train_config.batch_size),
        adjacency=train_config.adjacency,
        sensitivity_squared=float(strategy.sensitivity_squared),
    )
    step, optimizer = make_nonamplified_bandinv_stp_dpadamw_train_step(
        lambda params, batch: params["w"] * batch["x"][0],
        strategy,
        calibration,
        ParticipationSpec(strategy.horizon, strategy.min_sep, strategy.max_participations),
        learning_rate=train_config.learning_rate,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
        eps=train_config.eps,
        scale_eps=train_config.scale_eps,
        weight_decay=train_config.weight_decay,
    )
    initial = init_nonamplified_bandinv_stp_dpadamw_state(
        {"w": jnp.array(0.0)}, strategy, jax.random.key(12), optimizer
    )
    final, _ = run_training(
        initial_state=initial,
        train_step=step,
        logical_batches=[{"x": jnp.array([1.0])} for _ in range(strategy.horizon)],
        horizon=strategy.horizon,
        experiment_config=asdict(train_config),
        artifact_identifiers={"algorithm": STP_BANDINV_DPADAMW_ALGORITHM},
        checkpoint_path=kwargs["checkpoint_path"],
        eval_every=1,
    )
    captured["final_count"] = int(final.optimizer_state.count)
    return final, []

  monkeypatch.setattr(experiment, "train_cifar10_bandinv_stp_dpadamw", fake_train)
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", str(config_path)]
  )
  run_cifar10.main()
  assert captured["final_count"] == 4
