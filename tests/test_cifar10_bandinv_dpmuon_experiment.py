"""Configuration and strategy orchestration for naive correlated DP-Muon."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from dp_muon.bandinvmf import BandInvMFStrategy, save_bandinv_strategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.training.bandinvmf_strategy_manager import LoadedStrategySnapshot
from dp_muon.training.cifar10_driver import (
    BANDINV_DPMUON_ALGORITHM,
    Cifar10BandInvDPMuonTrainConfig,
)
from dp_muon.training import cifar10_bandinv_dpmuon_experiment as experiment
from dp_muon.training import cifar10_experiment
from scripts import run_cifar10


CONFIG = Path("config/cifar10_bandinv_dpmuon_naive.yaml")


def _config(tmp_path, **changes):
  defaults = {
      "strategy_dir": str(tmp_path / "strategies"),
      "checkpoint_dir": str(tmp_path / "checkpoints"),
      "log_dir": str(tmp_path / "logs"),
  }
  defaults.update(changes)
  return replace(
      experiment.load_cifar10_bandinv_dpmuon_config(CONFIG),
      **defaults,
  )


def _strategy(config, participation):
  workload = fixed_lr_nesterov_trajectory_workload_coef(
      participation.horizon, config.momentum, config.muon_learning_rate
  )
  return BandInvMFStrategy(
      horizon=participation.horizon,
      bandwidth=config.bandwidth,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      workload_coef=workload,
      noising_coef=jnp.ones((config.bandwidth,), jnp.float32),
      strategy_coef=jnp.ones((participation.horizon,), jnp.float32),
      sensitivity_squared=jnp.array(1.0, jnp.float32),
      objective=jnp.array(1.0, jnp.float32),
  )


def _save_matching_artifact(config, participation, *, reduction=None):
  path = experiment.strategy_artifact_path(config, participation)
  save_bandinv_strategy(
      path,
      _strategy(config, participation),
      reduction=reduction or config.reduction,
      workload_type="nesterov-trajectory",
      momentum=config.momentum,
      learning_rate=config.muon_learning_rate,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  return path


def test_naive_yaml_parses_and_maps_trainer_fields(tmp_path):
  config = _config(tmp_path)
  assert config.algorithm == BANDINV_DPMUON_ALGORITHM
  assert config.epochs == 5
  assert config.schedule_mode == "fixed_cycle"
  train_config = experiment._train_config(config, tmp_path / "strategy.npz")
  assert isinstance(train_config, Cifar10BandInvDPMuonTrainConfig)
  for field in asdict(train_config):
    if field != "strategy":
      assert getattr(train_config, field) == getattr(config, field)
  assert train_config.strategy == str(tmp_path / "strategy.npz")


def test_derives_required_cifar10_fixed_cycle_participation():
  actual = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  assert (actual.horizon, actual.min_sep, actual.max_participations) == (488, 97, 5)


def test_missing_artifact_is_fitted_and_saved(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  path, _, action = experiment.get_or_fit_strategy(config, participation)
  assert action == "fit"
  assert path.is_file()
  assert len(calls) == 1


def test_compatible_artifact_is_reused(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  path = _save_matching_artifact(config, participation)
  monkeypatch.setattr(
      experiment,
      "fit_bandinv_strategy",
      lambda *args, **kwargs: pytest.fail("compatible artifact must be reused"),
  )
  actual_path, _, action = experiment.get_or_fit_strategy(config, participation)
  assert actual_path == path
  assert action == "reuse"


def test_metadata_mismatch_refits(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  _save_matching_artifact(config, participation, reduction="last")
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  _, _, action = experiment.get_or_fit_strategy(config, participation)
  assert action == "fit"
  assert len(calls) == 1


def test_force_refit_ignores_compatible_artifact(tmp_path, monkeypatch):
  config = _config(tmp_path, force_refit=True)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  _save_matching_artifact(config, participation)
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  _, _, action = experiment.get_or_fit_strategy(config, participation)
  assert action == "fit"
  assert len(calls) == 1


def test_strategy_workload_uses_muon_learning_rate_not_adamw(tmp_path, monkeypatch):
  config = _config(tmp_path, muon_learning_rate=0.0001, adamw_learning_rate=0.5)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  captured = {}

  def fake_fit(*args, **kwargs):
    captured["workload"] = np.asarray(kwargs["workload_coef"])
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  experiment.get_or_fit_strategy(config, participation)
  expected = fixed_lr_nesterov_trajectory_workload_coef(
      participation.horizon, config.momentum, config.muon_learning_rate
  )
  adamw_workload = fixed_lr_nesterov_trajectory_workload_coef(
      participation.horizon, config.momentum, config.adamw_learning_rate
  )
  assert np.array_equal(captured["workload"], np.asarray(expected))
  assert not np.array_equal(captured["workload"], np.asarray(adamw_workload))


def test_relative_strategy_dir_is_resolved_from_repository_root(tmp_path, monkeypatch):
  config = _config(tmp_path, strategy_dir="ci-relative-strategies")
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  monkeypatch.chdir(tmp_path)
  path = experiment.strategy_artifact_path(config, participation)
  assert path.parent == experiment.REPOSITORY_ROOT / "ci-relative-strategies"


def test_run_fits_strategy_writes_resolved_config_and_passes_final_path(tmp_path, monkeypatch):
  config = _config(tmp_path)
  config_path = tmp_path / "naive.yaml"
  config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  strategy_path = tmp_path / "strategies" / "strategy.npz"
  captured = {}

  monkeypatch.setattr(experiment, "load_cifar10_bandinv_dpmuon_config", lambda _: config)
  monkeypatch.setattr(
      experiment,
      "load_cifar10",
      lambda *args, **kwargs: (np.empty((50_000, 32, 32, 3), np.uint8), np.empty(50_000)),
  )
  monkeypatch.setattr(
      experiment,
      "get_or_fit_strategy_snapshot",
      lambda actual_config, actual_participation: (
          LoadedStrategySnapshot(strategy_path, _strategy(config, participation), "test-sha"), "fit"
      ),
  )

  def fake_train(train_config, **kwargs):
    captured["train_config"] = train_config
    captured["kwargs"] = kwargs
    return None, []

  monkeypatch.setattr(experiment, "train_cifar10_bandinv_dpmuon", fake_train)
  assert experiment.run_cifar10_bandinv_dpmuon(config_path) == (None, [])
  assert captured["train_config"].strategy == str(strategy_path)
  resolved = yaml.safe_load(
      (captured["kwargs"]["checkpoint_path"].parent.parent / "resolved_config.yaml").read_text()
  )
  assert resolved["participation"]["horizon"] == 488
  assert resolved["strategy"]["artifact"] == str(strategy_path.resolve())
  assert resolved["strategy"]["action"] == "fit"
  assert resolved["privacy_calibration"]


def test_prepare_run_snapshots_without_loading_data_or_training(tmp_path):
  config_path = tmp_path / "naive.yaml"
  log_dir = tmp_path / "runs"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("log_dir: logs", f"log_dir: {log_dir}"),
      encoding="utf-8",
  )
  paths = experiment.prepare_cifar10_bandinv_dpmuon_run(config_path)
  assert paths.directory.parent == log_dir
  assert paths.config.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
  assert paths.resolved_config.is_file()
  assert paths.metrics.is_file()


def test_cli_print_log_dir_and_gpu_for_naive(monkeypatch, capsys):
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", str(CONFIG), "--print-log-dir"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out.strip() == str(experiment.REPOSITORY_ROOT / "logs")
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", str(CONFIG), "--print-gpu"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out == "3\n"


def test_cli_prepare_and_run_dispatch_to_naive_adapter(tmp_path, monkeypatch, capsys):
  calls = []

  def fake_prepare(path):
    calls.append(("prepare", path))
    return type("Paths", (), {"directory": tmp_path / "run"})()

  def fake_run(path, *, resume_checkpoint=None, run_dir=None):
    calls.append(("run", path, resume_checkpoint, run_dir))

  monkeypatch.setattr(run_cifar10, "prepare_cifar10_bandinv_dpmuon_run", fake_prepare)
  monkeypatch.setattr(run_cifar10, "run_cifar10_bandinv_dpmuon", fake_run)
  monkeypatch.setattr(run_cifar10, "_config_algorithm", lambda _: BANDINV_DPMUON_ALGORITHM)
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", "naive.yaml", "--prepare-run"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out == f"{tmp_path / 'run'}\n"
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", "naive.yaml", "--resume-checkpoint", "checkpoint.pkl"],
  )
  run_cifar10.main()
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", "naive.yaml", "--run-dir", "logs/run"],
  )
  run_cifar10.main()
  assert calls == [
      ("prepare", "naive.yaml"),
      ("run", "naive.yaml", "checkpoint.pkl", None),
      ("run", "naive.yaml", None, "logs/run"),
  ]


def test_cli_unknown_algorithm_still_fails(tmp_path):
  config_path = tmp_path / "unknown.yaml"
  config_path.write_text("algorithm: unknown\n", encoding="utf-8")
  with pytest.raises(ValueError, match="unknown config.algorithm"):
    run_cifar10._config_algorithm(str(config_path))
