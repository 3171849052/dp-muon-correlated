"""Configuration and CLI plumbing for naive correlated DP-Muon."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

from dp_muon.training.cifar10_driver import (
    BANDINV_DPMUON_ALGORITHM,
    Cifar10BandInvDPMuonTrainConfig,
)
from dp_muon.training import cifar10_bandinv_dpmuon_experiment as experiment
from scripts import run_cifar10


CONFIG = Path("config/cifar10_bandinv_dpmuon_naive.yaml")


def _config(**changes):
  return replace(experiment.load_cifar10_bandinv_dpmuon_config(CONFIG), **changes)


def test_naive_yaml_parses_and_maps_every_trainer_field():
  config = _config()
  assert config.algorithm == BANDINV_DPMUON_ALGORITHM
  train_config = experiment._train_config(config)
  assert isinstance(train_config, Cifar10BandInvDPMuonTrainConfig)
  assert asdict(train_config) == {
      field: getattr(config, field)
      for field in asdict(train_config)
  }


def test_run_delegates_to_existing_driver_with_mapped_config(tmp_path, monkeypatch):
  config = _config(log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "checkpoints"))
  captured = {}

  def fake_train(train_config, **kwargs):
    captured["train_config"] = train_config
    captured["kwargs"] = kwargs
    return None, []

  monkeypatch.setattr(experiment, "load_cifar10_bandinv_dpmuon_config", lambda _: config)
  monkeypatch.setattr(experiment, "train_cifar10_bandinv_dpmuon", fake_train)
  assert experiment.run_cifar10_bandinv_dpmuon(CONFIG) == (None, [])
  assert captured["train_config"] == experiment._train_config(config)
  assert captured["kwargs"]["resume_checkpoint"] is None
  assert captured["kwargs"]["checkpoint_path"].name == "latest.pkl"
  assert captured["kwargs"]["metrics_path"].name == "metrics.csv"


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
  assert capsys.readouterr().out == "0\n"


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
      sys,
      "argv",
      ["run_cifar10.py", "--config", "naive.yaml", "--resume-checkpoint", "checkpoint.pkl"],
  )
  run_cifar10.main()
  monkeypatch.setattr(
      sys,
      "argv",
      ["run_cifar10.py", "--config", "naive.yaml", "--run-dir", "logs/run"],
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
