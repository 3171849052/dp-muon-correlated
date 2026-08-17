"""Configuration loading and runner dispatch for IID DP-AdamW CIFAR-10."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from dp_muon.training import cifar10_experiment
from dp_muon.training.cifar10_dpadamw_experiment import (
    Cifar10DPAdamWExperimentConfig,
    load_cifar10_dpadamw_config,
    prepare_cifar10_dpadamw_run,
    resolve_output_log_dir,
    run_cifar10_dpadamw,
)
from dp_muon.training.cifar10_driver import Cifar10DPAdamWTrainConfig
from scripts import run_cifar10


CONFIG = Path("config/cifar10_dpadamw.yaml")


def test_parses_default_yaml():
  config = load_cifar10_dpadamw_config(CONFIG)
  assert config.algorithm == "dpadamw"
  assert config.epochs == 5
  assert config.logical_batch_size == 512
  assert config.learning_rate == 0.0005
  assert config.beta1 == 0.9
  assert config.beta2 == 0.999
  assert config.eps == 1e-8
  assert config.weight_decay == 0.01
  assert config.adjacency == "add_remove"
  assert config.gpu == 3


def test_derives_fixed_cycle_participation():
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  assert (participation.horizon, participation.min_sep, participation.max_participations) == (488, 97, 5)


def test_yaml_runner_rejects_invalid_adjacency(tmp_path):
  config_path = tmp_path / "bad_adjacency.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("adjacency: add_remove", "adjacency: replace_one"),
      encoding="utf-8",
  )
  config = load_cifar10_dpadamw_config(config_path)
  assert config.adjacency == "replace_one"


def test_yaml_runner_rejects_unknown_algorithm(tmp_path):
  config_path = tmp_path / "unknown.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("dpadamw", "unknown")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="DP-AdamW config requires algorithm: dpadamw"):
    load_cifar10_dpadamw_config(config_path)


def test_yaml_runner_rejects_missing_sections(tmp_path):
  config_path = tmp_path / "missing_section.yaml"
  config_path.write_text("algorithm: dpadamw\n", encoding="utf-8")
  with pytest.raises(ValueError, match="config sections must be exactly"):
    load_cifar10_dpadamw_config(config_path)


def test_yaml_runner_rejects_invalid_batch_divisibility(tmp_path):
  config_path = tmp_path / "bad_batch.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("microbatch_size: 16", "microbatch_size: 17")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="divisible"):
    load_cifar10_dpadamw_config(config_path)


def test_yaml_runner_rejects_invalid_delta(tmp_path):
  config_path = tmp_path / "bad_delta.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("delta: 1.0e-5", "delta: 1.0")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="delta must be less than 1"):
    load_cifar10_dpadamw_config(config_path)


def test_yaml_runner_rejects_invalid_beta(tmp_path):
  config_path = tmp_path / "bad_beta.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("beta1: 0.9", "beta1: 1.5")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="beta1 must be in"):
    load_cifar10_dpadamw_config(config_path)


def test_print_log_dir_cli_resolves_relative_path(tmp_path, monkeypatch, capsys):
  config_path = tmp_path / "logs.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("log_dir: logs", "log_dir: ci-logs"),
      encoding="utf-8",
  )
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", str(config_path), "--print-log-dir"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out.strip() == str(resolve_output_log_dir(config_path))


def test_print_gpu_cli_outputs_gpu_without_starting_training(monkeypatch, capsys):
  def fail_training(config_path):
    raise AssertionError("--print-gpu must not start training")
  monkeypatch.setattr(run_cifar10, "run_cifar10_dpadamw", fail_training)
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", str(CONFIG), "--print-gpu"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out == "3\n"


def test_prepare_run_cli_returns_run_directory(monkeypatch, capsys, tmp_path):
  def fake_prepare(config_path):
    return SimpleNamespace(directory=tmp_path / "run")
  monkeypatch.setattr(run_cifar10, "prepare_cifar10_dpadamw_run", fake_prepare)
  monkeypatch.setattr(run_cifar10, "_config_algorithm", lambda _: "dpadamw")
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", "experiment.yaml", "--prepare-run"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out == f"{tmp_path / 'run'}\n"


def test_prepare_run_snapshots_without_training(tmp_path):
  config_path = tmp_path / "dpadamw.yaml"
  log_dir = tmp_path / "runs"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("log_dir: logs", f"log_dir: {log_dir}"),
      encoding="utf-8",
  )
  paths = prepare_cifar10_dpadamw_run(config_path)
  assert paths.directory.parent == log_dir
  assert paths.config.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
  assert paths.resolved_config.is_file()
  assert paths.metrics.is_file()


def test_run_dir_cli_is_forwarded(monkeypatch):
  calls = []
  def fake_run(config_path, *, resume_checkpoint=None, run_dir=None):
    calls.append((config_path, resume_checkpoint, run_dir))
  monkeypatch.setattr(run_cifar10, "run_cifar10_dpadamw", fake_run)
  monkeypatch.setattr(run_cifar10, "_config_algorithm", lambda _: "dpadamw")
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", "experiment.yaml", "--run-dir", "logs/current"]
  )
  run_cifar10.main()
  assert calls == [("experiment.yaml", None, "logs/current")]


def test_runner_uses_same_config_for_train_config(tmp_path, monkeypatch):
  config = load_cifar10_dpadamw_config(CONFIG)
  config = replace(config, checkpoint_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  captured = {}

  monkeypatch.setattr(
      "dp_muon.training.cifar10_dpadamw_experiment.load_cifar10_dpadamw_config",
      lambda _: config,
  )
  monkeypatch.setattr(
      "dp_muon.training.cifar10_dpadamw_experiment.load_cifar10",
      lambda *args, **kwargs: (np.empty((50_000, 32, 32, 3), np.uint8), np.empty(50_000, np.int32)),
  )
  import numpy as np

  def fake_train(train_config, **kwargs):
    captured["train_config"] = train_config
    captured["kwargs"] = kwargs
    return None, []

  monkeypatch.setattr(
      "dp_muon.training.cifar10_dpadamw_experiment.train_cifar10_dpadamw", fake_train
  )
  config_path = tmp_path / "dpadamw.yaml"
  config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
  run_cifar10_dpadamw(config_path)
  tc = captured["train_config"]
  assert isinstance(tc, Cifar10DPAdamWTrainConfig)
  assert tc.learning_rate == config.learning_rate
  assert tc.beta1 == config.beta1
  assert tc.beta2 == config.beta2
  assert tc.eps == config.eps
  assert tc.weight_decay == config.weight_decay
  assert tc.horizon == 488
  assert tc.microbatch_size == 16
  assert tc.batch_size == 512
  assert tc.adjacency == "add_remove"


def test_same_config_creates_new_run_dir_each_time(tmp_path, monkeypatch):
  config = load_cifar10_dpadamw_config(CONFIG)
  config = replace(config, checkpoint_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
  monkeypatch.setattr(
      "dp_muon.training.cifar10_dpadamw_experiment.load_cifar10_dpadamw_config",
      lambda _: config,
  )
  monkeypatch.setattr(
      "dp_muon.training.cifar10_dpadamw_experiment.load_cifar10",
      lambda *args, **kwargs: (
          np.empty((50_000, 32, 32, 3), np.uint8), np.empty(50_000, np.int32)
      ),
  )
  import numpy as np
  calls = []
  monkeypatch.setattr(
      "dp_muon.training.cifar10_dpadamw_experiment.train_cifar10_dpadamw",
      lambda *args, **kwargs: calls.append(kwargs) or (None, []),
  )
  config_path = tmp_path / "dpadamw.yaml"
  config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
  run_cifar10_dpadamw(config_path)
  run_cifar10_dpadamw(config_path)
  run_dirs = {call["checkpoint_path"].parent.parent for call in calls}
  assert len(run_dirs) == 2
  assert all((directory / "config.yaml").is_file() for directory in run_dirs)