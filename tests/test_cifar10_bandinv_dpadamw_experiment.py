"""Configuration and strategy orchestration for naive correlated DP-AdamW."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from dp_muon.bandinvmf import BandInvMFStrategy, save_bandinv_strategy
from dp_muon.training.bandinvmf_strategy_manager import LoadedStrategySnapshot
from dp_muon.training.cifar10_driver import (
    BANDINV_DPADAMW_ALGORITHM,
    Cifar10BandInvDPAdamWTrainConfig,
)
from dp_muon.training import cifar10_bandinv_dpadamw_experiment as experiment
from dp_muon.training import cifar10_experiment
from scripts import run_cifar10


CONFIG = Path("config/cifar10_bandinv_dpadamw_naive.yaml")


def _config(tmp_path, **changes):
  defaults = {
      "strategy_dir": str(tmp_path / "strategies"),
      "checkpoint_dir": str(tmp_path / "checkpoints"),
      "log_dir": str(tmp_path / "logs"),
  }
  defaults.update(changes)
  return replace(
      experiment.load_cifar10_bandinv_dpadamw_config(CONFIG),
      **defaults,
  )


def _strategy(config, participation):
  return BandInvMFStrategy(
      horizon=participation.horizon,
      bandwidth=config.bandwidth,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      workload_coef=jnp.ones((participation.horizon,), jnp.float32),
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
      workload_type="prefix-sum",
      momentum=None,
      learning_rate=None,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  return path


# --- Test 1: YAML parses and maps to trainer fields ---

def test_naive_yaml_parses_and_maps_trainer_fields(tmp_path):
  config = _config(tmp_path)
  assert config.algorithm == BANDINV_DPADAMW_ALGORITHM
  assert config.epochs == 5
  assert config.schedule_mode == "fixed_cycle"
  train_config = experiment._train_config(config, tmp_path / "strategy.npz")
  assert isinstance(train_config, Cifar10BandInvDPAdamWTrainConfig)
  for field in asdict(train_config):
    if field != "strategy":
      assert getattr(train_config, field) == getattr(config, field)
  assert train_config.strategy == str(tmp_path / "strategy.npz")


# --- Test 2: derives fixed-cycle participation ---

def test_derives_required_cifar10_fixed_cycle_participation():
  actual = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  assert (actual.horizon, actual.min_sep, actual.max_participations) == (488, 97, 5)


# --- Test 3: missing artifact is fitted and saved ---

def test_missing_artifact_is_fitted_and_saved(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  snapshot, action = experiment.get_or_fit_strategy_snapshot(config, participation)
  assert action == "fit"
  assert snapshot.path.is_file()
  assert len(calls) == 1


# --- Test 4: compatible artifact is reused ---

def test_compatible_artifact_is_reused(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  _save_matching_artifact(config, participation)
  monkeypatch.setattr(
      experiment,
      "fit_bandinv_strategy",
      lambda *args, **kwargs: pytest.fail("compatible artifact must be reused"),
  )
  snapshot, action = experiment.get_or_fit_strategy_snapshot(config, participation)
  assert action == "reuse"


# --- Test 5: metadata mismatch refits ---

def test_metadata_mismatch_refits(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  _save_matching_artifact(config, participation, reduction="last")
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  _, action = experiment.get_or_fit_strategy_snapshot(config, participation)
  assert action == "fit"
  assert len(calls) == 1


# --- Test 6: force_refit ignores compatible artifact ---

def test_force_refit_ignores_compatible_artifact(tmp_path, monkeypatch):
  config = _config(tmp_path, force_refit=True)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  _save_matching_artifact(config, participation)
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  _, action = experiment.get_or_fit_strategy_snapshot(config, participation)
  assert action == "fit"
  assert len(calls) == 1


# --- Test 7: relative strategy dir resolved from repo root ---

def test_relative_strategy_dir_is_resolved_from_repository_root(tmp_path, monkeypatch):
  config = _config(tmp_path, strategy_dir="ci-relative-strategies")
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  monkeypatch.chdir(tmp_path)
  path = experiment.strategy_artifact_path(config, participation)
  assert path.parent == experiment.REPOSITORY_ROOT / "ci-relative-strategies"


# --- Test 8: run fits strategy, writes resolved config, passes final path ---

def test_run_fits_strategy_writes_resolved_config_and_passes_final_path(tmp_path, monkeypatch):
  config = _config(tmp_path)
  config_path = tmp_path / "naive.yaml"
  config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  strategy_path = tmp_path / "strategies" / "strategy.npz"
  captured = {}

  monkeypatch.setattr(experiment, "load_cifar10_bandinv_dpadamw_config", lambda _: config)
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

  monkeypatch.setattr(experiment, "train_cifar10_bandinv_dpadamw", fake_train)
  assert experiment.run_cifar10_bandinv_dpadamw(config_path) == (None, [])
  assert captured["train_config"].strategy == str(strategy_path)
  resolved = yaml.safe_load(
      (captured["kwargs"]["checkpoint_path"].parent.parent / "resolved_config.yaml").read_text()
  )
  assert resolved["participation"]["horizon"] == 488
  assert resolved["strategy"]["artifact"] == str(strategy_path.resolve())
  assert resolved["strategy"]["action"] == "fit"
  assert resolved["privacy_calibration"]


# --- Test 9: prepare run snapshots without loading data or training ---

def test_prepare_run_snapshots_without_loading_data_or_training(tmp_path):
  config_path = tmp_path / "naive.yaml"
  log_dir = tmp_path / "runs"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("log_dir: logs", f"log_dir: {log_dir}"),
      encoding="utf-8",
  )
  paths = experiment.prepare_cifar10_bandinv_dpadamw_run(config_path)
  assert paths.directory.parent == log_dir
  assert paths.config.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
  assert paths.resolved_config.is_file()
  assert paths.metrics.is_file()


# --- Test 10: CLI print-log-dir and print-gpu ---

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


# --- Test 11: CLI prepare and run dispatch ---

def test_cli_prepare_and_run_dispatch_to_naive_adapter(tmp_path, monkeypatch, capsys):
  calls = []

  def fake_prepare(path):
    calls.append(("prepare", path))
    return type("Paths", (), {"directory": tmp_path / "run"})()

  def fake_run(path, *, resume_checkpoint=None, run_dir=None):
    calls.append(("run", path, resume_checkpoint, run_dir))

  monkeypatch.setattr(run_cifar10, "prepare_cifar10_bandinv_dpadamw_run", fake_prepare)
  monkeypatch.setattr(run_cifar10, "run_cifar10_bandinv_dpadamw", fake_run)
  monkeypatch.setattr(run_cifar10, "_config_algorithm", lambda _: BANDINV_DPADAMW_ALGORITHM)
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


# --- Test 12: YAML schema validation ---

def test_yaml_rejects_missing_sections(tmp_path):
  config_path = tmp_path / "missing.yaml"
  config_path.write_text("algorithm: dp-adamw-correlated-naive\n", encoding="utf-8")
  with pytest.raises(ValueError, match="config sections must be exactly"):
    experiment.load_cifar10_bandinv_dpadamw_config(config_path)


def test_yaml_rejects_invalid_batch_divisibility(tmp_path):
  config_path = tmp_path / "bad_batch.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("microbatch_size: 16", "microbatch_size: 17")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="divisible"):
    experiment.load_cifar10_bandinv_dpadamw_config(config_path)


def test_yaml_rejects_invalid_delta(tmp_path):
  config_path = tmp_path / "bad_delta.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("delta: 1.0e-5", "delta: 1.0")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="delta must be less than 1"):
    experiment.load_cifar10_bandinv_dpadamw_config(config_path)


def test_yaml_rejects_invalid_adjacency(tmp_path):
  config_path = tmp_path / "bad_adj.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("adjacency: add_remove", "adjacency: invalid")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="privacy.adjacency"):
    experiment.load_cifar10_bandinv_dpadamw_config(config_path)


def test_yaml_rejects_unknown_algorithm(tmp_path):
  config_path = tmp_path / "unknown.yaml"
  content = CONFIG.read_text(encoding="utf-8").replace("dp-adamw-correlated-naive", "unknown")
  config_path.write_text(content, encoding="utf-8")
  with pytest.raises(ValueError, match="DP-AdamW config requires algorithm"):
    experiment.load_cifar10_bandinv_dpadamw_config(config_path)


# --- Test 13: prefix-sum artifact does not use momentum/learning_rate ---

def test_strategy_artifact_path_has_no_momentum_or_lr(tmp_path):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  path = experiment.strategy_artifact_path(config, participation)
  assert "prefix-sum" in path.name
  assert "_m" not in path.name
  assert "_lr" not in path.name


# --- Test 14: workload uses prefix-sum (all-ones), not nesterov ---

def test_strategy_workload_is_prefix_sum_not_nesterov(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = cifar10_experiment.derive_fixed_cycle_participation(50_000, 5, 512)
  captured = {}

  def fake_fit(*args, **kwargs):
    captured["workload"] = kwargs.get("workload_coef")
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  experiment.get_or_fit_strategy_snapshot(config, participation)
  # The manager must pass None so fit_bandinv_strategy uses its default
  # prefix-sum (all-ones) workload instead of a Nesterov trajectory.
  assert captured["workload"] is None


# --- Test 15: AdamW defaults match IID DP-AdamW ---

def test_adamw_defaults_match_iid_standard(tmp_path):
  config = _config(tmp_path)
  assert config.learning_rate == 0.0005
  assert config.beta1 == 0.9
  assert config.beta2 == 0.999
  assert config.eps == 1e-8
  assert config.weight_decay == 0.01