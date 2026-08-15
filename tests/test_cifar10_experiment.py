from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from dp_muon.bandinvmf import BandInvMFStrategy, save_bandinv_strategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.training import cifar10_experiment as experiment
from scripts import run_cifar10 as run_cifar10_script


CONFIG = Path("config/cifar10_nonamplified.yaml")


def _config(tmp_path, **changes):
  config = experiment.load_cifar10_nonamplified_config(CONFIG)
  return replace(
      config,
      strategy_dir=str(tmp_path / "strategies"),
      checkpoint_dir=str(tmp_path / "checkpoints"),
      log_dir=str(tmp_path / "logs"),
      **changes,
  )


def _strategy(config, participation):
  workload = fixed_lr_nesterov_trajectory_workload_coef(
      participation.horizon, config.momentum, config.learning_rate
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


def _save_matching_artifact(config, participation):
  path = experiment.strategy_artifact_path(
      config.strategy_dir,
      participation=participation,
      bandwidth=config.bandwidth,
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  save_bandinv_strategy(
      path,
      _strategy(config, participation),
      reduction=config.reduction,
      workload_type="nesterov-trajectory",
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  return path


def test_derives_cifar10_fixed_cycle_participation():
  actual = experiment.derive_fixed_cycle_participation(50_000, 10, 512)
  assert actual.horizon == 976
  assert actual.max_participations == 10
  assert actual.min_sep == 97
  assert actual.effective_epochs == pytest.approx(9.99424)


@pytest.mark.parametrize(
    ("epochs", "batch_size"), [(0, 512), (-1, 512), (10, 0), (10, 50_001)]
)
def test_derive_fixed_cycle_rejects_invalid_inputs(epochs, batch_size):
  with pytest.raises(ValueError):
    experiment.derive_fixed_cycle_participation(50_000, epochs, batch_size)


def test_parses_default_yaml_without_manual_participation_values():
  config = experiment.load_cifar10_nonamplified_config(CONFIG)
  assert config.epochs == 10
  assert config.logical_batch_size == 512
  assert config.momentum == 0.9
  assert config.learning_rate == 0.5
  assert config.adjacency == "add_remove"


def test_yaml_runner_rejects_replace_one_adjacency(tmp_path):
  config_path = tmp_path / "replace_one.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace(
          "adjacency: add_remove", "adjacency: replace_one"
      ),
      encoding="utf-8",
  )
  with pytest.raises(
      ValueError,
      match="CIFAR-10 YAML runner currently supports only adjacency='add_remove'",
  ):
    experiment.load_cifar10_nonamplified_config(config_path)


def test_print_log_dir_cli_resolves_relative_path_from_repository_root(
    tmp_path, monkeypatch, capsys
):
  config_path = tmp_path / "logs.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace("log_dir: logs", "log_dir: ci-logs"),
      encoding="utf-8",
  )
  monkeypatch.setattr(
      sys,
      "argv",
      ["run_cifar10.py", "--config", str(config_path), "--print-log-dir"],
  )
  run_cifar10_script.main()
  assert capsys.readouterr().out.strip() == str(experiment.REPOSITORY_ROOT / "ci-logs")


def test_print_log_dir_cli_preserves_absolute_path(tmp_path, monkeypatch, capsys):
  absolute_log_dir = tmp_path / "absolute-logs"
  config_path = tmp_path / "absolute.yaml"
  config_path.write_text(
      CONFIG.read_text(encoding="utf-8").replace(
          "log_dir: logs", f"log_dir: {absolute_log_dir}"
      ),
      encoding="utf-8",
  )
  monkeypatch.setattr(
      sys,
      "argv",
      ["run_cifar10.py", "--config", str(config_path), "--print-log-dir"],
  )
  run_cifar10_script.main()
  assert capsys.readouterr().out.strip() == str(absolute_log_dir)


def test_runner_cli_does_not_print_history_after_training(monkeypatch, capsys):
  calls = []

  def fake_run(config_path):
    calls.append(config_path)
    return None, [{"step": 50, "accuracy": 0.8}]

  monkeypatch.setattr(run_cifar10_script, "run_cifar10_nonamplified", fake_run)
  monkeypatch.setattr(sys, "argv", ["run_cifar10.py", "--config", "experiment.yaml"])
  run_cifar10_script.main()
  assert calls == ["experiment.yaml"]
  assert capsys.readouterr().out == ""


def test_matching_artifact_is_reused_without_optimizer(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = experiment.derive_fixed_cycle_participation(50_000, 10, 512)
  path = _save_matching_artifact(config, participation)

  def fail_optimizer(*args, **kwargs):
    raise AssertionError("optimizer must not run for an exactly matching artifact")

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fail_optimizer)
  actual_path, _, action = experiment.get_or_fit_strategy(config, participation)
  assert actual_path == path
  assert action == "reuse"


def test_strategy_identity_changes_for_each_defining_parameter(tmp_path):
  config = _config(tmp_path)
  participation = experiment.derive_fixed_cycle_participation(50_000, 10, 512)
  base = experiment.strategy_artifact_path(
      config.strategy_dir,
      participation=participation,
      bandwidth=config.bandwidth,
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  variants = (
      dict(participation=replace(participation, horizon=975)),
      dict(participation=replace(participation, min_sep=96)),
      dict(participation=replace(participation, max_participations=9)),
      dict(bandwidth=3),
      dict(momentum=0.8),
      dict(learning_rate=0.4),
      dict(reduction="last"),
      dict(max_optimizer_steps=999),
  )
  for overrides in variants:
    path = experiment.strategy_artifact_path(
        config.strategy_dir,
        participation=overrides.get("participation", participation),
        bandwidth=overrides.get("bandwidth", config.bandwidth),
        momentum=overrides.get("momentum", config.momentum),
        learning_rate=overrides.get("learning_rate", config.learning_rate),
        reduction=overrides.get("reduction", config.reduction),
        max_optimizer_steps=overrides.get("max_optimizer_steps", config.max_optimizer_steps),
    )
    assert path != base


def test_metadata_mismatch_is_refit_instead_of_reused(tmp_path, monkeypatch):
  config = _config(tmp_path)
  participation = experiment.derive_fixed_cycle_participation(50_000, 10, 512)
  path = experiment.strategy_artifact_path(
      config.strategy_dir,
      participation=participation,
      bandwidth=config.bandwidth,
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  save_bandinv_strategy(
      path,
      _strategy(config, participation),
      reduction="last",
      workload_type="nesterov-trajectory",
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  _, _, action = experiment.get_or_fit_strategy(config, participation)
  assert action == "fit"
  assert len(calls) == 1


def test_force_refit_runs_optimizer_even_for_matching_artifact(tmp_path, monkeypatch):
  config = _config(tmp_path, force_refit=True)
  participation = experiment.derive_fixed_cycle_participation(50_000, 10, 512)
  _save_matching_artifact(config, participation)
  calls = []

  def fake_fit(*args, **kwargs):
    calls.append((args, kwargs))
    return _strategy(config, participation)

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  _, _, action = experiment.get_or_fit_strategy(config, participation)
  assert action == "fit"
  assert len(calls) == 1


def test_runner_uses_same_momentum_learning_rate_for_fit_and_training(tmp_path, monkeypatch):
  config_path = tmp_path / "experiment.yaml"
  config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
  captured = {}
  config = _config(tmp_path)
  participation = experiment.derive_fixed_cycle_participation(50_000, 10, 512)
  strategy = _strategy(config, participation)

  monkeypatch.setattr(
      experiment,
      "load_cifar10_nonamplified_config",
      lambda _: config,
  )
  monkeypatch.setattr(
      experiment,
      "load_cifar10",
      lambda *args, **kwargs: (np.empty((50_000, 32, 32, 3), np.uint8), np.empty(50_000, np.int32)),
  )

  def fake_get_or_fit(actual_config, actual_participation):
    captured["fit_config"] = actual_config
    captured["fit_participation"] = actual_participation
    return Path(actual_config.strategy_dir) / "strategy.npz", strategy, "reuse"

  def fake_validate(actual_strategy, calibration, spec, momentum, learning_rate):
    captured["validated"] = (actual_strategy, spec, momentum, learning_rate)

  def fake_train(train_config):
    captured["train"] = train_config
    return None, []

  monkeypatch.setattr(experiment, "get_or_fit_strategy", fake_get_or_fit)
  monkeypatch.setattr(experiment, "validate_nonamplified_bandinv_setup", fake_validate)
  monkeypatch.setattr(experiment, "train_cifar10", fake_train)
  experiment.run_cifar10_nonamplified(config_path)

  assert captured["fit_config"].momentum == captured["train"].momentum == 0.9
  assert captured["fit_config"].learning_rate == captured["train"].learning_rate == 0.5
  assert captured["validated"][2:] == (0.9, 0.5)
  assert captured["validated"][1].horizon == 976
