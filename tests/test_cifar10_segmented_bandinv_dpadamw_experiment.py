"""Configuration and artifact-cache tests for segmented CIFAR-10 DP-AdamW."""

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.training import cifar10_segmented_bandinv_dpadamw_experiment as experiment
from dp_muon.training.cifar10_experiment import FixedCycleParticipation
from scripts import run_cifar10


CONFIG = Path("config/cifar10_bandinv_dpadamw_segmented.yaml")


def test_segmented_yaml_and_cli_dispatch_parse():
  config = experiment.load_cifar10_segmented_bandinv_dpadamw_config(CONFIG)
  assert config.algorithm == "dp-adamw-correlated-segmented"
  assert config.segment_length == 97
  assert run_cifar10._config_algorithm(str(CONFIG)) == config.algorithm


def test_unique_segment_lengths_are_fitted_once_and_cached(tmp_path, monkeypatch):
  config = replace(
      experiment.load_cifar10_segmented_bandinv_dpadamw_config(CONFIG),
      segment_length=2,
      strategy_dir=str(tmp_path / "strategies"),
  )
  participation = FixedCycleParticipation(5, 3, 2, 1.0)
  calls = []

  def fake_fit(horizon, bandwidth, min_sep, **kwargs):
    calls.append((horizon, bandwidth, min_sep))
    return BandInvMFStrategy(
        horizon=horizon,
        bandwidth=bandwidth,
        min_sep=min_sep,
        max_participations=kwargs["max_participations"],
        workload_coef=jnp.asarray(kwargs["workload_coef"]),
        noising_coef=jnp.ones(bandwidth),
        strategy_coef=jnp.ones(horizon),
        sensitivity_squared=jnp.array(1.0),
        objective=jnp.array(1.0),
    )

  monkeypatch.setattr(experiment, "fit_bandinv_strategy", fake_fit)
  plan, snapshots, actions = experiment.get_or_fit_segmented_plan(config, participation)
  assert plan.block_lengths == (2, 2, 1)
  assert calls == [(1, 1, 1), (2, 2, 2)]
  assert actions == {1: "fit", 2: "fit"}
  assert set(snapshots) == {1, 2}
  _, _, resumed_actions = experiment.get_or_fit_segmented_plan(
      config, participation, require_existing=True
  )
  assert resumed_actions == {1: "reuse", 2: "reuse"}


def test_segmented_cli_prints_without_training(monkeypatch, capsys):
  monkeypatch.setattr(
      sys, "argv", ["run_cifar10.py", "--config", str(CONFIG), "--print-gpu"]
  )
  run_cifar10.main()
  assert capsys.readouterr().out == "0\n"
