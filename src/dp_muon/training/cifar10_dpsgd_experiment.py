"""YAML orchestration for the non-amplified IID DP-SGD-Momentum baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.data import load_cifar10
from dp_muon.privacy import calibrate_nonamplified_iid

from .cifar10_driver import (
    Cifar10DPSGDMomentumTrainConfig,
    train_cifar10_dpsgd_momentum,
)
from .cifar10_experiment import FixedCycleParticipation, derive_fixed_cycle_participation


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Cifar10DPSGDMomentumExperimentConfig:
  name: str
  seed: int
  gpu: int
  data_dir: str
  pretrained: str
  epochs: int
  logical_batch_size: int
  microbatch_size: int
  momentum: float
  learning_rate: float
  clip_norm: float
  eval_every: int
  schedule_mode: Literal["fixed_cycle"]
  epsilon: float
  delta: float
  adjacency: Literal["add_remove", "replace_one"]
  checkpoint_dir: str
  log_dir: str


def _positive_int(value: object, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
    raise ValueError(f"{name} must be a positive integer")
  return int(value)


def _nonnegative_int(value: object, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
    raise ValueError(f"{name} must be a non-negative integer")
  return int(value)


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
  if isinstance(value, bool) or not isinstance(value, Real):
    raise ValueError(f"{name} must be a finite number")
  result = float(value)
  if not math.isfinite(result) or (positive and result <= 0):
    raise ValueError(f"{name} must be a{' positive' if positive else ''} finite number")
  return result


def _string(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{name} must be a non-empty string")
  return value


def _section(document: Mapping[str, Any], name: str, keys: set[str]) -> Mapping[str, Any]:
  value = document.get(name)
  if not isinstance(value, Mapping):
    raise ValueError(f"config.{name} must be a mapping")
  missing, extra = keys.difference(value), set(value).difference(keys)
  if missing or extra:
    raise ValueError(
        f"config.{name} must contain exactly {sorted(keys)} "
        f"(missing={sorted(missing)}, extra={sorted(extra)})"
    )
  return value


def load_cifar10_dpsgd_momentum_config(
    path: str | Path,
) -> Cifar10DPSGDMomentumExperimentConfig:
  """Loads the strategy-free YAML schema for IID DP-SGD-Momentum."""
  source = Path(path)
  try:
    with source.open(encoding="utf-8") as stream:
      document = yaml.safe_load(stream)
  except OSError as error:
    raise ValueError(f"could not read config {source}") from error
  except yaml.YAMLError as error:
    raise ValueError(f"could not parse YAML config {source}") from error
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  expected_sections = {
      "experiment", "runtime", "data", "model", "training", "schedule", "privacy", "output"
  }
  if set(document) != expected_sections:
    raise ValueError(f"config sections must be exactly {sorted(expected_sections)}")
  experiment = _section(document, "experiment", {"name", "seed"})
  runtime = _section(document, "runtime", {"gpu"})
  data = _section(document, "data", {"data_dir"})
  model = _section(document, "model", {"pretrained"})
  training = _section(
      document,
      "training",
      {"epochs", "logical_batch_size", "microbatch_size", "momentum", "learning_rate", "clip_norm", "eval_every"},
  )
  schedule = _section(document, "schedule", {"mode"})
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _section(document, "output", {"checkpoint_dir", "log_dir"})
  mode = _string(schedule["mode"], "schedule.mode")
  if mode != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  adjacency = _string(privacy["adjacency"], "privacy.adjacency")
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("privacy.adjacency must be 'add_remove' or 'replace_one'")
  momentum = _finite_float(training["momentum"], "training.momentum")
  if not 0.0 <= momentum < 1.0:
    raise ValueError("training.momentum must be in [0, 1)")
  config = Cifar10DPSGDMomentumExperimentConfig(
      name=_string(experiment["name"], "experiment.name"),
      seed=_nonnegative_int(experiment["seed"], "experiment.seed"),
      gpu=_nonnegative_int(runtime["gpu"], "runtime.gpu"),
      data_dir=_string(data["data_dir"], "data.data_dir"),
      pretrained=_string(model["pretrained"], "model.pretrained"),
      epochs=_positive_int(training["epochs"], "training.epochs"),
      logical_batch_size=_positive_int(training["logical_batch_size"], "training.logical_batch_size"),
      microbatch_size=_positive_int(training["microbatch_size"], "training.microbatch_size"),
      momentum=momentum,
      learning_rate=_finite_float(training["learning_rate"], "training.learning_rate", positive=True),
      clip_norm=_finite_float(training["clip_norm"], "training.clip_norm", positive=True),
      eval_every=_positive_int(training["eval_every"], "training.eval_every"),
      schedule_mode=mode,  # type: ignore[arg-type]
      epsilon=_finite_float(privacy["epsilon"], "privacy.epsilon", positive=True),
      delta=_finite_float(privacy["delta"], "privacy.delta", positive=True),
      adjacency=adjacency,  # type: ignore[arg-type]
      checkpoint_dir=_string(output["checkpoint_dir"], "output.checkpoint_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
  )
  if config.delta >= 1.0:
    raise ValueError("privacy.delta must be less than 1")
  if config.logical_batch_size % config.microbatch_size != 0:
    raise ValueError("training.logical_batch_size must be divisible by training.microbatch_size")
  return config


def resolve_output_log_dir(config_path: str | Path) -> Path:
  log_dir = Path(load_cifar10_dpsgd_momentum_config(config_path).log_dir)
  return log_dir if log_dir.is_absolute() else REPOSITORY_ROOT / log_dir


def _print_resolved_config(
    config: Cifar10DPSGDMomentumExperimentConfig,
    num_examples: int,
    participation: FixedCycleParticipation,
) -> None:
  calibration = calibrate_nonamplified_iid(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.logical_batch_size),
      adjacency=config.adjacency,
      max_participations=participation.max_participations,
  )
  print(
      "\n".join((
          f"Dataset size: {num_examples}",
          "Runtime:",
          f"  physical GPU: {config.gpu}",
          f"epochs: {config.epochs}",
          f"logical_batch_size: {config.logical_batch_size}",
          f"microbatch_size: {config.microbatch_size}",
          "",
          "Derived:",
          f"  horizon n: {participation.horizon}",
          f"  max_participations k: {participation.max_participations}",
          f"  min_sep b: {participation.min_sep}",
          f"  effective_epochs: {participation.effective_epochs:.10g}",
          "",
          "IID DP-SGD-Momentum:",
          f"  momentum: {config.momentum}",
          f"  learning_rate: {config.learning_rate}",
          f"  per-step IID noise std tau: {calibration.iid_noise_std:.10g}",
      ))
  )


def run_cifar10_dpsgd_momentum(config_path: str | Path):
  """Derives the fixed-cycle privacy bound and runs IID DP-SGD-Momentum."""
  config = load_cifar10_dpsgd_momentum_config(config_path)
  log_dir = Path(config.log_dir)
  (log_dir if log_dir.is_absolute() else REPOSITORY_ROOT / log_dir).mkdir(
      parents=True, exist_ok=True
  )
  train_images, _ = load_cifar10(config.data_dir, train=True)
  num_examples = len(train_images)
  participation = derive_fixed_cycle_participation(
      num_examples, config.epochs, config.logical_batch_size
  )
  del train_images
  _print_resolved_config(config, num_examples, participation)
  train_config = Cifar10DPSGDMomentumTrainConfig(
      pretrained=config.pretrained,
      data_dir=config.data_dir,
      batch_size=config.logical_batch_size,
      microbatch_size=config.microbatch_size,
      clip_norm=config.clip_norm,
      epsilon=config.epsilon,
      delta=config.delta,
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      seed=config.seed,
      checkpoint_dir=config.checkpoint_dir,
      eval_every=config.eval_every,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      adjacency=config.adjacency,
  )
  return train_cifar10_dpsgd_momentum(train_config)


__all__ = [
    "Cifar10DPSGDMomentumExperimentConfig",
    "load_cifar10_dpsgd_momentum_config",
    "resolve_output_log_dir",
    "run_cifar10_dpsgd_momentum",
]
