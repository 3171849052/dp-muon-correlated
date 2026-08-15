"""YAML-driven CIFAR-10 non-amplified BandInvMF experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml

from dp_muon.bandinvmf import (
    BandInvMFStrategy,
    fit_bandinv_strategy,
    load_bandinv_strategy,
    load_bandinv_strategy_metadata,
    save_bandinv_strategy,
)
from dp_muon.data import load_cifar10
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv

from .cifar10_driver import Cifar10TrainConfig, train_cifar10
from .nonamplified_linear import validate_nonamplified_bandinv_setup


@dataclass(frozen=True)
class FixedCycleParticipation:
  """Public fixed-cycle values derived from epoch and logical batch settings."""

  horizon: int
  max_participations: int
  min_sep: int
  effective_epochs: float


@dataclass(frozen=True)
class Cifar10NonAmplifiedExperimentConfig:
  name: str
  seed: int
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
  bandwidth: int
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  force_refit: bool
  strategy_dir: str
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
    qualifier = " positive finite" if positive else " finite"
    raise ValueError(f"{name} must be a{qualifier} number")
  return result


def _string(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{name} must be a non-empty string")
  return value


def _section(document: Mapping[str, Any], name: str, keys: set[str]) -> Mapping[str, Any]:
  value = document.get(name)
  if not isinstance(value, Mapping):
    raise ValueError(f"config.{name} must be a mapping")
  missing = keys.difference(value)
  extra = set(value).difference(keys)
  if missing or extra:
    raise ValueError(
        f"config.{name} must contain exactly {sorted(keys)} "
        f"(missing={sorted(missing)}, extra={sorted(extra)})"
    )
  return value


def derive_fixed_cycle_participation(
    num_examples: int, epochs: int, logical_batch_size: int
) -> FixedCycleParticipation:
  """Derives the M4 fixed-cycle contract without accepting hand-written n/k/b."""
  num_examples = _positive_int(num_examples, "num_examples")
  epochs = _positive_int(epochs, "epochs")
  logical_batch_size = _positive_int(logical_batch_size, "logical_batch_size")
  if logical_batch_size > num_examples:
    raise ValueError("logical_batch_size must not exceed num_examples")
  horizon = (epochs * num_examples) // logical_batch_size
  min_sep = num_examples // logical_batch_size
  return FixedCycleParticipation(
      horizon=horizon,
      max_participations=epochs,
      min_sep=min_sep,
      effective_epochs=horizon * logical_batch_size / num_examples,
  )


def load_cifar10_nonamplified_config(path: str | Path) -> Cifar10NonAmplifiedExperimentConfig:
  """Loads and strictly validates the one supported experiment schema."""
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
      "experiment", "data", "model", "training", "schedule", "privacy", "strategy", "output"
  }
  if set(document) != expected_sections:
    raise ValueError(f"config sections must be exactly {sorted(expected_sections)}")
  experiment = _section(document, "experiment", {"name", "seed"})
  data = _section(document, "data", {"data_dir"})
  model = _section(document, "model", {"pretrained"})
  training = _section(
      document,
      "training",
      {"epochs", "logical_batch_size", "microbatch_size", "momentum", "learning_rate", "clip_norm", "eval_every"},
  )
  schedule = _section(document, "schedule", {"mode"})
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  strategy = _section(
      document,
      "strategy",
      {"bandwidth", "reduction", "max_optimizer_steps", "force_refit"},
  )
  output = _section(document, "output", {"strategy_dir", "checkpoint_dir", "log_dir"})
  mode = _string(schedule["mode"], "schedule.mode")
  if mode != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  adjacency = _string(privacy["adjacency"], "privacy.adjacency")
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("privacy.adjacency must be 'add_remove' or 'replace_one'")
  reduction = _string(strategy["reduction"], "strategy.reduction")
  if reduction not in {"mean", "max", "last"}:
    raise ValueError("strategy.reduction must be one of: mean, max, last")
  momentum = _finite_float(training["momentum"], "training.momentum")
  if not 0.0 <= momentum < 1.0:
    raise ValueError("training.momentum must be in [0, 1)")
  config = Cifar10NonAmplifiedExperimentConfig(
      name=_string(experiment["name"], "experiment.name"),
      seed=_nonnegative_int(experiment["seed"], "experiment.seed"),
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
      bandwidth=_positive_int(strategy["bandwidth"], "strategy.bandwidth"),
      reduction=reduction,  # type: ignore[arg-type]
      max_optimizer_steps=_positive_int(strategy["max_optimizer_steps"], "strategy.max_optimizer_steps"),
      force_refit=strategy["force_refit"],
      strategy_dir=_string(output["strategy_dir"], "output.strategy_dir"),
      checkpoint_dir=_string(output["checkpoint_dir"], "output.checkpoint_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
  )
  if not isinstance(config.force_refit, bool):
    raise ValueError("strategy.force_refit must be a boolean")
  if config.delta >= 1.0:
    raise ValueError("privacy.delta must be less than 1")
  if config.logical_batch_size % config.microbatch_size != 0:
    raise ValueError("training.logical_batch_size must be divisible by training.microbatch_size")
  return config


def strategy_artifact_path(
    strategy_dir: str | Path,
    *,
    participation: FixedCycleParticipation,
    bandwidth: int,
    momentum: float,
    learning_rate: float,
    reduction: str,
    max_optimizer_steps: int,
) -> Path:
  """Returns a deterministic filename covering every strategy-defining value."""
  return Path(strategy_dir) / (
      f"nesterov-trajectory_n{participation.horizon}_p{bandwidth}"
      f"_b{participation.min_sep}_k{participation.max_participations}"
      f"_m{momentum}_lr{learning_rate}_r{reduction}_opt{max_optimizer_steps}.npz"
  )


def _strategy_is_compatible(
    strategy: BandInvMFStrategy,
    *,
    config: Cifar10NonAmplifiedExperimentConfig,
    participation: FixedCycleParticipation,
) -> bool:
  expected_workload = np.asarray(
      fixed_lr_nesterov_trajectory_workload_coef(
          participation.horizon, config.momentum, config.learning_rate
      )
  )
  return (
      strategy.horizon == participation.horizon
      and strategy.bandwidth == config.bandwidth
      and strategy.min_sep == participation.min_sep
      and strategy.max_participations == participation.max_participations
      and np.array_equal(np.asarray(strategy.workload_coef), expected_workload)
  )


def _metadata_is_compatible(
    path: Path, config: Cifar10NonAmplifiedExperimentConfig
) -> bool:
  metadata = load_bandinv_strategy_metadata(path)
  return (
      metadata.workload_type == "nesterov-trajectory"
      and metadata.momentum == config.momentum
      and metadata.learning_rate == config.learning_rate
      and metadata.reduction == config.reduction
      and metadata.max_optimizer_steps == config.max_optimizer_steps
  )


def get_or_fit_strategy(
    config: Cifar10NonAmplifiedExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[Path, BandInvMFStrategy, Literal["reuse", "fit"]]:
  """Returns an exactly compatible artifact or refits the deterministic path."""
  if config.bandwidth > participation.horizon:
    raise ValueError("strategy.bandwidth must not exceed derived horizon")
  path = strategy_artifact_path(
      config.strategy_dir,
      participation=participation,
      bandwidth=config.bandwidth,
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  if path.is_file() and not config.force_refit:
    try:
      existing = load_bandinv_strategy(path)
      if _strategy_is_compatible(
          existing, config=config, participation=participation
      ) and _metadata_is_compatible(path, config):
        return path, existing, "reuse"
    except ValueError:
      pass
  workload_coef = fixed_lr_nesterov_trajectory_workload_coef(
      participation.horizon, config.momentum, config.learning_rate
  )
  fitted = fit_bandinv_strategy(
      participation.horizon,
      config.bandwidth,
      participation.min_sep,
      max_participations=participation.max_participations,
      workload_coef=workload_coef,
      max_optimizer_steps=config.max_optimizer_steps,
      reduction=config.reduction,
  )
  save_bandinv_strategy(
      path,
      fitted,
      reduction=config.reduction,
      workload_type="nesterov-trajectory",
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  return path, fitted, "fit"


def _print_resolved_config(
    config: Cifar10NonAmplifiedExperimentConfig,
    num_examples: int,
    participation: FixedCycleParticipation,
    strategy_path: Path,
    action: str,
) -> None:
  print(
      "\n".join((
          f"Dataset size: {num_examples}",
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
          "BandInvMF:",
          f"  bandwidth p: {config.bandwidth}",
          f"  momentum: {config.momentum}",
          f"  learning_rate: {config.learning_rate}",
          f"  strategy path: {strategy_path}",
          f"  strategy action: {action}",
      ))
  )


def run_cifar10_nonamplified(config_path: str | Path):
  """Fits/reuses the public strategy, validates M6 setup, then trains CIFAR-10."""
  config = load_cifar10_nonamplified_config(config_path)
  Path(config.log_dir).mkdir(parents=True, exist_ok=True)
  train_images, _ = load_cifar10(config.data_dir, train=True)
  num_examples = len(train_images)
  del train_images
  participation = derive_fixed_cycle_participation(
      num_examples, config.epochs, config.logical_batch_size
  )
  strategy_path = strategy_artifact_path(
      config.strategy_dir,
      participation=participation,
      bandwidth=config.bandwidth,
      momentum=config.momentum,
      learning_rate=config.learning_rate,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  # Determine whether a compatible cache exists before printing the audited action.
  action = "fit"
  if strategy_path.is_file() and not config.force_refit:
    try:
      candidate = load_bandinv_strategy(strategy_path)
      if (
          _strategy_is_compatible(candidate, config=config, participation=participation)
          and _metadata_is_compatible(strategy_path, config)
      ):
        action = "reuse"
    except ValueError:
      pass
  _print_resolved_config(config, num_examples, participation, strategy_path, action)
  path, strategy, actual_action = get_or_fit_strategy(config, participation)
  if actual_action == "reuse":
    print(f"Reusing existing BandInvMF strategy: {path}")
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.logical_batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  # Preserve M6 as the setup authority even when a public artifact is reused.
  validate_nonamplified_bandinv_setup(
      strategy,
      calibration,
      ParticipationSpec(
          participation.horizon, participation.min_sep, participation.max_participations
      ),
      config.momentum,
      config.learning_rate,
  )
  train_config = Cifar10TrainConfig(
      strategy=str(path),
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
      adjacency=config.adjacency,
  )
  return train_cifar10(train_config)


__all__ = [
    "Cifar10NonAmplifiedExperimentConfig",
    "FixedCycleParticipation",
    "derive_fixed_cycle_participation",
    "get_or_fit_strategy",
    "load_cifar10_nonamplified_config",
    "run_cifar10_nonamplified",
    "strategy_artifact_path",
]
