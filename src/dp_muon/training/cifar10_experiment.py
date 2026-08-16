"""YAML-driven CIFAR-10 non-amplified BandInvMF experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.bandinvmf import (
    BandInvMFStrategy,
    fit_bandinv_strategy,
)
from dp_muon.data import load_cifar10
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv

from .bandinvmf_strategy_manager import (
    BandInvMFFitRequest,
    get_or_fit_strategy as _get_or_fit_shared_strategy,
    require_compatible_strategy as _require_compatible_shared_strategy,
    strategy_artifact_path as _shared_strategy_artifact_path,
)
from .cifar10_driver import Cifar10TrainConfig, train_cifar10
from .nonamplified_linear import validate_nonamplified_bandinv_setup
from .run_logging import (
    MetricsCSVWriter,
    config_content_hash,
    create_run_directory,
    existing_run_paths,
    run_paths_from_directory,
    write_run_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FixedCycleParticipation:
  """Public fixed-cycle values derived from epoch and logical batch settings."""

  horizon: int
  max_participations: int
  min_sep: int
  effective_epochs: float


@dataclass(frozen=True)
class Cifar10NonAmplifiedExperimentConfig:
  algorithm: Literal["bandinv"]
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
  adjacency: Literal["add_remove"]
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
      "algorithm", "experiment", "runtime", "data", "model", "training", "schedule", "privacy", "strategy", "output"
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
  strategy = _section(
      document,
      "strategy",
      {"bandwidth", "reduction", "max_optimizer_steps", "force_refit"},
  )
  output = _section(document, "output", {"strategy_dir", "checkpoint_dir", "log_dir"})
  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != "bandinv":
    raise ValueError("BandInvMF config requires algorithm: bandinv")
  mode = _string(schedule["mode"], "schedule.mode")
  if mode != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  adjacency = _string(privacy["adjacency"], "privacy.adjacency")
  if adjacency != "add_remove":
    raise ValueError(
        "CIFAR-10 YAML runner currently supports only adjacency='add_remove'"
    )
  reduction = _string(strategy["reduction"], "strategy.reduction")
  if reduction not in {"mean", "max", "last"}:
    raise ValueError("strategy.reduction must be one of: mean, max, last")
  momentum = _finite_float(training["momentum"], "training.momentum")
  if not 0.0 <= momentum < 1.0:
    raise ValueError("training.momentum must be in [0, 1)")
  config = Cifar10NonAmplifiedExperimentConfig(
      algorithm=algorithm,  # type: ignore[arg-type]
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


def resolve_output_log_dir(config_path: str | Path) -> Path:
  """Returns the YAML log directory, resolving relative paths from repo root."""
  log_dir = Path(load_cifar10_nonamplified_config(config_path).log_dir)
  return log_dir if log_dir.is_absolute() else REPOSITORY_ROOT / log_dir


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
  return _shared_strategy_artifact_path(
      strategy_dir,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      bandwidth=bandwidth,
      momentum=momentum,
      learning_rate=learning_rate,
      reduction=reduction,
      max_optimizer_steps=max_optimizer_steps,
  )


def get_or_fit_strategy(
    config: Cifar10NonAmplifiedExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[Path, BandInvMFStrategy, Literal["reuse", "fit"]]:
  """Returns an exactly compatible artifact or refits the deterministic path.

  This wrapper preserves the original experiment-facing API while delegating
  cache semantics to the shared manager used by correlated DP-Muon as well.
  """
  return _get_or_fit_shared_strategy(
      BandInvMFFitRequest(
          horizon=participation.horizon,
          min_sep=participation.min_sep,
          max_participations=participation.max_participations,
          bandwidth=config.bandwidth,
          momentum=config.momentum,
          learning_rate=config.learning_rate,
          reduction=config.reduction,
          max_optimizer_steps=config.max_optimizer_steps,
          strategy_dir=config.strategy_dir,
          force_refit=config.force_refit,
      ),
      fit_strategy=fit_bandinv_strategy,
  )


def require_compatible_strategy(
    config: Cifar10NonAmplifiedExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[Path, BandInvMFStrategy]:
  """Loads the prior public artifact for resume without ever refitting it."""
  return _require_compatible_shared_strategy(
      BandInvMFFitRequest(
          horizon=participation.horizon, min_sep=participation.min_sep,
          max_participations=participation.max_participations,
          bandwidth=config.bandwidth, momentum=config.momentum,
          learning_rate=config.learning_rate, reduction=config.reduction,
          max_optimizer_steps=config.max_optimizer_steps,
          strategy_dir=config.strategy_dir, force_refit=False,
      )
  )


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
          "BandInvMF:",
          f"  bandwidth p: {config.bandwidth}",
          f"  momentum: {config.momentum}",
          f"  learning_rate: {config.learning_rate}",
          f"  strategy path: {strategy_path}",
          f"  strategy action: {action}",
      ))
  )


def _source_config_document(config_path: str | Path) -> tuple[str, Mapping[str, Any]]:
  source = Path(config_path)
  source_yaml = source.read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  return source_yaml, document


def _resolved_config(
    config: Cifar10NonAmplifiedExperimentConfig,
    participation: FixedCycleParticipation,
    strategy_path: Path,
    strategy: BandInvMFStrategy,
    action: str,
    calibration: Any,
) -> dict[str, Any]:
  return {
      "experiment": asdict(config),
      "participation": asdict(participation),
      "strategy": {
          "artifact": str(strategy_path.resolve()),
          "action": action,
          "horizon": strategy.horizon,
          "bandwidth": strategy.bandwidth,
          "min_sep": strategy.min_sep,
          "max_participations": strategy.max_participations,
          "sensitivity_squared": float(strategy.sensitivity_squared),
      },
      "privacy_calibration": asdict(calibration),
  }


def _run_metadata(run_paths: Any) -> dict[str, str]:
  return {
      "directory": str(run_paths.directory.resolve()),
      "metrics": str(run_paths.metrics.resolve()),
      "checkpoint": str(run_paths.checkpoint.resolve()),
  }


def _create_run(
    config: Cifar10NonAmplifiedExperimentConfig,
    source_yaml: str,
    document: Mapping[str, Any],
):
  log_root = Path(config.log_dir)
  if not log_root.is_absolute():
    log_root = REPOSITORY_ROOT / log_root
  run_paths = create_run_directory(
      log_root,
      epsilon=config.epsilon,
      bandwidth=config.bandwidth,
      learning_rate=config.learning_rate,
      clip_norm=config.clip_norm,
      seed=config.seed,
      config_hash=config_content_hash(document),
  )
  # This provisional record lets the shell safely redirect every subsequent
  # training message before public strategy/data setup is complete.
  write_run_configuration(
      run_paths,
      source_yaml=source_yaml,
      resolved={"experiment": asdict(config), "run": _run_metadata(run_paths)},
  )
  MetricsCSVWriter(run_paths.metrics)
  return run_paths


def prepare_cifar10_nonamplified_run(config_path: str | Path):
  """Creates and snapshots a run directory without starting training."""
  config = load_cifar10_nonamplified_config(config_path)
  source_yaml, document = _source_config_document(config_path)
  return _create_run(config, source_yaml, document)


def run_cifar10_nonamplified(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
):
  """Fits/reuses the public strategy, validates M6 setup, then trains CIFAR-10."""
  config = load_cifar10_nonamplified_config(config_path)
  source_yaml, document = _source_config_document(config_path)
  if resume_checkpoint is not None and run_dir is not None:
    raise ValueError("resume_checkpoint and run_dir are mutually exclusive")
  if resume_checkpoint is not None:
    run_paths = existing_run_paths(resume_checkpoint)
  elif run_dir is not None:
    run_paths = run_paths_from_directory(run_dir)
  else:
    run_paths = _create_run(config, source_yaml, document)
  train_images, _ = load_cifar10(config.data_dir, train=True)
  num_examples = len(train_images)
  del train_images
  participation = derive_fixed_cycle_participation(
      num_examples, config.epochs, config.logical_batch_size
  )
  if resume_checkpoint is not None:
    path, strategy = require_compatible_strategy(config, participation)
    actual_action = "reuse"
  else:
    path, strategy, actual_action = get_or_fit_strategy(config, participation)
  _print_resolved_config(config, num_examples, participation, path, actual_action)
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
  if resume_checkpoint is None:
    resolved = _resolved_config(
        config, participation, path, strategy, actual_action, calibration
    )
    resolved["run"] = _run_metadata(run_paths)
    write_run_configuration(run_paths, source_yaml=source_yaml, resolved=resolved)
    MetricsCSVWriter(run_paths.metrics)
  return train_cifar10(
      train_config,
      resume_checkpoint=resume_checkpoint,
      checkpoint_path=run_paths.checkpoint,
      metrics_path=run_paths.metrics,
  )


__all__ = [
    "Cifar10NonAmplifiedExperimentConfig",
    "FixedCycleParticipation",
    "derive_fixed_cycle_participation",
    "get_or_fit_strategy",
    "load_cifar10_nonamplified_config",
    "prepare_cifar10_nonamplified_run",
    "require_compatible_strategy",
    "resolve_output_log_dir",
    "run_cifar10_nonamplified",
    "strategy_artifact_path",
]
