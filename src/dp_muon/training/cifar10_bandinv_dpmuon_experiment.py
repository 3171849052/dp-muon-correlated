"""YAML orchestration for naive BandInvMF-correlated DP-Muon."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.bandinvmf import BandInvMFStrategy, fit_bandinv_strategy
from dp_muon.data import load_cifar10
from dp_muon.privacy import calibrate_nonamplified_bandinv

from .bandinvmf_strategy_manager import (
    BandInvMFFitRequest,
    LoadedStrategySnapshot,
    get_or_fit_strategy as _get_or_fit_shared_strategy,
    get_or_fit_strategy_snapshot as _get_or_fit_shared_snapshot,
    require_compatible_strategy as _require_compatible_shared_strategy,
    require_compatible_strategy_snapshot as _require_compatible_shared_snapshot,
    strategy_artifact_path as _shared_strategy_artifact_path,
)
from .cifar10_driver import (
    BANDINV_DPMUON_ALGORITHM,
    Cifar10BandInvDPMuonTrainConfig,
    train_cifar10_bandinv_dpmuon,
)
from .cifar10_experiment import FixedCycleParticipation, derive_fixed_cycle_participation
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
class Cifar10BandInvDPMuonExperimentConfig:
  """YAML-facing fields for ``Cifar10BandInvDPMuonTrainConfig`` plus runtime."""

  algorithm: Literal["dp-muon-correlated-naive"]
  name: str
  pretrained: str
  data_dir: str
  epochs: int
  batch_size: int
  microbatch_size: int
  clip_norm: float
  epsilon: float
  delta: float
  muon_learning_rate: float
  muon_weight_decay: float
  momentum: float
  ns_steps: int
  consistent_rms: float
  adamw_learning_rate: float
  adamw_beta1: float
  adamw_beta2: float
  adamw_eps: float
  adamw_weight_decay: float
  seed: int
  checkpoint_dir: str
  eval_every: int
  adjacency: Literal["add_remove", "replace_one"]
  use_bf16_ns: bool
  gpu: int
  schedule_mode: Literal["fixed_cycle"]
  bandwidth: int
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  force_refit: bool
  strategy_dir: str
  log_dir: str


def _section(document: Mapping[str, Any], name: str, keys: set[str]) -> Mapping[str, Any]:
  value = document.get(name)
  if not isinstance(value, Mapping) or set(value) != keys:
    raise ValueError(f"config.{name} must contain exactly {sorted(keys)}")
  return value


def _string(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{name} must be a non-empty string")
  return value


def _integer(value: object, name: str, *, positive: bool = True) -> int:
  if (
      isinstance(value, bool)
      or not isinstance(value, Integral)
      or (positive and value < 1)
      or (not positive and value < 0)
  ):
    qualifier = "positive" if positive else "non-negative"
    raise ValueError(f"{name} must be a {qualifier} integer")
  return int(value)


def _number(
    value: object, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
  if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
    raise ValueError(f"{name} must be finite")
  result = float(value)
  if (positive and result <= 0) or (nonnegative and result < 0):
    raise ValueError(f"{name} has an invalid value")
  return result


def load_cifar10_bandinv_dpmuon_config(
    path: str | Path,
) -> Cifar10BandInvDPMuonExperimentConfig:
  """Loads the fixed-schema naive correlated DP-Muon YAML configuration."""
  source = Path(path)
  try:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {source}") from error
  expected_sections = {
      "algorithm", "experiment", "runtime", "data", "model", "schedule", "strategy",
      "training", "muon", "adamw", "privacy", "output",
  }
  if not isinstance(document, Mapping) or set(document) != expected_sections:
    raise ValueError(f"config sections must be exactly {sorted(expected_sections)}")
  experiment = _section(document, "experiment", {"name", "seed"})
  runtime = _section(document, "runtime", {"gpu"})
  data = _section(document, "data", {"data_dir"})
  model = _section(document, "model", {"pretrained"})
  training = _section(
      document, "training",
      {"epochs", "logical_batch_size", "microbatch_size", "clip_norm", "eval_every"},
  )
  schedule = _section(document, "schedule", {"mode"})
  strategy = _section(
      document, "strategy",
      {"bandwidth", "reduction", "max_optimizer_steps", "force_refit"},
  )
  muon = _section(
      document, "muon",
      {"learning_rate", "weight_decay", "momentum", "ns_steps", "consistent_rms", "use_bf16_ns"},
  )
  adamw = _section(document, "adamw", {"learning_rate", "beta1", "beta2", "eps", "weight_decay"})
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _section(document, "output", {"strategy_dir", "checkpoint_dir", "log_dir"})
  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != BANDINV_DPMUON_ALGORITHM:
    raise ValueError(
        f"naive correlated DP-Muon config requires algorithm: {BANDINV_DPMUON_ALGORITHM}"
    )
  adjacency = _string(privacy["adjacency"], "privacy.adjacency")
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("privacy.adjacency is invalid")
  mode = _string(schedule["mode"], "schedule.mode")
  if mode != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  reduction = _string(strategy["reduction"], "strategy.reduction")
  if reduction not in {"mean", "max", "last"}:
    raise ValueError("strategy.reduction must be one of: mean, max, last")
  if not isinstance(muon["use_bf16_ns"], bool):
    raise ValueError("muon.use_bf16_ns must be boolean")
  if not isinstance(strategy["force_refit"], bool):
    raise ValueError("strategy.force_refit must be a boolean")
  config = Cifar10BandInvDPMuonExperimentConfig(
      algorithm=algorithm,  # type: ignore[arg-type]
      name=_string(experiment["name"], "experiment.name"),
      pretrained=_string(model["pretrained"], "model.pretrained"),
      data_dir=_string(data["data_dir"], "data.data_dir"),
      epochs=_integer(training["epochs"], "training.epochs"),
      batch_size=_integer(training["logical_batch_size"], "training.logical_batch_size"),
      microbatch_size=_integer(training["microbatch_size"], "training.microbatch_size"),
      clip_norm=_number(training["clip_norm"], "training.clip_norm", positive=True),
      epsilon=_number(privacy["epsilon"], "privacy.epsilon", positive=True),
      delta=_number(privacy["delta"], "privacy.delta", positive=True),
      muon_learning_rate=_number(muon["learning_rate"], "muon.learning_rate", positive=True),
      muon_weight_decay=_number(muon["weight_decay"], "muon.weight_decay", nonnegative=True),
      momentum=_number(muon["momentum"], "muon.momentum", nonnegative=True),
      ns_steps=_integer(muon["ns_steps"], "muon.ns_steps"),
      consistent_rms=_number(muon["consistent_rms"], "muon.consistent_rms", positive=True),
      adamw_learning_rate=_number(adamw["learning_rate"], "adamw.learning_rate", positive=True),
      adamw_beta1=_number(adamw["beta1"], "adamw.beta1", nonnegative=True),
      adamw_beta2=_number(adamw["beta2"], "adamw.beta2", nonnegative=True),
      adamw_eps=_number(adamw["eps"], "adamw.eps", positive=True),
      adamw_weight_decay=_number(adamw["weight_decay"], "adamw.weight_decay", nonnegative=True),
      seed=_integer(experiment["seed"], "experiment.seed", positive=False),
      checkpoint_dir=_string(output["checkpoint_dir"], "output.checkpoint_dir"),
      eval_every=_integer(training["eval_every"], "training.eval_every"),
      adjacency=adjacency,  # type: ignore[arg-type]
      use_bf16_ns=muon["use_bf16_ns"],
      gpu=_integer(runtime["gpu"], "runtime.gpu", positive=False),
      schedule_mode=mode,  # type: ignore[arg-type]
      bandwidth=_integer(strategy["bandwidth"], "strategy.bandwidth"),
      reduction=reduction,  # type: ignore[arg-type]
      max_optimizer_steps=_integer(strategy["max_optimizer_steps"], "strategy.max_optimizer_steps"),
      force_refit=strategy["force_refit"],
      strategy_dir=_string(output["strategy_dir"], "output.strategy_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
  )
  if (
      config.delta >= 1
      or config.momentum >= 1
      or config.adamw_beta1 >= 1
      or config.adamw_beta2 >= 1
      or config.batch_size % config.microbatch_size
  ):
    raise ValueError("invalid batch division, privacy, or momentum setting")
  return config


def strategy_artifact_path(
    config: Cifar10BandInvDPMuonExperimentConfig,
    participation: FixedCycleParticipation,
) -> Path:
  """Returns the repository-root-resolved deterministic strategy path."""
  return _shared_strategy_artifact_path(
      config.strategy_dir,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      bandwidth=config.bandwidth,
      momentum=config.momentum,
      learning_rate=config.muon_learning_rate,
      weight_decay=config.muon_weight_decay,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )


def get_or_fit_strategy(
    config: Cifar10BandInvDPMuonExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[Path, BandInvMFStrategy, Literal["reuse", "fit"]]:
  """Fits/reuses the correlated DP-Muon strategy using its Muon trajectory."""
  return _get_or_fit_shared_strategy(
      BandInvMFFitRequest(
          horizon=participation.horizon,
          min_sep=participation.min_sep,
          max_participations=participation.max_participations,
          bandwidth=config.bandwidth,
          momentum=config.momentum,
          learning_rate=config.muon_learning_rate,
          weight_decay=config.muon_weight_decay,
          reduction=config.reduction,
          max_optimizer_steps=config.max_optimizer_steps,
          strategy_dir=config.strategy_dir,
          force_refit=config.force_refit,
      ),
      fit_strategy=fit_bandinv_strategy,
  )


def require_compatible_strategy(
    config: Cifar10BandInvDPMuonExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[Path, BandInvMFStrategy]:
  """Loads the existing resume artifact and never replaces it."""
  return _require_compatible_shared_strategy(
      BandInvMFFitRequest(
          horizon=participation.horizon, min_sep=participation.min_sep,
          max_participations=participation.max_participations,
          bandwidth=config.bandwidth, momentum=config.momentum,
          learning_rate=config.muon_learning_rate, reduction=config.reduction,
          weight_decay=config.muon_weight_decay,
          max_optimizer_steps=config.max_optimizer_steps,
          strategy_dir=config.strategy_dir, force_refit=False,
      )
  )


def get_or_fit_strategy_snapshot(
    config: Cifar10BandInvDPMuonExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[LoadedStrategySnapshot, Literal["reuse", "fit"]]:
  return _get_or_fit_shared_snapshot(
      BandInvMFFitRequest(
          horizon=participation.horizon, min_sep=participation.min_sep,
          max_participations=participation.max_participations,
          bandwidth=config.bandwidth, momentum=config.momentum,
          learning_rate=config.muon_learning_rate, reduction=config.reduction,
          weight_decay=config.muon_weight_decay,
          max_optimizer_steps=config.max_optimizer_steps,
          strategy_dir=config.strategy_dir, force_refit=config.force_refit,
      ), fit_strategy=fit_bandinv_strategy,
  )


def require_compatible_strategy_snapshot(
    config: Cifar10BandInvDPMuonExperimentConfig,
    participation: FixedCycleParticipation,
) -> LoadedStrategySnapshot:
  return _require_compatible_shared_snapshot(
      BandInvMFFitRequest(
          horizon=participation.horizon, min_sep=participation.min_sep,
          max_participations=participation.max_participations,
          bandwidth=config.bandwidth, momentum=config.momentum,
          learning_rate=config.muon_learning_rate, reduction=config.reduction,
          weight_decay=config.muon_weight_decay,
          max_optimizer_steps=config.max_optimizer_steps,
          strategy_dir=config.strategy_dir, force_refit=False,
      )
  )


def resolve_output_log_dir(config_path: str | Path) -> Path:
  """Resolves the YAML log root relative to the repository."""
  directory = Path(load_cifar10_bandinv_dpmuon_config(config_path).log_dir)
  return directory if directory.is_absolute() else REPOSITORY_ROOT / directory


def _run_metadata(paths: Any) -> dict[str, str]:
  return {
      "directory": str(paths.directory.resolve()),
      "metrics": str(paths.metrics.resolve()),
      "checkpoint": str(paths.checkpoint.resolve()),
  }


def _create_run(
    config: Cifar10BandInvDPMuonExperimentConfig,
    document: Mapping[str, Any],
    source_yaml: str,
):
  root = Path(config.log_dir)
  if not root.is_absolute():
    root = REPOSITORY_ROOT / root
  paths = create_run_directory(
      root,
      epsilon=config.epsilon,
      bandwidth="bandinv-naive",
      learning_rate=config.muon_learning_rate,
      clip_norm=config.clip_norm,
      seed=config.seed,
      config_hash=config_content_hash(document),
  )
  write_run_configuration(
      paths,
      source_yaml=source_yaml,
      resolved={
          "experiment": asdict(config),
          "run": _run_metadata(paths),
      },
  )
  MetricsCSVWriter(paths.metrics)
  return paths


def prepare_cifar10_bandinv_dpmuon_run(config_path: str | Path):
  """Creates the run directory and snapshots public config without training."""
  config = load_cifar10_bandinv_dpmuon_config(config_path)
  source_yaml = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  return _create_run(config, document, source_yaml)


def _train_config(
    config: Cifar10BandInvDPMuonExperimentConfig,
    strategy_path: str | Path,
) -> Cifar10BandInvDPMuonTrainConfig:
  return Cifar10BandInvDPMuonTrainConfig(
      strategy=str(strategy_path),
      pretrained=config.pretrained,
      data_dir=config.data_dir,
      batch_size=config.batch_size,
      microbatch_size=config.microbatch_size,
      clip_norm=config.clip_norm,
      epsilon=config.epsilon,
      delta=config.delta,
      muon_learning_rate=config.muon_learning_rate,
      muon_weight_decay=config.muon_weight_decay,
      momentum=config.momentum,
      ns_steps=config.ns_steps,
      consistent_rms=config.consistent_rms,
      adamw_learning_rate=config.adamw_learning_rate,
      adamw_beta1=config.adamw_beta1,
      adamw_beta2=config.adamw_beta2,
      adamw_eps=config.adamw_eps,
      adamw_weight_decay=config.adamw_weight_decay,
      seed=config.seed,
      checkpoint_dir=config.checkpoint_dir,
      eval_every=config.eval_every,
      adjacency=config.adjacency,
      use_bf16_ns=config.use_bf16_ns,
  )


def _resolved_config(
    config: Cifar10BandInvDPMuonExperimentConfig,
    participation: FixedCycleParticipation,
    strategy_path: Path,
    strategy: BandInvMFStrategy,
    strategy_sha256: str,
    action: Literal["reuse", "fit"],
    calibration: Any,
) -> dict[str, Any]:
  return {
      "experiment": asdict(config),
      "participation": asdict(participation),
      "strategy": {
          "artifact": str(strategy_path.resolve()),
          "sha256": strategy_sha256,
          "action": action,
          "workload_type": "nesterov-decayed-trajectory",
          "horizon": strategy.horizon,
          "bandwidth": strategy.bandwidth,
          "min_sep": strategy.min_sep,
          "max_participations": strategy.max_participations,
          "momentum": config.momentum,
          "learning_rate": config.muon_learning_rate,
          "weight_decay": config.muon_weight_decay,
          "reduction": config.reduction,
          "max_optimizer_steps": config.max_optimizer_steps,
          "sensitivity_squared": float(strategy.sensitivity_squared),
      },
      "privacy_calibration": asdict(calibration),
  }


def run_cifar10_bandinv_dpmuon(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
):
  """Delegates all training and privacy work to the existing naive trainer."""
  config = load_cifar10_bandinv_dpmuon_config(config_path)
  source_yaml = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  if resume_checkpoint is not None and run_dir is not None:
    raise ValueError("resume_checkpoint and run_dir are mutually exclusive")
  paths = (
      existing_run_paths(resume_checkpoint) if resume_checkpoint is not None
      else run_paths_from_directory(run_dir) if run_dir is not None
      else _create_run(config, document, source_yaml)
  )
  train_images, _ = load_cifar10(config.data_dir, train=True)
  participation = derive_fixed_cycle_participation(
      len(train_images), config.epochs, config.batch_size
  )
  del train_images
  if resume_checkpoint is not None:
    snapshot = require_compatible_strategy_snapshot(config, participation)
    action = "reuse"
  else:
    snapshot, action = get_or_fit_strategy_snapshot(config, participation)
  strategy_path, strategy = snapshot.path, snapshot.strategy
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  if resume_checkpoint is None:
    resolved = _resolved_config(
        config, participation, strategy_path, strategy, snapshot.sha256, action, calibration
    )
    resolved["run"] = _run_metadata(paths)
    write_run_configuration(paths, source_yaml=source_yaml, resolved=resolved)
    MetricsCSVWriter(paths.metrics)
  return train_cifar10_bandinv_dpmuon(
      _train_config(config, strategy_path),
      strategy_snapshot=snapshot,
      resume_checkpoint=resume_checkpoint,
      checkpoint_path=paths.checkpoint,
      metrics_path=paths.metrics,
  )


__all__ = [
    "Cifar10BandInvDPMuonExperimentConfig",
    "get_or_fit_strategy",
    "load_cifar10_bandinv_dpmuon_config",
    "prepare_cifar10_bandinv_dpmuon_run",
    "require_compatible_strategy",
    "resolve_output_log_dir",
    "run_cifar10_bandinv_dpmuon",
    "strategy_artifact_path",
]
