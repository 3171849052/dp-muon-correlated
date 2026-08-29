"""YAML orchestration for frozen-p continuous BandInvMF DP-AdamW."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.bandinvmf import BandInvMFStrategy, fit_bandinv_strategy
from dp_muon.data import load_cifar10
from dp_muon.privacy import (
    calibrate_nonamplified_bandinv,
    continuous_hybrid_sensitivity_squared,
)

from .bandinvmf_strategy_manager import (
    FrozenPBandInvMFFitRequest,
    LoadedStrategySnapshot,
    frozen_p_strategy_artifact_path,
    get_or_fit_frozen_p_strategy_snapshot,
    require_compatible_frozen_p_strategy_snapshot,
)
from .cifar10_driver import (
    Cifar10FrozenPBandInvDPAdamWTrainConfig,
    FROZEN_P_BANDINV_DPADAMW_ALGORITHM,
    train_cifar10_frozen_p_bandinv_dpadamw,
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
class Cifar10FrozenPBandInvDPAdamWExperimentConfig:
  """YAML-facing configuration for the formal frozen-p algorithm."""

  algorithm: Literal["dp-adamw-correlated-frozen-p"]
  name: str
  pretrained: str
  data_dir: str
  epochs: int
  batch_size: int
  microbatch_size: int
  clip_norm: float
  epsilon: float
  delta: float
  learning_rate: float
  beta1: float
  beta2: float
  eps: float
  weight_decay: float
  seed: int
  checkpoint_dir: str
  eval_every: int
  adjacency: Literal["add_remove", "replace_one"]
  gpu: int
  schedule_mode: Literal["fixed_cycle"]
  bandwidth: int
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  force_refit: bool
  strategy_dir: str
  log_dir: str
  switch_step: int


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


def load_cifar10_frozen_p_bandinv_dpadamw_config(
    path: str | Path,
) -> Cifar10FrozenPBandInvDPAdamWExperimentConfig:
  """Loads and strictly validates the frozen-p YAML schema."""
  source = Path(path)
  try:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {source}") from error
  expected_sections = {
      "algorithm", "experiment", "runtime", "data", "model", "schedule",
      "strategy", "training", "adamw", "frozen_p", "privacy", "output",
  }
  if not isinstance(document, Mapping) or set(document) != expected_sections:
    raise ValueError(f"config sections must be exactly {sorted(expected_sections)}")
  experiment = _section(document, "experiment", {"name", "seed"})
  runtime = _section(document, "runtime", {"gpu"})
  data = _section(document, "data", {"data_dir"})
  model = _section(document, "model", {"pretrained"})
  schedule = _section(document, "schedule", {"mode"})
  strategy = _section(
      document, "strategy",
      {"bandwidth", "reduction", "max_optimizer_steps", "force_refit"},
  )
  training = _section(
      document, "training",
      {"epochs", "logical_batch_size", "microbatch_size", "clip_norm", "eval_every"},
  )
  adamw = _section(
      document, "adamw", {"learning_rate", "beta1", "beta2", "eps", "weight_decay"}
  )
  frozen_p = _section(document, "frozen_p", {"switch_step"})
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _section(document, "output", {"strategy_dir", "checkpoint_dir", "log_dir"})

  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != FROZEN_P_BANDINV_DPADAMW_ALGORITHM:
    raise ValueError(
        "frozen-p correlated DP-AdamW config requires algorithm: "
        f"{FROZEN_P_BANDINV_DPADAMW_ALGORITHM}"
    )
  mode = _string(schedule["mode"], "schedule.mode")
  if mode != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  adjacency = _string(privacy["adjacency"], "privacy.adjacency")
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("privacy.adjacency is invalid")
  reduction = _string(strategy["reduction"], "strategy.reduction")
  if reduction not in {"mean", "max", "last"}:
    raise ValueError("strategy.reduction must be one of: mean, max, last")
  if not isinstance(strategy["force_refit"], bool):
    raise ValueError("strategy.force_refit must be a boolean")
  beta1 = _number(adamw["beta1"], "adamw.beta1", nonnegative=True)
  beta2 = _number(adamw["beta2"], "adamw.beta2", nonnegative=True)
  if not 0.0 <= beta1 < 1.0:
    raise ValueError("adamw.beta1 must be in [0, 1)")
  if not 0.0 <= beta2 < 1.0:
    raise ValueError("adamw.beta2 must be in [0, 1)")
  config = Cifar10FrozenPBandInvDPAdamWExperimentConfig(
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
      learning_rate=_number(adamw["learning_rate"], "adamw.learning_rate", positive=True),
      beta1=beta1,
      beta2=beta2,
      eps=_number(adamw["eps"], "adamw.eps", positive=True),
      weight_decay=_number(adamw["weight_decay"], "adamw.weight_decay", nonnegative=True),
      seed=_integer(experiment["seed"], "experiment.seed", positive=False),
      checkpoint_dir=_string(output["checkpoint_dir"], "output.checkpoint_dir"),
      eval_every=_integer(training["eval_every"], "training.eval_every"),
      adjacency=adjacency,  # type: ignore[arg-type]
      gpu=_integer(runtime["gpu"], "runtime.gpu", positive=False),
      schedule_mode=mode,  # type: ignore[arg-type]
      bandwidth=_integer(strategy["bandwidth"], "strategy.bandwidth"),
      reduction=reduction,  # type: ignore[arg-type]
      max_optimizer_steps=_integer(
          strategy["max_optimizer_steps"], "strategy.max_optimizer_steps"
      ),
      force_refit=strategy["force_refit"],
      strategy_dir=_string(output["strategy_dir"], "output.strategy_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
      switch_step=_integer(frozen_p["switch_step"], "frozen_p.switch_step"),
  )
  if config.delta >= 1.0:
    raise ValueError("privacy.delta must be less than 1")
  if config.batch_size % config.microbatch_size:
    raise ValueError("training.logical_batch_size must be divisible by training.microbatch_size")
  return config


def _request(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
) -> FrozenPBandInvMFFitRequest:
  if not 1 <= config.switch_step < participation.horizon:
    raise ValueError("frozen_p.switch_step must lie in [1, derived horizon)")
  return FrozenPBandInvMFFitRequest(
      horizon=participation.horizon,
      switch_step=config.switch_step,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      bandwidth=config.bandwidth,
      beta1=config.beta1,
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
      strategy_dir=config.strategy_dir,
      force_refit=config.force_refit,
  )


def strategy_artifact_path(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
) -> Path:
  request = _request(config, participation)
  return frozen_p_strategy_artifact_path(
      request.strategy_dir,
      horizon=request.horizon,
      switch_step=request.switch_step,
      min_sep=request.min_sep,
      max_participations=request.max_participations,
      bandwidth=request.bandwidth,
      beta1=request.beta1,
      learning_rate=request.learning_rate,
      weight_decay=request.weight_decay,
      reduction=request.reduction,
      max_optimizer_steps=request.max_optimizer_steps,
  )


def get_or_fit_strategy_snapshot(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[LoadedStrategySnapshot, Literal["reuse", "fit"]]:
  return get_or_fit_frozen_p_strategy_snapshot(
      _request(config, participation), fit_strategy=fit_bandinv_strategy
  )


def require_compatible_strategy_snapshot(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
) -> LoadedStrategySnapshot:
  request = _request(config, participation)
  return require_compatible_frozen_p_strategy_snapshot(request)


def resolve_output_log_dir(config_path: str | Path) -> Path:
  directory = Path(load_cifar10_frozen_p_bandinv_dpadamw_config(config_path).log_dir)
  return directory if directory.is_absolute() else REPOSITORY_ROOT / directory


def _run_metadata(paths: Any) -> dict[str, str]:
  return {
      "directory": str(paths.directory.resolve()),
      "metrics": str(paths.metrics.resolve()),
      "checkpoint": str(paths.checkpoint.resolve()),
  }


def _create_run(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    document: Mapping[str, Any],
    source_yaml: str,
):
  root = Path(config.log_dir)
  if not root.is_absolute():
    root = REPOSITORY_ROOT / root
  paths = create_run_directory(
      root,
      epsilon=config.epsilon,
      bandwidth="bandinv-frozen-p",
      learning_rate=config.learning_rate,
      clip_norm=config.clip_norm,
      seed=config.seed,
      config_hash=config_content_hash(document),
  )
  write_run_configuration(
      paths,
      source_yaml=source_yaml,
      resolved={"experiment": asdict(config), "run": _run_metadata(paths)},
  )
  MetricsCSVWriter(paths.metrics)
  return paths


def prepare_cifar10_frozen_p_bandinv_dpadamw_run(config_path: str | Path):
  config = load_cifar10_frozen_p_bandinv_dpadamw_config(config_path)
  source_yaml = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  return _create_run(config, document, source_yaml)


def _train_config(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    strategy_path: str | Path,
    participation: FixedCycleParticipation,
) -> Cifar10FrozenPBandInvDPAdamWTrainConfig:
  return Cifar10FrozenPBandInvDPAdamWTrainConfig(
      strategy=str(strategy_path),
      pretrained=config.pretrained,
      data_dir=config.data_dir,
      batch_size=config.batch_size,
      microbatch_size=config.microbatch_size,
      clip_norm=config.clip_norm,
      epsilon=config.epsilon,
      delta=config.delta,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      beta2=config.beta2,
      eps=config.eps,
      weight_decay=config.weight_decay,
      seed=config.seed,
      checkpoint_dir=config.checkpoint_dir,
      eval_every=config.eval_every,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      switch_step=config.switch_step,
      adjacency=config.adjacency,
  )


def _calibration(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
    strategy: BandInvMFStrategy,
):
  sensitivity = continuous_hybrid_sensitivity_squared(
      config.switch_step,
      strategy,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
  )
  return calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=sensitivity,
  )


def _resolved_config(
    config: Cifar10FrozenPBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
    snapshot: LoadedStrategySnapshot,
    action: Literal["reuse", "fit"],
    calibration: Any,
) -> dict[str, Any]:
  strategy = snapshot.strategy
  return {
      "experiment": asdict(config),
      "participation": asdict(participation),
      "frozen_p": {
          "switch_step": config.switch_step,
          "phase_horizon": strategy.horizon,
          "optimizer": "state-preserving Optax AdamW -> FrozenPAdamW",
          "hybrid_sensitivity_squared": float(calibration.matrix_sensitivity ** 2),
      },
      "strategy": {
          "artifact": str(snapshot.path.resolve()),
          "sha256": snapshot.sha256,
          "action": action,
          "workload_type": "frozen-p-continuous",
          "horizon": strategy.horizon,
          "bandwidth": strategy.bandwidth,
          "min_sep": strategy.min_sep,
          "max_participations": strategy.max_participations,
          "phase_strategy_sensitivity_squared": float(strategy.sensitivity_squared),
      },
      "privacy_calibration": {
          **asdict(calibration),
          "scope": "one full hybrid transcript: blockdiag(I_tau, D_phase)",
          "warmup_and_phase_have_separate_epsilon": False,
      },
  }


def run_cifar10_frozen_p_bandinv_dpadamw(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
):
  config = load_cifar10_frozen_p_bandinv_dpadamw_config(config_path)
  source_yaml = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  if resume_checkpoint is not None and run_dir is not None:
    raise ValueError("resume_checkpoint and run_dir are mutually exclusive")
  paths = (
      existing_run_paths(resume_checkpoint)
      if resume_checkpoint is not None
      else run_paths_from_directory(run_dir)
      if run_dir is not None
      else _create_run(config, document, source_yaml)
  )
  train_images, _ = load_cifar10(config.data_dir, train=True)
  participation = derive_fixed_cycle_participation(
      len(train_images), config.epochs, config.batch_size
  )
  del train_images
  if resume_checkpoint is not None:
    snapshot = require_compatible_strategy_snapshot(config, participation)
    action: Literal["reuse", "fit"] = "reuse"
  else:
    snapshot, action = get_or_fit_strategy_snapshot(config, participation)
  calibration = _calibration(config, participation, snapshot.strategy)
  if resume_checkpoint is None:
    resolved = _resolved_config(config, participation, snapshot, action, calibration)
    resolved["run"] = _run_metadata(paths)
    write_run_configuration(paths, source_yaml=source_yaml, resolved=resolved)
    MetricsCSVWriter(paths.metrics)
  return train_cifar10_frozen_p_bandinv_dpadamw(
      _train_config(config, snapshot.path, participation),
      strategy_snapshot=snapshot,
      resume_checkpoint=resume_checkpoint,
      checkpoint_path=paths.checkpoint,
      metrics_path=paths.metrics,
  )


__all__ = [
    "Cifar10FrozenPBandInvDPAdamWExperimentConfig",
    "get_or_fit_strategy_snapshot",
    "load_cifar10_frozen_p_bandinv_dpadamw_config",
    "prepare_cifar10_frozen_p_bandinv_dpadamw_run",
    "require_compatible_strategy_snapshot",
    "resolve_output_log_dir",
    "run_cifar10_frozen_p_bandinv_dpadamw",
    "strategy_artifact_path",
]
