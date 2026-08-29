"""YAML orchestration for correlated warmup plus shadow-JME DP-AdamW."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.data import load_cifar10

from .cifar10_driver import (
    Cifar10ShadowJMEBandInvDPAdamWTrainConfig,
    SHADOW_JME_BANDINV_DPADAMW_ALGORITHM,
    train_cifar10_shadow_jme_bandinv_dpadamw,
)
from .cifar10_experiment import FixedCycleParticipation, derive_fixed_cycle_participation
from .nonamplified_shadow_jme_bandinv_dpadamw import ShadowJMEPlan, fit_shadow_jme_plan
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
class Cifar10ShadowJMEBandInvDPAdamWExperimentConfig:
  algorithm: Literal["dp-adamw-correlated-shadow-jme"]
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
  warmup_steps: int
  segment_length: int
  bandwidth: int
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  v_floor: float
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


def _number(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
  if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
    raise ValueError(f"{name} must be finite")
  result = float(value)
  if (positive and result <= 0) or (nonnegative and result < 0):
    raise ValueError(f"{name} has an invalid value")
  return result


def load_cifar10_shadow_jme_bandinv_dpadamw_config(
    path: str | Path,
) -> Cifar10ShadowJMEBandInvDPAdamWExperimentConfig:
  source = Path(path)
  try:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {source}") from error
  expected = {
      "algorithm", "experiment", "runtime", "data", "model", "training",
      "schedule", "strategy", "adamw", "privacy", "output",
  }
  if not isinstance(document, Mapping) or set(document) != expected:
    raise ValueError(f"config sections must be exactly {sorted(expected)}")
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
      {"warmup_steps", "segment_length", "bandwidth", "reduction", "max_optimizer_steps", "v_floor", "force_refit"},
  )
  adamw = _section(document, "adamw", {"learning_rate", "beta1", "beta2", "eps", "weight_decay"})
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _section(document, "output", {"strategy_dir", "checkpoint_dir", "log_dir"})
  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != SHADOW_JME_BANDINV_DPADAMW_ALGORITHM:
    raise ValueError(
        "shadow-JME config requires algorithm: "
        f"{SHADOW_JME_BANDINV_DPADAMW_ALGORITHM}"
    )
  if _string(schedule["mode"], "schedule.mode") != "fixed_cycle":
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
  if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
    raise ValueError("adamw.beta1 and adamw.beta2 must be in [0, 1)")
  config = Cifar10ShadowJMEBandInvDPAdamWExperimentConfig(
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
      schedule_mode="fixed_cycle",
      warmup_steps=_integer(strategy["warmup_steps"], "strategy.warmup_steps"),
      segment_length=_integer(strategy["segment_length"], "strategy.segment_length"),
      bandwidth=_integer(strategy["bandwidth"], "strategy.bandwidth"),
      reduction=reduction,  # type: ignore[arg-type]
      max_optimizer_steps=_integer(strategy["max_optimizer_steps"], "strategy.max_optimizer_steps"),
      v_floor=_number(strategy["v_floor"], "strategy.v_floor", nonnegative=True),
      force_refit=strategy["force_refit"],
      strategy_dir=_string(output["strategy_dir"], "output.strategy_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
  )
  if config.delta >= 1.0:
    raise ValueError("privacy.delta must be less than 1")
  if config.batch_size % config.microbatch_size:
    raise ValueError("training.logical_batch_size must be divisible by training.microbatch_size")
  return config


def get_or_fit_shadow_jme_plan(
    config: Cifar10ShadowJMEBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
) -> ShadowJMEPlan:
  """Fits the initial unit-``P`` plan; later pairs are refit at boundaries."""
  return fit_shadow_jme_plan(
      horizon=participation.horizon,
      warmup_steps=config.warmup_steps,
      segment_length=config.segment_length,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      bandwidth=config.bandwidth,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      beta2=config.beta2,
      eps=config.eps,
      weight_decay=config.weight_decay,
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      v_floor=config.v_floor,
  )


def resolve_output_log_dir(config_path: str | Path) -> Path:
  directory = Path(load_cifar10_shadow_jme_bandinv_dpadamw_config(config_path).log_dir)
  return directory if directory.is_absolute() else REPOSITORY_ROOT / directory


def _run_metadata(paths: Any) -> dict[str, str]:
  return {
      "directory": str(paths.directory.resolve()),
      "metrics": str(paths.metrics.resolve()),
      "checkpoint": str(paths.checkpoint.resolve()),
  }


def _create_run(config: Cifar10ShadowJMEBandInvDPAdamWExperimentConfig, document: Mapping[str, Any], source_yaml: str):
  root = Path(config.log_dir)
  if not root.is_absolute():
    root = REPOSITORY_ROOT / root
  paths = create_run_directory(
      root,
      epsilon=config.epsilon,
      bandwidth=f"bandinv-shadow-jme-seg{config.segment_length}",
      learning_rate=config.learning_rate,
      clip_norm=config.clip_norm,
      seed=config.seed,
      config_hash=config_content_hash(document),
  )
  write_run_configuration(paths, source_yaml=source_yaml, resolved={"experiment": asdict(config), "run": _run_metadata(paths)})
  MetricsCSVWriter(paths.metrics)
  return paths


def prepare_cifar10_shadow_jme_bandinv_dpadamw_run(config_path: str | Path):
  config = load_cifar10_shadow_jme_bandinv_dpadamw_config(config_path)
  source_yaml = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  return _create_run(config, document, source_yaml)


def _train_config(config: Cifar10ShadowJMEBandInvDPAdamWExperimentConfig) -> Cifar10ShadowJMEBandInvDPAdamWTrainConfig:
  return Cifar10ShadowJMEBandInvDPAdamWTrainConfig(
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
      warmup_steps=config.warmup_steps,
      segment_length=config.segment_length,
      bandwidth=config.bandwidth,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
      v_floor=config.v_floor,
      seed=config.seed,
      checkpoint_dir=config.checkpoint_dir,
      eval_every=config.eval_every,
      adjacency=config.adjacency,
  )


def _resolved_config(config: Cifar10ShadowJMEBandInvDPAdamWExperimentConfig, participation: FixedCycleParticipation, plan: ShadowJMEPlan) -> dict[str, Any]:
  return {
      "experiment": asdict(config),
      "participation": asdict(participation),
      "shadow_jme": {
          "condition": plan.condition,
          "warmup_steps": plan.warmup_steps,
          "segment_lengths": list(plan.segment_lengths),
          "global_sensitivity_squared": plan.calibration.total_sensitivity_squared,
          "warmup_sensitivity_squared": plan.calibration.warmup_sensitivity_squared,
          "segment_sensitivity_squared": list(plan.calibration.segment_sensitivity_squared),
          "workload_first": "fixed-P AdamW trajectory",
          "workload_second": "endpoint beta2 EMA",
      },
      "privacy_calibration": asdict(plan.calibration),
  }


def run_cifar10_shadow_jme_bandinv_dpadamw(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
):
  config = load_cifar10_shadow_jme_bandinv_dpadamw_config(config_path)
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
  participation = derive_fixed_cycle_participation(len(train_images), config.epochs, config.batch_size)
  del train_images
  plan = get_or_fit_shadow_jme_plan(config, participation)
  if resume_checkpoint is None:
    resolved = _resolved_config(config, participation, plan)
    resolved["run"] = _run_metadata(paths)
    write_run_configuration(paths, source_yaml=source_yaml, resolved=resolved)
    MetricsCSVWriter(paths.metrics)
  return train_cifar10_shadow_jme_bandinv_dpadamw(
      _train_config(config),
      plan,
      resume_checkpoint=resume_checkpoint,
      checkpoint_path=paths.checkpoint,
      metrics_path=paths.metrics,
  )


__all__ = [
    "Cifar10ShadowJMEBandInvDPAdamWExperimentConfig",
    "get_or_fit_shadow_jme_plan",
    "load_cifar10_shadow_jme_bandinv_dpadamw_config",
    "prepare_cifar10_shadow_jme_bandinv_dpadamw_run",
    "resolve_output_log_dir",
    "run_cifar10_shadow_jme_bandinv_dpadamw",
]
