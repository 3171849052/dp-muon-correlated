"""YAML orchestration for segmented correlated DP-AdamW."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.bandinvmf import BandInvMFStrategy, fit_bandinv_strategy
from dp_muon.data import load_cifar10
from .bandinvmf_strategy_manager import (
    LoadedStrategySnapshot,
    PrefixSumBandInvMFFitRequest,
    get_or_fit_prefix_sum_strategy_snapshot,
    prefix_sum_strategy_artifact_path,
    require_compatible_prefix_sum_strategy_snapshot,
)
from .cifar10_driver import (
    SEGMENTED_BANDINV_DPADAMW_ALGORITHM,
    Cifar10SegmentedBandInvDPAdamWTrainConfig,
    train_cifar10_segmented_bandinv_dpadamw,
)
from .cifar10_experiment import FixedCycleParticipation, derive_fixed_cycle_participation
from .nonamplified_segmented_bandinv_dpadamw import SegmentedPlan, fit_segmented_plan
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
class Cifar10SegmentedBandInvDPAdamWExperimentConfig:
  algorithm: Literal["dp-adamw-correlated-segmented"]
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
  segment_length: int
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


def load_cifar10_segmented_bandinv_dpadamw_config(
    path: str | Path,
) -> Cifar10SegmentedBandInvDPAdamWExperimentConfig:
  """Loads the strict segmented correlated DP-AdamW YAML schema."""
  source = Path(path)
  try:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {source}") from error
  expected_sections = {
      "algorithm", "experiment", "runtime", "data", "model", "schedule",
      "strategy", "training", "adamw", "privacy", "output",
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
      {"segment_length", "bandwidth", "reduction", "max_optimizer_steps", "force_refit"},
  )
  adamw = _section(
      document, "adamw", {"learning_rate", "beta1", "beta2", "eps", "weight_decay"}
  )
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _section(document, "output", {"strategy_dir", "checkpoint_dir", "log_dir"})

  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != SEGMENTED_BANDINV_DPADAMW_ALGORITHM:
    raise ValueError(
        "segmented correlated DP-AdamW config requires algorithm: "
        f"{SEGMENTED_BANDINV_DPADAMW_ALGORITHM}"
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
  if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
    raise ValueError("adamw.beta1 and adamw.beta2 must be in [0, 1)")
  config = Cifar10SegmentedBandInvDPAdamWExperimentConfig(
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
      segment_length=_integer(strategy["segment_length"], "strategy.segment_length"),
      bandwidth=_integer(strategy["bandwidth"], "strategy.bandwidth"),
      reduction=reduction,  # type: ignore[arg-type]
      max_optimizer_steps=_integer(
          strategy["max_optimizer_steps"], "strategy.max_optimizer_steps"
      ),
      force_refit=strategy["force_refit"],
      strategy_dir=_string(output["strategy_dir"], "output.strategy_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
  )
  if config.delta >= 1.0:
    raise ValueError("privacy.delta must be less than 1")
  if config.batch_size % config.microbatch_size:
    raise ValueError("training.logical_batch_size must be divisible by training.microbatch_size")
  return config


def strategy_artifact_path(
    config: Cifar10SegmentedBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
    length: int,
) -> Path:
  """Returns the cached strategy artifact for one segment length."""
  return prefix_sum_strategy_artifact_path(
      config.strategy_dir,
      horizon=length,
      min_sep=min(participation.min_sep, length),
      max_participations=participation.max_participations,
      bandwidth=min(config.bandwidth, length),
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )


def strategy_artifact_paths(
    config: Cifar10SegmentedBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
) -> tuple[Path, ...]:
  lengths = []
  full, remainder = divmod(participation.horizon, config.segment_length)
  lengths.extend([config.segment_length] * full)
  if remainder:
    lengths.append(remainder)
  return tuple(strategy_artifact_path(config, participation, length) for length in sorted(set(lengths)))


def _strategy_request(
    config: Cifar10SegmentedBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
    length: int,
) -> PrefixSumBandInvMFFitRequest:
  return PrefixSumBandInvMFFitRequest(
      horizon=length,
      min_sep=min(participation.min_sep, length),
      max_participations=participation.max_participations,
      bandwidth=min(config.bandwidth, length),
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
      strategy_dir=config.strategy_dir,
      force_refit=config.force_refit,
  )


def get_or_fit_segmented_plan(
    config: Cifar10SegmentedBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
    *,
    require_existing: bool = False,
) -> tuple[SegmentedPlan, dict[int, LoadedStrategySnapshot], dict[int, str]]:
  """Loads/fits unique segment lengths, then performs one global calibration."""
  snapshots: dict[int, LoadedStrategySnapshot] = {}
  actions: dict[int, str] = {}

  def fit_cached(length: int, bandwidth: int, min_sep: int, **kwargs: Any) -> BandInvMFStrategy:
    del bandwidth, min_sep, kwargs
    request = _strategy_request(config, participation, length)
    if require_existing:
      snapshot = require_compatible_prefix_sum_strategy_snapshot(request)
      action = "reuse"
    else:
      snapshot, action = get_or_fit_prefix_sum_strategy_snapshot(
          request, fit_strategy=fit_bandinv_strategy
      )
    snapshots[length] = snapshot
    actions[length] = action
    return snapshot.strategy

  plan = fit_segmented_plan(
      horizon=participation.horizon,
      segment_length=config.segment_length,
      bandwidth=config.bandwidth,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      max_optimizer_steps=config.max_optimizer_steps,
      reduction=config.reduction,
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      fit_strategy=fit_cached,
  )
  return plan, snapshots, actions


def resolve_output_log_dir(config_path: str | Path) -> Path:
  directory = Path(load_cifar10_segmented_bandinv_dpadamw_config(config_path).log_dir)
  return directory if directory.is_absolute() else REPOSITORY_ROOT / directory


def _run_metadata(paths: Any) -> dict[str, str]:
  return {
      "directory": str(paths.directory.resolve()),
      "metrics": str(paths.metrics.resolve()),
      "checkpoint": str(paths.checkpoint.resolve()),
  }


def _create_run(config: Cifar10SegmentedBandInvDPAdamWExperimentConfig, document: Mapping[str, Any], source_yaml: str):
  root = Path(config.log_dir)
  if not root.is_absolute():
    root = REPOSITORY_ROOT / root
  paths = create_run_directory(
      root,
      epsilon=config.epsilon,
      bandwidth=f"bandinv-seg{config.segment_length}",
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


def prepare_cifar10_segmented_bandinv_dpadamw_run(config_path: str | Path):
  config = load_cifar10_segmented_bandinv_dpadamw_config(config_path)
  source_yaml = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  return _create_run(config, document, source_yaml)


def _train_config(
    config: Cifar10SegmentedBandInvDPAdamWExperimentConfig,
) -> Cifar10SegmentedBandInvDPAdamWTrainConfig:
  return Cifar10SegmentedBandInvDPAdamWTrainConfig(
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
      segment_length=config.segment_length,
      bandwidth=config.bandwidth,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
      seed=config.seed,
      checkpoint_dir=config.checkpoint_dir,
      eval_every=config.eval_every,
      adjacency=config.adjacency,
  )


def _resolved_config(
    config: Cifar10SegmentedBandInvDPAdamWExperimentConfig,
    participation: FixedCycleParticipation,
    plan: SegmentedPlan,
    snapshots: dict[int, LoadedStrategySnapshot],
    actions: dict[int, str],
) -> dict[str, Any]:
  return {
      "experiment": asdict(config),
      "participation": asdict(participation),
      "segmentation": {
          "segment_length": config.segment_length,
          "block_lengths": list(plan.block_lengths),
          "global_sensitivity_squared": plan.sensitivity_squared,
          "strategies": [
              {
                  "length": length,
                  "artifact": str(snapshots[length].path.resolve()),
                  "sha256": snapshots[length].sha256,
                  "action": actions[length],
                  "workload_type": "decayed-prefix-sum",
                  "horizon": snapshots[length].strategy.horizon,
                  "bandwidth": snapshots[length].strategy.bandwidth,
                  "min_sep": snapshots[length].strategy.min_sep,
                  "max_participations": snapshots[length].strategy.max_participations,
                  "learning_rate": config.learning_rate,
                  "weight_decay": config.weight_decay,
                  "sensitivity_squared": float(snapshots[length].strategy.sensitivity_squared),
              }
              for length in sorted(snapshots)
          ],
      },
      "privacy_calibration": asdict(plan.calibration),
  }


def run_cifar10_segmented_bandinv_dpadamw(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
):
  config = load_cifar10_segmented_bandinv_dpadamw_config(config_path)
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
  plan, snapshots, actions = get_or_fit_segmented_plan(
      config, participation, require_existing=resume_checkpoint is not None
  )
  if resume_checkpoint is None:
    resolved = _resolved_config(config, participation, plan, snapshots, actions)
    resolved["run"] = _run_metadata(paths)
    write_run_configuration(paths, source_yaml=source_yaml, resolved=resolved)
    MetricsCSVWriter(paths.metrics)
  return train_cifar10_segmented_bandinv_dpadamw(
      _train_config(config),
      plan,
      strategy_snapshots=snapshots,
      resume_checkpoint=resume_checkpoint,
      checkpoint_path=paths.checkpoint,
      metrics_path=paths.metrics,
  )


__all__ = [
    "Cifar10SegmentedBandInvDPAdamWExperimentConfig",
    "get_or_fit_segmented_plan",
    "load_cifar10_segmented_bandinv_dpadamw_config",
    "prepare_cifar10_segmented_bandinv_dpadamw_run",
    "resolve_output_log_dir",
    "run_cifar10_segmented_bandinv_dpadamw",
    "strategy_artifact_path",
    "strategy_artifact_paths",
]
