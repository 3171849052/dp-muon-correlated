"""YAML orchestration for Public-(V) + Frozen AdamW + BandInvMF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.data import load_public_private_cifar

from .cifar10_driver import (
    PUBLIC_V_BANDINV_DPADAMW_ALGORITHM,
    Cifar10PublicVBandInvDPAdamWTrainConfig,
    train_dp_public_v_bandinv,
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
class Cifar10PublicVBandInvExperimentConfig:
  algorithm: Literal["dp-adamw-public-v-bandinv"]
  name: str
  seed: int
  gpu: int
  data_dir: str
  public_source: Literal["cifar10_split", "cifar100_10class"]
  cifar10_public_size: int
  public_split_seed: int
  cifar100_public_classes: tuple[int, ...]
  pretrained: str
  epochs: int
  logical_batch_size: int
  microbatch_size: int
  clip_norm: float
  eval_every: int
  segment_length: int
  public_v_beta2: float
  public_v_eps: float
  public_v_batches_per_segment: int
  learning_rate: float
  beta1: float
  weight_decay: float
  schedule_mode: Literal["fixed_cycle"]
  bandwidth: int
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  epsilon: float
  delta: float
  adjacency: Literal["add_remove", "replace_one"]
  checkpoint_dir: str
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
      or (value < 1 if positive else value < 0)
  ):
    qualifier = "positive" if positive else "non-negative"
    raise ValueError(f"{name} must be a {qualifier} integer")
  return int(value)


def _number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
  if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
    raise ValueError(f"{name} must be finite")
  result = float(value)
  if (positive and result <= 0) or (nonnegative and result < 0):
    raise ValueError(f"{name} has an invalid value")
  return result


def load_cifar10_public_v_bandinv_config(
    path: str | Path,
) -> Cifar10PublicVBandInvExperimentConfig:
  """Loads the strict Public-V BandInvMF experiment schema."""
  source = Path(path)
  try:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {source}") from error
  expected = {
      "algorithm",
      "experiment",
      "runtime",
      "data",
      "model",
      "training",
      "public_v",
      "adamw",
      "schedule",
      "strategy",
      "privacy",
      "output",
  }
  if not isinstance(document, Mapping) or set(document) != expected:
    raise ValueError(f"config sections must be exactly {sorted(expected)}")
  experiment = _section(document, "experiment", {"name", "seed"})
  runtime = _section(document, "runtime", {"gpu"})
  data = _section(
      document,
      "data",
      {
          "data_dir",
          "public_source",
          "cifar10_public_size",
          "public_split_seed",
          "cifar100_public_classes",
      },
  )
  model = _section(document, "model", {"pretrained"})
  training = _section(
      document,
      "training",
      {
          "epochs",
          "logical_batch_size",
          "microbatch_size",
          "clip_norm",
          "eval_every",
          "segment_length",
      },
  )
  public_v = _section(
      document,
      "public_v",
      {"public_v_beta2", "public_v_eps", "public_v_batches_per_segment"},
  )
  adamw = _section(document, "adamw", {"learning_rate", "beta1", "weight_decay"})
  schedule = _section(document, "schedule", {"mode"})
  strategy = _section(
      document, "strategy", {"bandwidth", "reduction", "max_optimizer_steps"}
  )
  privacy = _section(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _section(document, "output", {"checkpoint_dir", "log_dir"})

  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != PUBLIC_V_BANDINV_DPADAMW_ALGORITHM:
    raise ValueError(f"config requires algorithm: {PUBLIC_V_BANDINV_DPADAMW_ALGORITHM}")
  public_source = _string(data["public_source"], "data.public_source")
  if public_source not in {"cifar10_split", "cifar100_10class"}:
    raise ValueError("data.public_source is invalid")
  classes_value = data["cifar100_public_classes"]
  if not isinstance(classes_value, list):
    raise ValueError("data.cifar100_public_classes must be a list")
  classes = tuple(
      _integer(value, "data.cifar100_public_classes entry", positive=False)
      for value in classes_value
  )
  if (
      len(classes) != 10
      or len(set(classes)) != 10
      or any(value >= 100 for value in classes)
  ):
    raise ValueError("data.cifar100_public_classes must contain 10 unique IDs in [0, 99]")
  beta1 = _number(adamw["beta1"], "adamw.beta1", nonnegative=True)
  beta2 = _number(public_v["public_v_beta2"], "public_v.public_v_beta2", nonnegative=True)
  if beta1 >= 1 or beta2 >= 1:
    raise ValueError("Adam beta values must be less than 1")
  mode = _string(schedule["mode"], "schedule.mode")
  reduction = _string(strategy["reduction"], "strategy.reduction")
  adjacency = _string(privacy["adjacency"], "privacy.adjacency")
  if mode != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  if reduction not in {"mean", "max", "last"}:
    raise ValueError("strategy.reduction is invalid")
  if adjacency not in {"add_remove", "replace_one"}:
    raise ValueError("privacy.adjacency is invalid")

  config = Cifar10PublicVBandInvExperimentConfig(
      algorithm=algorithm,  # type: ignore[arg-type]
      name=_string(experiment["name"], "experiment.name"),
      seed=_integer(experiment["seed"], "experiment.seed", positive=False),
      gpu=_integer(runtime["gpu"], "runtime.gpu", positive=False),
      data_dir=_string(data["data_dir"], "data.data_dir"),
      public_source=public_source,  # type: ignore[arg-type]
      cifar10_public_size=_integer(data["cifar10_public_size"], "data.cifar10_public_size"),
      public_split_seed=_integer(data["public_split_seed"], "data.public_split_seed", positive=False),
      cifar100_public_classes=classes,
      pretrained=_string(model["pretrained"], "model.pretrained"),
      epochs=_integer(training["epochs"], "training.epochs"),
      logical_batch_size=_integer(training["logical_batch_size"], "training.logical_batch_size"),
      microbatch_size=_integer(training["microbatch_size"], "training.microbatch_size"),
      clip_norm=_number(training["clip_norm"], "training.clip_norm", positive=True),
      eval_every=_integer(training["eval_every"], "training.eval_every"),
      segment_length=_integer(training["segment_length"], "training.segment_length"),
      public_v_beta2=beta2,
      public_v_eps=_number(public_v["public_v_eps"], "public_v.public_v_eps", positive=True),
      public_v_batches_per_segment=_integer(
          public_v["public_v_batches_per_segment"],
          "public_v.public_v_batches_per_segment",
      ),
      learning_rate=_number(adamw["learning_rate"], "adamw.learning_rate", positive=True),
      beta1=beta1,
      weight_decay=_number(adamw["weight_decay"], "adamw.weight_decay", nonnegative=True),
      schedule_mode=mode,  # type: ignore[arg-type]
      bandwidth=_integer(strategy["bandwidth"], "strategy.bandwidth"),
      reduction=reduction,  # type: ignore[arg-type]
      max_optimizer_steps=_integer(
          strategy["max_optimizer_steps"], "strategy.max_optimizer_steps"
      ),
      epsilon=_number(privacy["epsilon"], "privacy.epsilon", positive=True),
      delta=_number(privacy["delta"], "privacy.delta", positive=True),
      adjacency=adjacency,  # type: ignore[arg-type]
      checkpoint_dir=_string(output["checkpoint_dir"], "output.checkpoint_dir"),
      log_dir=_string(output["log_dir"], "output.log_dir"),
  )
  if config.delta >= 1:
    raise ValueError("privacy.delta must be less than 1")
  if config.logical_batch_size % config.microbatch_size:
    raise ValueError("training.logical_batch_size must be divisible by microbatch_size")
  return config


def resolve_output_log_dir(path: str | Path) -> Path:
  directory = Path(load_cifar10_public_v_bandinv_config(path).log_dir)
  return directory if directory.is_absolute() else REPOSITORY_ROOT / directory


def _run_metadata(paths: Any) -> dict[str, str]:
  return {
      "directory": str(paths.directory.resolve()),
      "metrics": str(paths.metrics.resolve()),
      "checkpoint": str(paths.checkpoint.resolve()),
  }


def _create_run(
    config: Cifar10PublicVBandInvExperimentConfig,
    document: Mapping[str, Any],
    source_yaml: str,
):
  root = Path(config.log_dir)
  if not root.is_absolute():
    root = REPOSITORY_ROOT / root
  paths = create_run_directory(
      root,
      epsilon=config.epsilon,
      bandwidth=config.bandwidth,
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


def prepare_cifar10_public_v_bandinv_run(path: str | Path):
  """Creates a run directory without loading data, fitting, or training."""
  config = load_cifar10_public_v_bandinv_config(path)
  source_yaml = Path(path).read_text(encoding="utf-8")
  document = yaml.safe_load(source_yaml)
  if not isinstance(document, Mapping):
    raise ValueError("config must be a mapping")
  return _create_run(config, document, source_yaml)


def _train_config(
    config: Cifar10PublicVBandInvExperimentConfig,
    participation: FixedCycleParticipation,
) -> Cifar10PublicVBandInvDPAdamWTrainConfig:
  return Cifar10PublicVBandInvDPAdamWTrainConfig(
      pretrained=config.pretrained,
      data_dir=config.data_dir,
      batch_size=config.logical_batch_size,
      microbatch_size=config.microbatch_size,
      clip_norm=config.clip_norm,
      epsilon=config.epsilon,
      delta=config.delta,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      weight_decay=config.weight_decay,
      public_source=config.public_source,
      cifar10_public_size=config.cifar10_public_size,
      public_split_seed=config.public_split_seed,
      cifar100_public_classes=config.cifar100_public_classes,
      public_v_beta2=config.public_v_beta2,
      public_v_eps=config.public_v_eps,
      public_v_batches_per_segment=config.public_v_batches_per_segment,
      segment_length=config.segment_length,
      bandwidth=config.bandwidth,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
      seed=config.seed,
      checkpoint_dir=config.checkpoint_dir,
      eval_every=config.eval_every,
      horizon=participation.horizon,
      min_sep=participation.min_sep,
      max_participations=participation.max_participations,
      adjacency=config.adjacency,
  )


def run_cifar10_public_v_bandinv(
    path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
):
  """Loads datasets, derives private participation, and invokes the trainer."""
  config = load_cifar10_public_v_bandinv_config(path)
  source_yaml = Path(path).read_text(encoding="utf-8")
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
  data = load_public_private_cifar(
      config.data_dir,
      public_source=config.public_source,
      cifar10_public_size=config.cifar10_public_size,
      public_split_seed=config.public_split_seed,
      cifar100_public_classes=config.cifar100_public_classes,
  )
  participation = derive_fixed_cycle_participation(
      len(data.private_images), config.epochs, config.logical_batch_size
  )
  if resume_checkpoint is None:
    resolved = {
        "experiment": asdict(config),
        "participation": asdict(participation),
        "private_num_examples": len(data.private_images),
        "public_num_examples": len(data.public_images),
        "private_sample_rate": config.logical_batch_size / len(data.private_images),
        "num_segments": math.ceil(participation.horizon / config.segment_length),
        "privacy": {
            "accounting": "nonamplified GDP; equal mu^2 across independent segments"
        },
        "run": _run_metadata(paths),
    }
    write_run_configuration(paths, source_yaml=source_yaml, resolved=resolved)
    MetricsCSVWriter(paths.metrics)
  return train_dp_public_v_bandinv(
      _train_config(config, participation),
      resume_checkpoint=resume_checkpoint,
      checkpoint_path=paths.checkpoint,
      metrics_path=paths.metrics,
  )


__all__ = [
    "Cifar10PublicVBandInvExperimentConfig",
    "load_cifar10_public_v_bandinv_config",
    "prepare_cifar10_public_v_bandinv_run",
    "resolve_output_log_dir",
    "run_cifar10_public_v_bandinv",
]
