"""YAML orchestration for the non-amplified IID DP-Muon baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from dp_muon.data import load_cifar10
from dp_muon.privacy import calibrate_nonamplified_iid

from .cifar10_driver import Cifar10DPMuonTrainConfig, train_cifar10_dpmuon
from .cifar10_experiment import derive_fixed_cycle_participation
from .run_logging import (
    MetricsCSVWriter, config_content_hash, create_run_directory,
    existing_run_paths, run_paths_from_directory, write_run_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Cifar10DPMuonExperimentConfig:
  algorithm: Literal["dpmuon"]
  name: str; seed: int; gpu: int; data_dir: str; pretrained: str
  epochs: int; logical_batch_size: int; microbatch_size: int; clip_norm: float; eval_every: int
  muon_learning_rate: float; muon_weight_decay: float; momentum: float; ns_steps: int; consistent_rms: float
  adamw_learning_rate: float; adamw_beta1: float; adamw_beta2: float; adamw_eps: float; adamw_weight_decay: float
  epsilon: float; delta: float; adjacency: str; checkpoint_dir: str; log_dir: str
  use_bf16_ns: bool = True


def _mapping(document: Mapping[str, Any], name: str, keys: set[str]) -> Mapping[str, Any]:
  value = document.get(name)
  if not isinstance(value, Mapping) or set(value) != keys:
    raise ValueError(f"config.{name} must contain exactly {sorted(keys)}")
  return value


def _integer(value: object, name: str, *, positive: bool = True) -> int:
  if isinstance(value, bool) or not isinstance(value, Integral) or (positive and value < 1) or (not positive and value < 0):
    raise ValueError(f"{name} must be a {'positive' if positive else 'non-negative'} integer")
  return int(value)


def _number(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
  if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
    raise ValueError(f"{name} must be finite")
  value = float(value)
  if (positive and value <= 0) or (nonnegative and value < 0):
    raise ValueError(f"{name} has an invalid value")
  return value


def _string(value: object, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{name} must be a non-empty string")
  return value


def load_cifar10_dpmuon_config(path: str | Path) -> Cifar10DPMuonExperimentConfig:
  """Loads the explicit Muon/AdamW YAML schema."""
  source = Path(path)
  try:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
  except (OSError, yaml.YAMLError) as error:
    raise ValueError(f"could not read config {source}") from error
  if not isinstance(document, Mapping) or set(document) != {"algorithm", "experiment", "runtime", "data", "model", "training", "muon", "adamw", "schedule", "privacy", "output"}:
    raise ValueError("config has unexpected sections")
  experiment = _mapping(document, "experiment", {"name", "seed"})
  runtime = _mapping(document, "runtime", {"gpu"})
  data = _mapping(document, "data", {"data_dir"})
  model = _mapping(document, "model", {"pretrained"})
  training = _mapping(document, "training", {"epochs", "logical_batch_size", "microbatch_size", "clip_norm", "eval_every"})
  muon = _mapping(document, "muon", {"learning_rate", "weight_decay", "momentum", "ns_steps", "consistent_rms", "use_bf16_ns"})
  adamw = _mapping(document, "adamw", {"learning_rate", "beta1", "beta2", "eps", "weight_decay"})
  schedule = _mapping(document, "schedule", {"mode"})
  privacy = _mapping(document, "privacy", {"epsilon", "delta", "adjacency"})
  output = _mapping(document, "output", {"checkpoint_dir", "log_dir"})
  algorithm = _string(document["algorithm"], "algorithm")
  if algorithm != "dpmuon":
    raise ValueError("DP-Muon config requires algorithm: dpmuon")
  if schedule["mode"] != "fixed_cycle":
    raise ValueError("schedule.mode must be 'fixed_cycle'")
  if privacy["adjacency"] not in {"add_remove", "replace_one"}:
    raise ValueError("privacy.adjacency is invalid")
  if not isinstance(muon["use_bf16_ns"], bool):
    raise ValueError("muon.use_bf16_ns must be boolean")
  result = Cifar10DPMuonExperimentConfig(
      algorithm=algorithm,  # type: ignore[arg-type]
      name=_string(experiment["name"], "experiment.name"), seed=_integer(experiment["seed"], "experiment.seed", positive=False), gpu=_integer(runtime["gpu"], "runtime.gpu", positive=False),
      data_dir=_string(data["data_dir"], "data.data_dir"), pretrained=_string(model["pretrained"], "model.pretrained"),
      epochs=_integer(training["epochs"], "training.epochs"), logical_batch_size=_integer(training["logical_batch_size"], "training.logical_batch_size"), microbatch_size=_integer(training["microbatch_size"], "training.microbatch_size"), clip_norm=_number(training["clip_norm"], "training.clip_norm", positive=True), eval_every=_integer(training["eval_every"], "training.eval_every"),
      muon_learning_rate=_number(muon["learning_rate"], "muon.learning_rate", positive=True), muon_weight_decay=_number(muon["weight_decay"], "muon.weight_decay", nonnegative=True), momentum=_number(muon["momentum"], "muon.momentum", nonnegative=True), ns_steps=_integer(muon["ns_steps"], "muon.ns_steps"), consistent_rms=_number(muon["consistent_rms"], "muon.consistent_rms", positive=True),
      adamw_learning_rate=_number(adamw["learning_rate"], "adamw.learning_rate", positive=True), adamw_beta1=_number(adamw["beta1"], "adamw.beta1", nonnegative=True), adamw_beta2=_number(adamw["beta2"], "adamw.beta2", nonnegative=True), adamw_eps=_number(adamw["eps"], "adamw.eps", positive=True), adamw_weight_decay=_number(adamw["weight_decay"], "adamw.weight_decay", nonnegative=True),
      epsilon=_number(privacy["epsilon"], "privacy.epsilon", positive=True), delta=_number(privacy["delta"], "privacy.delta", positive=True), adjacency=privacy["adjacency"], checkpoint_dir=_string(output["checkpoint_dir"], "output.checkpoint_dir"), log_dir=_string(output["log_dir"], "output.log_dir"), use_bf16_ns=muon["use_bf16_ns"],
  )
  if result.logical_batch_size % result.microbatch_size or result.delta >= 1 or result.momentum >= 1 or result.adamw_beta1 >= 1 or result.adamw_beta2 >= 1:
    raise ValueError("invalid batch division, privacy, or momentum setting")
  return result


def resolve_output_log_dir(config_path: str | Path) -> Path:
  directory = Path(load_cifar10_dpmuon_config(config_path).log_dir)
  return directory if directory.is_absolute() else REPOSITORY_ROOT / directory


def _run_paths(config: Cifar10DPMuonExperimentConfig, document: Mapping[str, Any], source: str):
  root = Path(config.log_dir)
  if not root.is_absolute(): root = REPOSITORY_ROOT / root
  paths = create_run_directory(root, epsilon=config.epsilon, bandwidth="iid", learning_rate=config.muon_learning_rate, clip_norm=config.clip_norm, seed=config.seed, config_hash=config_content_hash(document))
  write_run_configuration(paths, source_yaml=source, resolved={"experiment": asdict(config), "run": {"directory": str(paths.directory.resolve()), "metrics": str(paths.metrics.resolve()), "checkpoint": str(paths.checkpoint.resolve())}})
  MetricsCSVWriter(paths.metrics)
  return paths


def prepare_cifar10_dpmuon_run(config_path: str | Path):
  config = load_cifar10_dpmuon_config(config_path)
  source = Path(config_path).read_text(encoding="utf-8")
  return _run_paths(config, yaml.safe_load(source), source)


def run_cifar10_dpmuon(config_path: str | Path, *, resume_checkpoint: str | Path | None = None, run_dir: str | Path | None = None):
  config = load_cifar10_dpmuon_config(config_path)
  source = Path(config_path).read_text(encoding="utf-8")
  document = yaml.safe_load(source)
  if resume_checkpoint is not None and run_dir is not None:
    raise ValueError("resume_checkpoint and run_dir are mutually exclusive")
  paths = existing_run_paths(resume_checkpoint) if resume_checkpoint else (run_paths_from_directory(run_dir) if run_dir else _run_paths(config, document, source))
  images, _ = load_cifar10(config.data_dir, train=True)
  participation = derive_fixed_cycle_participation(len(images), config.epochs, config.logical_batch_size)
  calibration = calibrate_nonamplified_iid(epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm, normalize_by=float(config.logical_batch_size), adjacency=config.adjacency, max_participations=participation.max_participations)
  if resume_checkpoint is None:
    write_run_configuration(paths, source_yaml=source, resolved={"experiment": asdict(config), "participation": asdict(participation), "strategy": {"algorithm": "nonamplified_iid_dpmuon"}, "privacy_calibration": asdict(calibration)})
  train_config = Cifar10DPMuonTrainConfig(pretrained=config.pretrained, data_dir=config.data_dir, batch_size=config.logical_batch_size, microbatch_size=config.microbatch_size, clip_norm=config.clip_norm, epsilon=config.epsilon, delta=config.delta, muon_learning_rate=config.muon_learning_rate, muon_weight_decay=config.muon_weight_decay, momentum=config.momentum, ns_steps=config.ns_steps, consistent_rms=config.consistent_rms, adamw_learning_rate=config.adamw_learning_rate, adamw_beta1=config.adamw_beta1, adamw_beta2=config.adamw_beta2, adamw_eps=config.adamw_eps, adamw_weight_decay=config.adamw_weight_decay, seed=config.seed, checkpoint_dir=config.checkpoint_dir, eval_every=config.eval_every, horizon=participation.horizon, min_sep=participation.min_sep, max_participations=participation.max_participations, adjacency=config.adjacency, use_bf16_ns=config.use_bf16_ns)
  return train_cifar10_dpmuon(train_config, resume_checkpoint=resume_checkpoint, checkpoint_path=paths.checkpoint, metrics_path=paths.metrics)


__all__ = ["Cifar10DPMuonExperimentConfig", "load_cifar10_dpmuon_config", "prepare_cifar10_dpmuon_run", "resolve_output_log_dir", "run_cifar10_dpmuon"]
