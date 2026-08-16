"""Durable per-run files for CIFAR-10 experiment orchestration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .file_locking import atomic_write_text, file_lock


METRICS_FIELDS = (
    "epoch",
    "step",
    "effective_epoch",
    "epsilon_spent",
    "test_loss",
    "test_accuracy",
    "elapsed_seconds",
    "eval_seconds",
)


@dataclass(frozen=True)
class RunPaths:
  directory: Path
  config: Path
  resolved_config: Path
  metrics: Path
  train_log: Path
  checkpoint: Path


def config_content_hash(document: Mapping[str, Any]) -> str:
  """Returns a short stable hash of the complete parsed YAML configuration."""
  encoded = yaml.safe_dump(dict(document), sort_keys=True).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()[:12]


def _value(value: float | int) -> str:
  """Uses the normal public configuration spelling in a directory component."""
  return str(value)


def create_run_directory(
    log_root: str | Path,
    *,
    epsilon: float,
    bandwidth: int | str,
    learning_rate: float,
    clip_norm: float,
    seed: int,
    config_hash: str,
    now: datetime | None = None,
) -> RunPaths:
  """Creates a collision-free timestamped run directory.

  Directory names have second precision by design.  When two launches happen
  in the same second, advancing the timestamp component keeps the documented
  name format while still ensuring that no experiment overwrites another.
  """
  root = Path(log_root)
  root.mkdir(parents=True, exist_ok=True)
  timestamp = now or datetime.now()
  for collision in range(10_000):
    stamp = (timestamp + timedelta(seconds=collision)).strftime("%Y%m%d-%H%M%S")
    directory = root / (
        f"{stamp}_eps{_value(epsilon)}_bw{_value(bandwidth)}"
        f"_lr{_value(learning_rate)}_clip{_value(clip_norm)}"
        f"_s{_value(seed)}_{config_hash}"
    )
    try:
      directory.mkdir()
      checkpoint = directory / "checkpoints" / "latest.pkl"
      return RunPaths(
          directory=directory,
          config=directory / "config.yaml",
          resolved_config=directory / "resolved_config.yaml",
          metrics=directory / "metrics.csv",
          train_log=directory / "train.log",
          checkpoint=checkpoint,
      )
    except FileExistsError:
      continue
  raise RuntimeError("could not allocate a unique run directory")


def existing_run_paths(resume_checkpoint: str | Path) -> RunPaths:
  """Finds the run directory owning ``checkpoints/latest.pkl``."""
  checkpoint = Path(resume_checkpoint).resolve()
  if checkpoint.name != "latest.pkl" or checkpoint.parent.name != "checkpoints":
    raise ValueError("resume checkpoint must be a run's checkpoints/latest.pkl")
  directory = checkpoint.parent.parent
  required = (directory / "config.yaml", directory / "resolved_config.yaml")
  if not checkpoint.is_file() or not all(path.is_file() for path in required):
    raise ValueError("resume checkpoint does not belong to a complete run directory")
  return RunPaths(
      directory=directory,
      config=directory / "config.yaml",
      resolved_config=directory / "resolved_config.yaml",
      metrics=directory / "metrics.csv",
      train_log=directory / "train.log",
      checkpoint=checkpoint,
  )


def run_paths_from_directory(directory: str | Path) -> RunPaths:
  """Returns an already-created run directory before it has a checkpoint."""
  root = Path(directory).resolve()
  if not root.is_dir() or not (root / "config.yaml").is_file():
    raise ValueError("run directory must have been created by the CIFAR runner")
  return RunPaths(
      directory=root,
      config=root / "config.yaml",
      resolved_config=root / "resolved_config.yaml",
      metrics=root / "metrics.csv",
      train_log=root / "train.log",
      checkpoint=root / "checkpoints" / "latest.pkl",
  )


def write_run_configuration(
    paths: RunPaths,
    *,
    source_yaml: str,
    resolved: Mapping[str, Any],
) -> None:
  """Stores the immutable source snapshot and the derived public settings."""
  # One run-level lock keeps the two snapshots mutually coherent to readers.
  with file_lock(paths.directory / ".snapshots"):
    atomic_write_text(paths.config, source_yaml)
    atomic_write_text(
        paths.resolved_config, yaml.safe_dump(dict(resolved), sort_keys=True)
    )
    paths.train_log.touch(exist_ok=True)


class MetricsCSVWriter:
  """Appends flushed evaluation rows and protects resumed runs from duplicates."""

  def __init__(self, path: str | Path):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._seen_steps: set[int] = set()
    with file_lock(self.path):
      self._initialize_locked()

  def _initialize_locked(self) -> set[int]:
    if not self.path.exists() or not self.path.stat().st_size:
      with self.path.open("w", newline="", encoding="utf-8") as target:
        csv.DictWriter(target, fieldnames=METRICS_FIELDS).writeheader()
        target.flush()
        os.fsync(target.fileno())
      return set()
    with self.path.open(newline="", encoding="utf-8") as source:
      reader = csv.DictReader(source)
      if tuple(reader.fieldnames or ()) != METRICS_FIELDS:
        raise ValueError("metrics.csv has unexpected fields")
      return {int(row["step"]) for row in reader}

  def append(self, record: Mapping[str, float | int]) -> bool:
    if set(record) != set(METRICS_FIELDS):
      raise ValueError("metrics record must contain exactly the CSV fields")
    step = int(record["step"])
    with file_lock(self.path):
      seen_steps = self._initialize_locked()
      if step in seen_steps:
        self._seen_steps = seen_steps
        return False
      with self.path.open("a", newline="", encoding="utf-8") as target:
        csv.DictWriter(target, fieldnames=METRICS_FIELDS).writerow(record)
        target.flush()
        os.fsync(target.fileno())
      self._seen_steps = seen_steps | {step}
      return True


__all__ = [
    "METRICS_FIELDS",
    "MetricsCSVWriter",
    "RunPaths",
    "config_content_hash",
    "create_run_directory",
    "existing_run_paths",
    "run_paths_from_directory",
    "write_run_configuration",
]
