"""Shared BandInvMF strategy artifact fitting and cache management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from dp_muon.bandinvmf import (
    BandInvMFStrategy,
    fit_bandinv_strategy,
    load_bandinv_strategy,
    load_bandinv_strategy_metadata,
    save_bandinv_strategy,
)
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef

from .file_locking import (
    atomic_replace,
    atomic_temporary_path,
    file_fingerprint,
    file_lock,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BandInvMFFitRequest:
  """All public values that determine a cached Nesterov BandInvMF strategy."""

  horizon: int
  min_sep: int
  max_participations: int
  bandwidth: int
  momentum: float
  learning_rate: float
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  strategy_dir: str | Path
  force_refit: bool


@dataclass(frozen=True)
class LoadedStrategySnapshot:
  """A strategy and SHA-256 read while its artifact lock was held."""

  path: Path
  strategy: BandInvMFStrategy
  sha256: str


def strategy_artifact_path(
    strategy_dir: str | Path,
    *,
    horizon: int,
    min_sep: int,
    max_participations: int,
    bandwidth: int,
    momentum: float,
    learning_rate: float,
    reduction: str,
    max_optimizer_steps: int,
) -> Path:
  """Returns a deterministic artifact path, relative directories using repo root."""
  directory = Path(strategy_dir)
  if not directory.is_absolute():
    directory = REPOSITORY_ROOT / directory
  return directory / (
      f"nesterov-trajectory_n{horizon}_p{bandwidth}"
      f"_b{min_sep}_k{max_participations}"
      f"_m{momentum}_lr{learning_rate}_r{reduction}_opt{max_optimizer_steps}.npz"
  )


def _strategy_is_compatible(
    strategy: BandInvMFStrategy, request: BandInvMFFitRequest
) -> bool:
  expected_workload = np.asarray(
      fixed_lr_nesterov_trajectory_workload_coef(
          request.horizon, request.momentum, request.learning_rate
      )
  )
  return (
      strategy.horizon == request.horizon
      and strategy.bandwidth == request.bandwidth
      and strategy.min_sep == request.min_sep
      and strategy.max_participations == request.max_participations
      and np.array_equal(np.asarray(strategy.workload_coef), expected_workload)
  )


def _metadata_is_compatible(path: Path, request: BandInvMFFitRequest) -> bool:
  metadata = load_bandinv_strategy_metadata(path)
  return (
      metadata.workload_type == "nesterov-trajectory"
      and metadata.momentum == request.momentum
      and metadata.learning_rate == request.learning_rate
      and metadata.reduction == request.reduction
      and metadata.max_optimizer_steps == request.max_optimizer_steps
  )


def _load_compatible_snapshot_unlocked(
    path: Path, request: BandInvMFFitRequest
) -> LoadedStrategySnapshot | None:
  try:
    strategy = load_bandinv_strategy(path)
    if _strategy_is_compatible(strategy, request) and _metadata_is_compatible(
        path, request
    ):
      return LoadedStrategySnapshot(path, strategy, file_fingerprint(path))
  except ValueError:
    pass
  return None


def load_strategy_snapshot(path: str | Path) -> LoadedStrategySnapshot:
  """Loads and fingerprints one immutable strategy version under its lock."""
  resolved = Path(path).resolve()
  with file_lock(resolved):
    try:
      strategy = load_bandinv_strategy(resolved)
      load_bandinv_strategy_metadata(resolved)
      return LoadedStrategySnapshot(
          resolved, strategy, file_fingerprint(resolved)
      )
    except ValueError as error:
      raise ValueError(f"could not load strategy snapshot {resolved}") from error


def _artifact_fingerprint(path: Path) -> str | None:
  try:
    return file_fingerprint(path)
  except ValueError:
    return None


def require_compatible_strategy(
    request: BandInvMFFitRequest,
) -> tuple[Path, BandInvMFStrategy]:
  """Loads the requested cache or fails; used for strict checkpoint resume."""
  path = strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      momentum=request.momentum, learning_rate=request.learning_rate,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
  )
  with file_lock(path):
    snapshot = _load_compatible_snapshot_unlocked(path, request)
  if snapshot is None:
    raise ValueError("required compatible BandInvMF strategy artifact is missing or invalid")
  return path, snapshot.strategy


def get_or_fit_strategy(
    request: BandInvMFFitRequest,
    *,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> tuple[Path, BandInvMFStrategy, Literal["reuse", "fit"]]:
  """Returns a compatible cache entry or fits and saves a replacement."""
  if request.bandwidth > request.horizon:
    raise ValueError("strategy.bandwidth must not exceed derived horizon")
  path = strategy_artifact_path(
      request.strategy_dir,
      horizon=request.horizon,
      min_sep=request.min_sep,
      max_participations=request.max_participations,
      bandwidth=request.bandwidth,
      momentum=request.momentum,
      learning_rate=request.learning_rate,
      reduction=request.reduction,
      max_optimizer_steps=request.max_optimizer_steps,
  )
  initial_fingerprint = _artifact_fingerprint(path)
  if not request.force_refit:
    existing = _load_compatible_snapshot_unlocked(path, request)
    if existing is not None:
      return path, existing.strategy, "reuse"
  with file_lock(path):
    existing = _load_compatible_snapshot_unlocked(path, request)
    # For a forced refit, a compatible publication which happened while this
    # caller waited satisfies its request for a newer version. A lone caller
    # observes the same fingerprint and therefore still truly refits.
    if existing is not None and (
        not request.force_refit or existing.sha256 != initial_fingerprint
    ):
      return path, existing.strategy, "reuse"
    workload_coef = fixed_lr_nesterov_trajectory_workload_coef(
        request.horizon, request.momentum, request.learning_rate
    )
    fitted = fit_strategy(
        request.horizon,
        request.bandwidth,
        request.min_sep,
        max_participations=request.max_participations,
        workload_coef=workload_coef,
        max_optimizer_steps=request.max_optimizer_steps,
        reduction=request.reduction,
    )
    # Validate the completed temporary artifact before it becomes visible.
    with atomic_temporary_path(path) as temporary:
      save_bandinv_strategy(
          temporary,
          fitted,
          reduction=request.reduction,
          workload_type="nesterov-trajectory",
          momentum=request.momentum,
          learning_rate=request.learning_rate,
          max_optimizer_steps=request.max_optimizer_steps,
      )
      if _load_compatible_snapshot_unlocked(temporary, request) is None:
        raise ValueError("fitted BandInvMF artifact failed validation")
      atomic_replace(temporary, path)
    return path, fitted, "fit"


__all__ = [
    "BandInvMFFitRequest",
    "LoadedStrategySnapshot",
    "REPOSITORY_ROOT",
    "get_or_fit_strategy",
    "load_strategy_snapshot",
    "require_compatible_strategy",
    "strategy_artifact_path",
]
