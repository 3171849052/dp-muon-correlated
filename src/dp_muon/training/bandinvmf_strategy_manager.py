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
  if path.is_file() and not request.force_refit:
    try:
      existing = load_bandinv_strategy(path)
      if _strategy_is_compatible(existing, request) and _metadata_is_compatible(
          path, request
      ):
        return path, existing, "reuse"
    except ValueError:
      pass
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
  save_bandinv_strategy(
      path,
      fitted,
      reduction=request.reduction,
      workload_type="nesterov-trajectory",
      momentum=request.momentum,
      learning_rate=request.learning_rate,
      max_optimizer_steps=request.max_optimizer_steps,
  )
  return path, fitted, "fit"


__all__ = [
    "BandInvMFFitRequest",
    "REPOSITORY_ROOT",
    "get_or_fit_strategy",
    "strategy_artifact_path",
]
