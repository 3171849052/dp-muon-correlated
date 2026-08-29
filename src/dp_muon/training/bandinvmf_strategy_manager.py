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
from dp_muon.optim import (
    decayed_prefix_sum_workload_coef,
    fixed_lr_nesterov_decayed_trajectory_workload_coef,
    frozen_p_time_workload,
)

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
  weight_decay: float = 0.0


@dataclass(frozen=True)
class PrefixSumBandInvMFFitRequest:
  """All public values that determine a cached prefix-sum BandInvMF strategy.

  The decayed-prefix workload is determined by AdamW's learning rate and
  decoupled weight decay.
  """

  horizon: int
  min_sep: int
  max_participations: int
  bandwidth: int
  reduction: Literal["mean", "max", "last"]
  max_optimizer_steps: int
  strategy_dir: str | Path
  force_refit: bool
  learning_rate: float = 1.0
  weight_decay: float = 0.0


@dataclass(frozen=True)
class FrozenPBandInvMFFitRequest:
  """All public values determining one continuous frozen-p Phase-II artifact."""

  horizon: int
  switch_step: int
  min_sep: int
  max_participations: int
  bandwidth: int
  beta1: float
  learning_rate: float
  weight_decay: float
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
    weight_decay: float = 0.0,
    reduction: str,
    max_optimizer_steps: int,
) -> Path:
  """Returns a deterministic artifact path, relative directories using repo root."""
  directory = Path(strategy_dir)
  if not directory.is_absolute():
    directory = REPOSITORY_ROOT / directory
  return directory / (
      f"nesterov-decayed-trajectory_n{horizon}_p{bandwidth}"
      f"_b{min_sep}_k{max_participations}"
      f"_m{momentum}_lr{learning_rate}_wd{weight_decay}_r{reduction}_opt{max_optimizer_steps}.npz"
  )


def prefix_sum_strategy_artifact_path(
    strategy_dir: str | Path,
    *,
    horizon: int,
    min_sep: int,
    max_participations: int,
    bandwidth: int,
    learning_rate: float = 1.0,
    weight_decay: float = 0.0,
    reduction: str,
    max_optimizer_steps: int,
) -> Path:
  """Returns a deterministic decayed-prefix artifact path."""
  directory = Path(strategy_dir)
  if not directory.is_absolute():
    directory = REPOSITORY_ROOT / directory
  return directory / (
      f"decayed-prefix-sum_n{horizon}_p{bandwidth}"
      f"_b{min_sep}_k{max_participations}"
      f"_lr{learning_rate}_wd{weight_decay}_r{reduction}_opt{max_optimizer_steps}.npz"
  )


def frozen_p_strategy_artifact_path(
    strategy_dir: str | Path,
    *,
    horizon: int,
    switch_step: int,
    min_sep: int,
    max_participations: int,
    bandwidth: int,
    beta1: float,
    learning_rate: float,
    weight_decay: float,
    reduction: str,
    max_optimizer_steps: int,
) -> Path:
  """Returns a deterministic path for the single continuous Phase-II plan."""
  directory = Path(strategy_dir)
  if not directory.is_absolute():
    directory = REPOSITORY_ROOT / directory
  return directory / (
      f"frozen-p-continuous_n{horizon}_tau{switch_step}_p{bandwidth}"
      f"_b{min_sep}_k{max_participations}_b1{beta1}_lr{learning_rate}"
      f"_wd{weight_decay}_r{reduction}_opt{max_optimizer_steps}.npz"
  )


def _strategy_is_compatible(
    strategy: BandInvMFStrategy, request: BandInvMFFitRequest
) -> bool:
  expected_workload = np.asarray(
      fixed_lr_nesterov_decayed_trajectory_workload_coef(
          request.horizon, request.momentum, request.learning_rate,
          request.weight_decay,
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
      metadata.workload_type == "nesterov-decayed-trajectory"
      and metadata.momentum == request.momentum
      and metadata.learning_rate == request.learning_rate
      and metadata.weight_decay == request.weight_decay
      and metadata.reduction == request.reduction
      and metadata.max_optimizer_steps == request.max_optimizer_steps
  )


def _prefix_sum_strategy_is_compatible(
    strategy: BandInvMFStrategy, request: PrefixSumBandInvMFFitRequest
) -> bool:
  return (
      strategy.horizon == request.horizon
      and strategy.bandwidth == request.bandwidth
      and strategy.min_sep == request.min_sep
      and strategy.max_participations == request.max_participations
      and np.array_equal(
          np.asarray(strategy.workload_coef),
          np.asarray(decayed_prefix_sum_workload_coef(
              request.horizon, request.learning_rate, request.weight_decay
          )),
      )
  )


def _prefix_sum_metadata_is_compatible(
    path: Path, request: PrefixSumBandInvMFFitRequest
) -> bool:
  metadata = load_bandinv_strategy_metadata(path)
  return (
      metadata.workload_type == "decayed-prefix-sum"
      and metadata.momentum is None
      and metadata.learning_rate == request.learning_rate
      and metadata.weight_decay == request.weight_decay
      and metadata.reduction == request.reduction
      and metadata.max_optimizer_steps == request.max_optimizer_steps
  )


def _frozen_p_strategy_is_compatible(
    strategy: BandInvMFStrategy, request: FrozenPBandInvMFFitRequest
) -> bool:
  expected_workload = np.abs(np.asarray(frozen_p_time_workload(
      request.horizon - request.switch_step,
      tau=request.switch_step,
      beta1=request.beta1,
      learning_rate=request.learning_rate,
      weight_decay=request.weight_decay,
  )))
  return (
      strategy.horizon == request.horizon - request.switch_step
      and strategy.bandwidth == min(request.bandwidth, strategy.horizon)
      and strategy.min_sep == min(request.min_sep, strategy.horizon)
      and strategy.max_participations == request.max_participations
      and strategy.workload_matrix is not None
      and np.allclose(np.asarray(strategy.workload_matrix), expected_workload,
                      rtol=1e-6, atol=1e-8)
  )


def _frozen_p_metadata_is_compatible(
    path: Path, request: FrozenPBandInvMFFitRequest
) -> bool:
  metadata = load_bandinv_strategy_metadata(path)
  return (
      metadata.workload_type == "frozen-p-continuous"
      and metadata.momentum is None
      and metadata.learning_rate == request.learning_rate
      and metadata.weight_decay == request.weight_decay
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


def require_compatible_strategy_snapshot(
    request: BandInvMFFitRequest,
) -> LoadedStrategySnapshot:
  """Loads the requested cache or fails; used for strict checkpoint resume."""
  path = strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      momentum=request.momentum, learning_rate=request.learning_rate,
      weight_decay=request.weight_decay,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
  )
  with file_lock(path):
    snapshot = _load_compatible_snapshot_unlocked(path, request)
  if snapshot is None:
    raise ValueError("required compatible BandInvMF strategy artifact is missing or invalid")
  return snapshot


def require_compatible_strategy(
    request: BandInvMFFitRequest,
) -> tuple[Path, BandInvMFStrategy]:
  """Compatibility adapter returning the historical path/strategy tuple."""
  snapshot = require_compatible_strategy_snapshot(request)
  return snapshot.path, snapshot.strategy


def get_or_fit_strategy_snapshot(
    request: BandInvMFFitRequest,
    *,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> tuple[LoadedStrategySnapshot, Literal["reuse", "fit"]]:
  """Returns the exact published snapshot or fits it under the artifact lock."""
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
      weight_decay=request.weight_decay,
      reduction=request.reduction,
      max_optimizer_steps=request.max_optimizer_steps,
  )
  initial_fingerprint = _artifact_fingerprint(path)
  if not request.force_refit:
    # Fast miss/hit observation; the locked check below supplies the snapshot.
    existing = _load_compatible_snapshot_unlocked(path, request)
  with file_lock(path):
    existing = _load_compatible_snapshot_unlocked(path, request)
    # For a forced refit, a compatible publication which happened while this
    # caller waited satisfies its request for a newer version. A lone caller
    # observes the same fingerprint and therefore still truly refits.
    if existing is not None and (
        not request.force_refit or existing.sha256 != initial_fingerprint
    ):
      return existing, "reuse"
    workload_coef = fixed_lr_nesterov_decayed_trajectory_workload_coef(
        request.horizon, request.momentum, request.learning_rate,
        request.weight_decay,
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
          workload_type="nesterov-decayed-trajectory",
          momentum=request.momentum,
          learning_rate=request.learning_rate,
          weight_decay=request.weight_decay,
          max_optimizer_steps=request.max_optimizer_steps,
      )
      if _load_compatible_snapshot_unlocked(temporary, request) is None:
        raise ValueError("fitted BandInvMF artifact failed validation")
      atomic_replace(temporary, path)
    snapshot = _load_compatible_snapshot_unlocked(path, request)
    if snapshot is None:
      raise ValueError("published BandInvMF artifact failed validation")
    return snapshot, "fit"


def get_or_fit_strategy(
    request: BandInvMFFitRequest,
    *,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> tuple[Path, BandInvMFStrategy, Literal["reuse", "fit"]]:
  """Compatibility adapter returning the historical path/strategy tuple."""
  snapshot, action = get_or_fit_strategy_snapshot(
      request, fit_strategy=fit_strategy
  )
  return snapshot.path, snapshot.strategy, action


def _load_compatible_prefix_sum_snapshot_unlocked(
    path: Path, request: PrefixSumBandInvMFFitRequest
) -> LoadedStrategySnapshot | None:
  try:
    strategy = load_bandinv_strategy(path)
    if _prefix_sum_strategy_is_compatible(strategy, request) and _prefix_sum_metadata_is_compatible(
        path, request
    ):
      return LoadedStrategySnapshot(path, strategy, file_fingerprint(path))
  except ValueError:
    pass
  return None


def require_compatible_prefix_sum_strategy_snapshot(
    request: PrefixSumBandInvMFFitRequest,
) -> LoadedStrategySnapshot:
  """Loads the requested prefix-sum strategy or fails; used for strict checkpoint resume."""
  path = prefix_sum_strategy_artifact_path(
      request.strategy_dir, horizon=request.horizon, min_sep=request.min_sep,
      max_participations=request.max_participations, bandwidth=request.bandwidth,
      reduction=request.reduction, max_optimizer_steps=request.max_optimizer_steps,
      learning_rate=request.learning_rate, weight_decay=request.weight_decay,
  )
  with file_lock(path):
    snapshot = _load_compatible_prefix_sum_snapshot_unlocked(path, request)
  if snapshot is None:
    raise ValueError("required compatible prefix-sum BandInvMF strategy artifact is missing or invalid")
  return snapshot


def get_or_fit_prefix_sum_strategy_snapshot(
    request: PrefixSumBandInvMFFitRequest,
    *,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> tuple[LoadedStrategySnapshot, Literal["reuse", "fit"]]:
  """Returns the exact published prefix-sum snapshot or fits it under the artifact lock."""
  if request.bandwidth > request.horizon:
    raise ValueError("strategy.bandwidth must not exceed derived horizon")
  path = prefix_sum_strategy_artifact_path(
      request.strategy_dir,
      horizon=request.horizon,
      min_sep=request.min_sep,
      max_participations=request.max_participations,
      bandwidth=request.bandwidth,
      reduction=request.reduction,
      max_optimizer_steps=request.max_optimizer_steps,
      learning_rate=request.learning_rate,
      weight_decay=request.weight_decay,
  )
  initial_fingerprint = _artifact_fingerprint(path)
  if not request.force_refit:
    existing = _load_compatible_prefix_sum_snapshot_unlocked(path, request)
  with file_lock(path):
    existing = _load_compatible_prefix_sum_snapshot_unlocked(path, request)
    if existing is not None and (
        not request.force_refit or existing.sha256 != initial_fingerprint
    ):
      return existing, "reuse"
    workload_coef = decayed_prefix_sum_workload_coef(
        request.horizon, request.learning_rate, request.weight_decay
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
    with atomic_temporary_path(path) as temporary:
      save_bandinv_strategy(
          temporary,
          fitted,
          reduction=request.reduction,
          workload_type="decayed-prefix-sum",
          momentum=None,
          learning_rate=request.learning_rate,
          weight_decay=request.weight_decay,
          max_optimizer_steps=request.max_optimizer_steps,
      )
      if _load_compatible_prefix_sum_snapshot_unlocked(temporary, request) is None:
        raise ValueError("fitted prefix-sum BandInvMF artifact failed validation")
      atomic_replace(temporary, path)
    snapshot = _load_compatible_prefix_sum_snapshot_unlocked(path, request)
    if snapshot is None:
      raise ValueError("published prefix-sum BandInvMF artifact failed validation")
    return snapshot, "fit"


def _load_compatible_frozen_p_snapshot_unlocked(
    path: Path, request: FrozenPBandInvMFFitRequest
) -> LoadedStrategySnapshot | None:
  try:
    strategy = load_bandinv_strategy(path)
    if _frozen_p_strategy_is_compatible(strategy, request) and _frozen_p_metadata_is_compatible(
        path, request
    ):
      return LoadedStrategySnapshot(path, strategy, file_fingerprint(path))
  except ValueError:
    pass
  return None


def require_compatible_frozen_p_strategy_snapshot(
    request: FrozenPBandInvMFFitRequest,
) -> LoadedStrategySnapshot:
  """Loads the existing continuous frozen-p artifact without refitting."""
  path = frozen_p_strategy_artifact_path(
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
  with file_lock(path):
    snapshot = _load_compatible_frozen_p_snapshot_unlocked(path, request)
  if snapshot is None:
    raise ValueError("required compatible frozen-p strategy artifact is missing or invalid")
  return snapshot


def get_or_fit_frozen_p_strategy_snapshot(
    request: FrozenPBandInvMFFitRequest,
    *,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> tuple[LoadedStrategySnapshot, Literal["reuse", "fit"]]:
  """Fits/reuses one continuous Phase-II BandInvMF strategy."""
  if not 1 <= request.switch_step < request.horizon:
    raise ValueError("switch_step must lie in [1, horizon)")
  phase_horizon = request.horizon - request.switch_step
  if request.bandwidth > phase_horizon:
    raise ValueError("strategy.bandwidth must not exceed the Phase-II horizon")
  path = frozen_p_strategy_artifact_path(
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
  initial_fingerprint = _artifact_fingerprint(path)
  with file_lock(path):
    existing = _load_compatible_frozen_p_snapshot_unlocked(path, request)
    if existing is not None and (
        not request.force_refit or existing.sha256 != initial_fingerprint
    ):
      return existing, "reuse"
    workload_matrix = np.abs(np.asarray(frozen_p_time_workload(
        phase_horizon,
        tau=request.switch_step,
        beta1=request.beta1,
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
    )))
    fitted = fit_strategy(
        phase_horizon,
        min(request.bandwidth, phase_horizon),
        min(request.min_sep, phase_horizon),
        max_participations=request.max_participations,
        workload_matrix=workload_matrix,
        max_optimizer_steps=request.max_optimizer_steps,
        reduction=request.reduction,
    )
    with atomic_temporary_path(path) as temporary:
      save_bandinv_strategy(
          temporary,
          fitted,
          reduction=request.reduction,
          workload_type="frozen-p-continuous",
          momentum=None,
          learning_rate=request.learning_rate,
          weight_decay=request.weight_decay,
          max_optimizer_steps=request.max_optimizer_steps,
      )
      if _load_compatible_frozen_p_snapshot_unlocked(temporary, request) is None:
        raise ValueError("fitted frozen-p strategy artifact failed validation")
      atomic_replace(temporary, path)
    snapshot = _load_compatible_frozen_p_snapshot_unlocked(path, request)
    if snapshot is None:
      raise ValueError("published frozen-p strategy artifact failed validation")
    return snapshot, "fit"


__all__ = [
    "BandInvMFFitRequest",
    "PrefixSumBandInvMFFitRequest",
    "FrozenPBandInvMFFitRequest",
    "LoadedStrategySnapshot",
    "REPOSITORY_ROOT",
    "get_or_fit_strategy",
    "get_or_fit_strategy_snapshot",
    "get_or_fit_prefix_sum_strategy_snapshot",
    "load_strategy_snapshot",
    "require_compatible_strategy",
    "require_compatible_strategy_snapshot",
    "require_compatible_prefix_sum_strategy_snapshot",
    "strategy_artifact_path",
    "prefix_sum_strategy_artifact_path",
    "frozen_p_strategy_artifact_path",
    "get_or_fit_frozen_p_strategy_snapshot",
    "require_compatible_frozen_p_strategy_snapshot",
]
