"""Public BandInvMF strategy construction for Experiment 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dp_muon.bandinvmf import (
    BandInvMFStrategy,
    fit_bandinv_strategy,
    load_bandinv_strategy,
    load_bandinv_strategy_metadata,
    save_bandinv_strategy,
)
from dp_muon.optim import adam_first_moment_workload_matrix, decayed_prefix_sum_workload_coef


DECAYED_PREFIX = "decayed-prefix"
ADAM_M_AWARE = "adam-m-aware"


@dataclass(frozen=True)
class StrategySpec:
  """All public fitting inputs for one Exp2 covariance strategy."""

  name: str
  horizon: int
  bandwidth: int
  min_sep: int
  max_participations: int
  learning_rate: float
  beta1: float
  weight_decay: float
  reduction: str = "mean"
  max_optimizer_steps: int = 1000

  def __post_init__(self) -> None:
    if self.horizon < 1 or self.bandwidth < 1 or self.bandwidth > self.horizon:
      raise ValueError("strategy horizon and bandwidth are invalid")
    if self.min_sep < 1 or self.max_participations < 1:
      raise ValueError("strategy min_sep and max_participations must be positive")


def workload_for(spec: StrategySpec):
  """Returns the workload representation mandated by ``spec.name``."""
  if spec.name == DECAYED_PREFIX:
    return {"workload_coef": decayed_prefix_sum_workload_coef(
        spec.horizon, spec.learning_rate, spec.weight_decay
    )}
  if spec.name == ADAM_M_AWARE:
    return {"workload_matrix": adam_first_moment_workload_matrix(
        spec.horizon, spec.beta1, spec.learning_rate, spec.weight_decay
    )}
  raise ValueError(f"unknown Exp2 strategy: {spec.name!r}")


def fit_strategy(spec: StrategySpec) -> BandInvMFStrategy:
  """Fits either the Toeplitz naive or general causal m-aware workload."""
  return fit_bandinv_strategy(
      spec.horizon, spec.bandwidth, spec.min_sep,
      max_participations=spec.max_participations,
      max_optimizer_steps=spec.max_optimizer_steps,
      reduction=spec.reduction, **workload_for(spec),
  )


def save_strategy(path: str | Path, strategy: BandInvMFStrategy, spec: StrategySpec) -> None:
  """Saves a strategy with enough metadata to reject a mismatched replay."""
  save_bandinv_strategy(
      path, strategy, reduction=spec.reduction, workload_type=spec.name,
      momentum=spec.beta1 if spec.name == ADAM_M_AWARE else None,
      learning_rate=spec.learning_rate, weight_decay=spec.weight_decay,
      max_optimizer_steps=spec.max_optimizer_steps,
  )


def load_or_fit_strategy(
    path: str | Path, spec: StrategySpec, *, force_refit: bool = False
) -> BandInvMFStrategy:
  """Loads a compatible artifact or fits and writes it at ``path``."""
  path = Path(path)
  if path.exists() and not force_refit:
    strategy = load_bandinv_strategy(path)
    metadata = load_bandinv_strategy_metadata(path)
    expected_matrix = spec.name == ADAM_M_AWARE
    def matches(value: float | None, expected: float) -> bool:
      return value is not None and bool(np.isclose(value, expected))

    if (
        strategy.horizon == spec.horizon
        and strategy.bandwidth == spec.bandwidth
        and strategy.min_sep == spec.min_sep
        and strategy.max_participations == spec.max_participations
        and metadata.workload_type == spec.name
        and metadata.reduction == spec.reduction
        and matches(metadata.learning_rate, spec.learning_rate)
        and matches(metadata.weight_decay, spec.weight_decay)
        and metadata.max_optimizer_steps == spec.max_optimizer_steps
        and (strategy.workload_matrix is not None) == expected_matrix
        and (spec.name != ADAM_M_AWARE or matches(metadata.momentum, spec.beta1))
    ):
      return strategy
  strategy = fit_strategy(spec)
  save_strategy(path, strategy, spec)
  return strategy


__all__ = [
    "ADAM_M_AWARE", "DECAYED_PREFIX", "StrategySpec", "fit_strategy",
    "load_or_fit_strategy", "save_strategy", "workload_for",
]
