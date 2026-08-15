"""Loading and validation for public BandInvMF strategy artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .strategy import BandInvMFStrategy


_REQUIRED_FIELDS = (
    "horizon", "bandwidth", "min_sep", "max_participations", "workload_coef",
    "noising_coef", "strategy_coef", "sensitivity_squared", "objective",
)


@dataclass(frozen=True)
class BandInvMFArtifactMetadata:
  """Public fitting metadata kept separate from the mathematical strategy."""

  reduction: Literal["mean", "max", "last"]
  workload_type: str
  momentum: float | None
  learning_rate: float | None
  max_optimizer_steps: int


def _integer_scalar(value: np.ndarray, name: str, *, positive: bool = True) -> int:
  array = np.asarray(value)
  if array.shape != () or not np.issubdtype(array.dtype, np.integer):
    raise ValueError(f"artifact field {name!r} must be an integer scalar")
  result = int(array)
  if positive and result < 1:
    raise ValueError(f"artifact field {name!r} must be positive")
  return result


def _finite_float_array(value: np.ndarray, name: str, shape: tuple[int, ...] | None = None) -> jnp.ndarray:
  array = np.asarray(value)
  if not np.issubdtype(array.dtype, np.floating) or (shape is not None and array.shape != shape):
    expected = "a floating array" if shape is None else f"a floating array with shape {shape}"
    raise ValueError(f"artifact field {name!r} must be {expected}")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"artifact field {name!r} must contain finite values")
  return jnp.asarray(array)


def _string_scalar(value: np.ndarray, name: str) -> str:
  array = np.asarray(value)
  if array.shape != () or array.dtype.kind not in {"U", "S"}:
    raise ValueError(f"artifact field {name!r} must be a string scalar")
  return str(array.item())


def _optional_finite_float_scalar(value: np.ndarray, name: str) -> float | None:
  array = np.asarray(value)
  if array.shape != () or not np.issubdtype(array.dtype, np.floating):
    raise ValueError(f"artifact field {name!r} must be a floating scalar")
  result = float(array)
  if math.isnan(result):
    return None
  if not math.isfinite(result):
    raise ValueError(f"artifact field {name!r} must be finite or NaN")
  return result


def load_bandinv_strategy(path: str | Path) -> BandInvMFStrategy:
  """Loads the ``.npz`` format written by :mod:`scripts.fit_bandinvmf`.

  Validation is intentionally limited to public artifact integrity; runtime
  compatibility with privacy and Nesterov settings remains M6's job.
  """
  source = Path(path)
  try:
    with np.load(source, allow_pickle=False) as archive:
      missing = set(_REQUIRED_FIELDS).difference(archive.files)
      if missing:
        raise ValueError(f"strategy artifact is missing fields: {sorted(missing)}")
      horizon = _integer_scalar(archive["horizon"], "horizon")
      bandwidth = _integer_scalar(archive["bandwidth"], "bandwidth")
      min_sep = _integer_scalar(archive["min_sep"], "min_sep")
      raw_max_participations = _integer_scalar(
          archive["max_participations"], "max_participations", positive=False
      )
      if raw_max_participations == 0 or raw_max_participations < -1:
        raise ValueError("artifact field 'max_participations' must be -1 or positive")
      max_participations = None if raw_max_participations == -1 else raw_max_participations
      if bandwidth > horizon:
        raise ValueError("artifact bandwidth must not exceed horizon")
      workload_coef = _finite_float_array(archive["workload_coef"], "workload_coef", (horizon,))
      noising_coef = _finite_float_array(archive["noising_coef"], "noising_coef", (bandwidth,))
      strategy_coef = _finite_float_array(archive["strategy_coef"], "strategy_coef")
      if strategy_coef.ndim != 1 or strategy_coef.shape[0] != horizon:
        raise ValueError("artifact field 'strategy_coef' must have shape (horizon,)")
      sensitivity_squared = _finite_float_array(archive["sensitivity_squared"], "sensitivity_squared", ())
      objective = _finite_float_array(archive["objective"], "objective", ())
  except OSError as error:
    raise ValueError(f"could not load strategy artifact {source}") from error
  if float(sensitivity_squared) <= 0:
    raise ValueError("artifact field 'sensitivity_squared' must be positive")
  return BandInvMFStrategy(
      horizon=horizon,
      bandwidth=bandwidth,
      min_sep=min_sep,
      max_participations=max_participations,
      workload_coef=workload_coef,
      noising_coef=noising_coef,
      strategy_coef=strategy_coef,
      sensitivity_squared=sensitivity_squared,
      objective=objective,
  )


def load_bandinv_strategy_metadata(path: str | Path) -> BandInvMFArtifactMetadata:
  """Loads non-mathematical metadata required for safe artifact reuse.

  Older artifacts without this provenance are intentionally not cache hits.
  They can still be loaded through :func:`load_bandinv_strategy` for existing
  callers that only need the mathematical factorization.
  """
  source = Path(path)
  required = ("reduction", "workload_type", "momentum", "learning_rate", "max_optimizer_steps")
  try:
    with np.load(source, allow_pickle=False) as archive:
      missing = set(required).difference(archive.files)
      if missing:
        raise ValueError(f"strategy artifact is missing reuse metadata: {sorted(missing)}")
      reduction = _string_scalar(archive["reduction"], "reduction")
      if reduction not in {"mean", "max", "last"}:
        raise ValueError("artifact field 'reduction' must be one of: mean, max, last")
      workload_type = _string_scalar(archive["workload_type"], "workload_type")
      max_optimizer_steps = _integer_scalar(
          archive["max_optimizer_steps"], "max_optimizer_steps"
      )
      momentum = _optional_finite_float_scalar(archive["momentum"], "momentum")
      learning_rate = _optional_finite_float_scalar(
          archive["learning_rate"], "learning_rate"
      )
  except OSError as error:
    raise ValueError(f"could not load strategy artifact {source}") from error
  return BandInvMFArtifactMetadata(
      reduction=reduction,  # type: ignore[arg-type]
      workload_type=workload_type,
      momentum=momentum,
      learning_rate=learning_rate,
      max_optimizer_steps=max_optimizer_steps,
  )


def save_bandinv_strategy(
    path: str | Path,
    strategy: BandInvMFStrategy,
    *,
    reduction: Literal["mean", "max", "last"],
    workload_type: str,
    momentum: float | None,
    learning_rate: float | None,
    max_optimizer_steps: int,
) -> None:
  """Writes a strategy and its cache-compatibility metadata."""
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      output,
      horizon=np.asarray(strategy.horizon),
      bandwidth=np.asarray(strategy.bandwidth),
      min_sep=np.asarray(strategy.min_sep),
      max_participations=np.asarray(
          -1 if strategy.max_participations is None else strategy.max_participations
      ),
      workload_coef=np.asarray(strategy.workload_coef),
      noising_coef=np.asarray(strategy.noising_coef),
      strategy_coef=np.asarray(strategy.strategy_coef),
      sensitivity_squared=np.asarray(strategy.sensitivity_squared),
      objective=np.asarray(strategy.objective),
      reduction=np.asarray(reduction),
      workload_type=np.asarray(workload_type),
      momentum=np.asarray(np.nan if momentum is None else momentum),
      learning_rate=np.asarray(np.nan if learning_rate is None else learning_rate),
      max_optimizer_steps=np.asarray(max_optimizer_steps),
  )


__all__ = [
    "BandInvMFArtifactMetadata",
    "load_bandinv_strategy",
    "load_bandinv_strategy_metadata",
    "save_bandinv_strategy",
]
