"""Loading and validation for public BandInvMF strategy artifacts."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .strategy import BandInvMFStrategy


_REQUIRED_FIELDS = (
    "horizon", "bandwidth", "min_sep", "max_participations", "workload_coef",
    "noising_coef", "strategy_coef", "sensitivity_squared", "objective",
)


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


__all__ = ["load_bandinv_strategy"]
