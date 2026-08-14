"""Public linear workloads induced by the fixed Muon-Nesterov baseline."""

from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp


def _validated_scalar(value: float, name: str, *, lower: float, strict_lower: bool = False) -> float:
  array = jnp.asarray(value)
  if array.ndim != 0 or not jnp.issubdtype(array.dtype, jnp.number):
    raise ValueError(f"{name} must be a finite scalar")
  if isinstance(array, jax.core.Tracer):
    raise ValueError(f"{name} must be a concrete finite scalar configuration")
  result = float(array)
  if not bool(jnp.isfinite(array)) or (result <= lower if strict_lower else result < lower):
    comparison = ">" if strict_lower else ">="
    raise ValueError(f"{name} must be finite and {comparison} {lower}")
  return result


def _validate_configuration(horizon: int, momentum: float) -> tuple[int, float]:
  if not isinstance(horizon, Integral) or horizon < 1:
    raise ValueError("horizon must be a positive integer")
  beta = _validated_scalar(momentum, "momentum", lower=0.0)
  if beta >= 1.0:
    raise ValueError("momentum must be finite and in [0, 1)")
  return int(horizon), beta


def nesterov_kernel_coef(horizon: int, momentum: float) -> jax.Array:
  """Returns coefficients of ``H_beta^Nes`` for EMA-then-Nesterov Muon."""
  horizon, beta = _validate_configuration(horizon, momentum)
  index = jnp.arange(horizon)
  beta_array = jnp.asarray(beta)
  return jnp.where(
      index == 0,
      1.0 - beta_array**2,
      (1.0 - beta_array) * beta_array ** (index + 1),
  )


def fixed_lr_nesterov_trajectory_workload_coef(
    horizon: int, momentum: float, learning_rate: float
) -> jax.Array:
  """Returns coefficients of the fixed-LR trajectory workload ``eta P H``."""
  horizon, beta = _validate_configuration(horizon, momentum)
  learning_rate = _validated_scalar(
      learning_rate, "learning_rate", lower=0.0, strict_lower=True
  )
  h = nesterov_kernel_coef(horizon, beta)
  return jnp.asarray(learning_rate, dtype=h.dtype) * jnp.cumsum(h)


__all__ = [
    "fixed_lr_nesterov_trajectory_workload_coef",
    "nesterov_kernel_coef",
]
