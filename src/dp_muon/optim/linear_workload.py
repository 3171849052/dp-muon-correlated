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


def decayed_prefix_sum_workload_coef(
    horizon: int, learning_rate: float, weight_decay: float
) -> jax.Array:
  """Returns ``P_rho`` coefficients for decoupled weight decay.

  Here ``rho = 1 - learning_rate * weight_decay``.  The coefficient at lag
  ``k`` is ``rho**k``; in particular zero decay exactly recovers prefix-sum.
  """
  if not isinstance(horizon, Integral) or horizon < 1:
    raise ValueError("horizon must be a positive integer")
  learning_rate = _validated_scalar(
      learning_rate, "learning_rate", lower=0.0, strict_lower=True
  )
  weight_decay = _validated_scalar(weight_decay, "weight_decay", lower=0.0)
  if weight_decay == 0.0:
    return jnp.ones((int(horizon),))
  rho = 1.0 - learning_rate * weight_decay
  return jnp.asarray(rho) ** jnp.arange(int(horizon))


def adam_first_moment_workload_matrix(
    horizon: int,
    beta1: float,
    learning_rate: float,
    weight_decay: float,
) -> jax.Array:
  """Returns the exact lower-triangular Adam first-moment workload.

  Rows use zero-based Adam steps.  The first-moment kernel includes bias
  correction at the query step, and decoupled weight decay is applied after
  the moment update, so the result is ``learning_rate * P_rho @ H^m``.
  """
  horizon, beta = _validate_configuration(horizon, beta1)
  learning_rate = _validated_scalar(
      learning_rate, "learning_rate", lower=0.0, strict_lower=True
  )
  weight_decay = _validated_scalar(weight_decay, "weight_decay", lower=0.0)
  index = jnp.arange(horizon)
  row = index[:, None]
  column = index[None, :]
  lag = row - column
  causal = row >= column
  beta_array = jnp.asarray(beta)
  rho = jnp.asarray(1.0 - learning_rate * weight_decay, dtype=beta_array.dtype)
  moment = jnp.where(
      causal,
      (1.0 - beta_array) * beta_array**lag / (1.0 - beta_array ** (row + 1)),
      0.0,
  )
  decay = jnp.where(causal, rho**lag, 0.0)
  return jnp.asarray(learning_rate, dtype=moment.dtype) * (decay @ moment)


def fixed_lr_nesterov_decayed_trajectory_workload_coef(
    horizon: int, momentum: float, learning_rate: float, weight_decay: float
) -> jax.Array:
  """Returns ``eta P_rho H_beta^Nes`` Toeplitz coefficients.

  This remains the naive Muon workload: it models Nesterov momentum and
  decoupled weight decay only, leaving Muon's nonlinear ``Q`` out of scope.
  """
  horizon, beta = _validate_configuration(horizon, momentum)
  learning_rate = _validated_scalar(
      learning_rate, "learning_rate", lower=0.0, strict_lower=True
  )
  weight_decay = _validated_scalar(weight_decay, "weight_decay", lower=0.0)
  if weight_decay == 0.0:
    return fixed_lr_nesterov_trajectory_workload_coef(
        horizon, beta, learning_rate
    )
  h = nesterov_kernel_coef(horizon, beta)
  rho = jnp.asarray(1.0 - learning_rate * weight_decay, dtype=h.dtype)
  powers = rho ** jnp.arange(horizon, dtype=h.dtype)
  return jnp.asarray(learning_rate, dtype=h.dtype) * jnp.convolve(h, powers)[:horizon]


__all__ = [
    "adam_first_moment_workload_matrix",
    "decayed_prefix_sum_workload_coef",
    "fixed_lr_nesterov_decayed_trajectory_workload_coef",
    "fixed_lr_nesterov_trajectory_workload_coef",
    "nesterov_kernel_coef",
]
