"""Public linear workloads induced by the fixed Muon-Nesterov baseline."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


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
  """Returns the exact lower-triangular full-horizon AdamW workload.

  Rows use zero-based Adam steps.  The first-moment kernel includes bias
  correction at the query step, and decoupled weight decay is applied after
  the moment update, so the result is ``learning_rate * P_rho @ H^m``.

  This is the temporal workload of the private gradients themselves.  The
  Adam second-moment preconditioner (whether changing or frozen) is a
  parameter-axis operation and is deliberately not represented here.
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


def frozen_p_time_workload(
    horizon: int,
    *,
    tau: int,
    beta1: float,
    learning_rate: float,
    weight_decay: float,
) -> np.ndarray:
  """Returns the complete post-switch frozen-p temporal workload.

  ``horizon`` is the number of Phase-II steps.  The Adam first-moment bias
  correction uses the global count ``tau + step + 1``; no Phase-II restart is
  represented by this matrix.  The parameter-axis ``p_star`` is applied later
  by the frozen optimizer and is intentionally separate from this temporal
  workload.
  """
  if not isinstance(horizon, Integral) or horizon < 1:
    raise ValueError("horizon must be a positive integer")
  if not isinstance(tau, Integral) or tau < 1:
    raise ValueError("tau must be a positive integer")
  beta1 = _validated_scalar(beta1, "beta1", lower=0.0)
  if beta1 >= 1.0:
    raise ValueError("beta1 must be finite and in [0, 1)")
  learning_rate = _validated_scalar(
      learning_rate, "learning_rate", lower=0.0, strict_lower=True
  )
  weight_decay = _validated_scalar(weight_decay, "weight_decay", lower=0.0)

  result = np.zeros((int(horizon), int(horizon)), dtype=np.float64)
  rho = 1.0 - learning_rate * weight_decay
  for source in range(int(horizon)):
    dm = 0.0
    dtheta = 0.0
    for step in range(int(horizon)):
      dm = beta1 * dm + ((1.0 - beta1) if step == source else 0.0)
      dtheta = rho * dtheta - learning_rate * dm / (
          1.0 - beta1 ** (int(tau) + step + 1)
      )
      result[step, source] = dtheta
  return result


def frozen_p_adamw_segment_workload_matrix(
    segment_length: int,
    *,
    beta1: float,
    learning_rate: float,
    weight_decay: float,
    frozen_preconditioner: float = 1.0,
    first_moment_start_step: int = 0,
) -> np.ndarray:
  """Returns a fixed-``P`` AdamW trajectory workload for one segment.

  The scalar ``frozen_preconditioner`` is the parameter-axis coordinate used
  for fitting a shared temporal BandInvMF factor.  The trainer applies the
  actual PyTree-valued ``P`` after the factor has been sampled.  Keeping this
  scalar explicit is important: it makes the workload contract include the
  frozen preconditioner instead of silently reverting to the prefix-sum
  workload used by the older segmented baseline.

  ``first_moment_start_step`` is the global number of Adam first-moment
  updates before this segment.  Consequently the bias correction does not
  restart at a segment boundary.
  """
  if not isinstance(segment_length, Integral) or isinstance(segment_length, bool) or segment_length < 1:
    raise ValueError("segment_length must be a positive integer")
  if not isinstance(first_moment_start_step, Integral) or isinstance(first_moment_start_step, bool) or first_moment_start_step < 0:
    raise ValueError("first_moment_start_step must be a non-negative integer")
  beta1 = _validated_scalar(beta1, "beta1", lower=0.0)
  if beta1 >= 1.0:
    raise ValueError("beta1 must be finite and in [0, 1)")
  learning_rate = _validated_scalar(learning_rate, "learning_rate", lower=0.0, strict_lower=True)
  weight_decay = _validated_scalar(weight_decay, "weight_decay", lower=0.0)
  frozen_preconditioner = _validated_scalar(
      frozen_preconditioner, "frozen_preconditioner", lower=0.0, strict_lower=True
  )

  length = int(segment_length)
  rho = 1.0 - learning_rate * weight_decay
  result = np.zeros((length, length), dtype=np.float64)
  for source in range(length):
    moment = 0.0
    trajectory = 0.0
    for step in range(length):
      moment = beta1 * moment + ((1.0 - beta1) if step == source else 0.0)
      trajectory = rho * trajectory - learning_rate * frozen_preconditioner * moment / (
          1.0 - beta1 ** (int(first_moment_start_step) + step + 1)
      )
      result[step, source] = trajectory
  return result


def shadow_jme_second_moment_endpoint_workload_coef(
    segment_length: int, beta2: float
) -> jax.Array:
  """Returns the endpoint-only beta2 EMA workload ``B_k``.

  Entry ``r`` is exactly ``(1-beta2) * beta2**(L-1-r)``.  This deliberately
  models only the segment-end shadow value, rather than the full second
  moment trajectory.
  """
  if not isinstance(segment_length, Integral) or isinstance(segment_length, bool) or segment_length < 1:
    raise ValueError("segment_length must be a positive integer")
  beta2 = _validated_scalar(beta2, "beta2", lower=0.0)
  if beta2 >= 1.0:
    raise ValueError("beta2 must be finite and in [0, 1)")
  length = int(segment_length)
  return (1.0 - jnp.asarray(beta2)) * jnp.asarray(beta2) ** jnp.arange(
      length - 1, -1, -1
  )


def public_v_adamw_segment_workload_matrix(
    segment_length: int,
    beta1: float,
    learning_rates: float | Sequence[float] | jax.Array,
    weight_decay: float,
    *,
    first_moment_start_step: int,
) -> jax.Array:
  """Returns the parameter-independent temporal workload ``P_lambda D H``.

  ``first_moment_start_step`` keeps Adam's first-moment bias correction global
  across segments.  The frozen public preconditioner acts separately on the
  parameter axis and is not part of this ``segment_length x segment_length``
  temporal workload:

  ``A_time^(k) = P_lambda,k D_k H_k(s_k)``.
  """
  segment_length, beta = _validate_configuration(segment_length, beta1)
  if not isinstance(first_moment_start_step, Integral) or first_moment_start_step < 0:
    raise ValueError("first_moment_start_step must be a non-negative integer")
  weight_decay = _validated_scalar(weight_decay, "weight_decay", lower=0.0)
  rates = jnp.asarray(learning_rates)
  if rates.ndim == 0:
    rates = jnp.full((segment_length,), rates)
  if rates.shape != (segment_length,):
    raise ValueError("learning_rates must be scalar or have shape (segment_length,)")
  if isinstance(rates, jax.core.Tracer) or not bool(
      jnp.all(jnp.isfinite(rates) & (rates > 0))
  ):
    raise ValueError("learning_rates must be finite and positive")

  index = jnp.arange(segment_length)
  row, column = index[:, None], index[None, :]
  causal = row >= column
  lag = row - column
  beta_array = jnp.asarray(beta, dtype=rates.dtype)
  global_row = first_moment_start_step + row
  moment = jnp.where(
      causal,
      (1.0 - beta_array) * beta_array**lag
      / (1.0 - beta_array ** (global_row + 1)),
      0.0,
  )

  # P[t, r] propagates optimizer update r through later decoupled-decay
  # factors rho[r+1] ... rho[t].  The explicit product also handles rho=0.
  rho = 1.0 - rates * weight_decay
  factor_index = index[None, None, :]
  decay_mask = (
      (factor_index > column[:, :, None])
      & (factor_index <= row[:, :, None])
  )
  propagation = jnp.where(
      causal,
      jnp.prod(jnp.where(decay_mask, rho[None, None, :], 1.0), axis=-1),
      0.0,
  )
  return propagation @ (rates[:, None] * moment)


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
    "frozen_p_time_workload",
    "frozen_p_adamw_segment_workload_matrix",
    "shadow_jme_second_moment_endpoint_workload_coef",
    "fixed_lr_nesterov_decayed_trajectory_workload_coef",
    "fixed_lr_nesterov_trajectory_workload_coef",
    "nesterov_kernel_coef",
    "public_v_adamw_segment_workload_matrix",
]
