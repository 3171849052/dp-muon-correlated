"""Experiment 7b optimizer update built on Experiment 7's private query.

The baseline is Experiment 7 verbatim.  The BC branch also executes the
Experiment 7 step to obtain the clipped query, one BandInvMF sample, and all
shadow moments, then replaces only the discarded baseline parameter update by
the paper-form gamma-prime update.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

from dp_muon.bandinvmf import BandInvMFStrategy
from dp_muon.privacy import ParticipationSpec, PrivacyCalibration
from exp7.core import Exp7TrainState, make_exp7_train_step


PyTree = Any
DEFAULT_GAMMA_PRIME_RATIO = 1.0


def phi_infinity(
    strategy: BandInvMFStrategy, iid_noise_std: float | jax.Array
) -> jax.Array:
  """Return sigma^2 times the squared norm of the full C^-1 FIR row."""
  coef = jnp.asarray(strategy.noising_coef)
  if coef.ndim != 1 or coef.shape != (strategy.bandwidth,):
    raise ValueError("strategy.noising_coef must match strategy.bandwidth")
  sigma = jnp.asarray(iid_noise_std, dtype=coef.dtype)
  if sigma.ndim != 0:
    raise ValueError("iid_noise_std must be scalar")
  return jnp.square(sigma) * jnp.sum(jnp.square(coef))


def gamma_prime_from_ratio(
    strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    gamma_prime_ratio: float = DEFAULT_GAMMA_PRIME_RATIO,
) -> tuple[jax.Array, jax.Array]:
  """Return ``(phi_infinity, gamma_prime_ratio * phi_infinity)``."""
  if not math.isfinite(gamma_prime_ratio) or gamma_prime_ratio <= 0:
    raise ValueError("gamma_prime_ratio must be finite and positive")
  phi_inf = phi_infinity(strategy, iid_noise_std)
  gamma_prime = jnp.asarray(gamma_prime_ratio, phi_inf.dtype) * phi_inf
  return phi_inf, gamma_prime


def paper_bc_preconditioner(
    corrected_v: jax.Array, gamma_prime: float | jax.Array
) -> jax.Array:
  """DP-AdamBC paper form: 1 / sqrt(max(corrected_v, gamma_prime))."""
  return 1.0 / jnp.sqrt(jnp.maximum(corrected_v, jnp.asarray(gamma_prime)))


def make_exp7b_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    algorithm: str,
    learning_rate: float,
    gamma_prime: float | jax.Array,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    microbatch_size: int | None = None,
) -> Callable[[Exp7TrainState, Any], Exp7TrainState]:
  """Create the unchanged baseline or paper-form gamma-prime BC step."""
  if algorithm not in ("baseline", "bc"):
    raise ValueError("algorithm must be 'baseline' or 'bc'")
  gamma_value = float(gamma_prime)
  if not math.isfinite(gamma_value) or gamma_value <= 0:
    raise ValueError("gamma_prime must be finite and positive")

  # Using the baseline path here is intentional: it is the exact Exp7 private
  # mechanism and shadow update.  For BC, its candidate params are discarded.
  exp7_step = make_exp7_train_step(
      loss_fn, strategy, calibration, participation_spec,
      algorithm="baseline", learning_rate=learning_rate, beta1=beta1,
      beta2=beta2, eps=eps, weight_decay=weight_decay,
      microbatch_size=microbatch_size,
  )
  if algorithm == "baseline":
    return exp7_step

  gamma = jnp.asarray(gamma_value)

  def step_fn(state: Exp7TrainState, batch: Any) -> Exp7TrainState:
    updated = exp7_step(state, batch)
    t = updated.step
    mhat_private = jax.tree_util.tree_map(
        lambda value: value / (1.0 - beta1 ** t), updated.dp_m
    )
    vhat_private = jax.tree_util.tree_map(
        lambda value: value / (1.0 - beta2 ** t), updated.v11
    )
    phi_t = updated.bias_v / (1.0 - beta2 ** t)
    p_bc = jax.tree_util.tree_map(
        lambda value: paper_bc_preconditioner(value - phi_t, gamma),
        vhat_private,
    )
    params = jax.tree_util.tree_map(
        lambda parameter, moment, preconditioner: (
            (1.0 - learning_rate * weight_decay) * parameter
            - learning_rate * moment * preconditioner
        ),
        state.params, mhat_private, p_bc,
    )
    return replace(updated, params=params)

  return step_fn


__all__ = [
    "DEFAULT_GAMMA_PRIME_RATIO",
    "gamma_prime_from_ratio",
    "make_exp7b_train_step",
    "paper_bc_preconditioner",
    "phi_infinity",
]
