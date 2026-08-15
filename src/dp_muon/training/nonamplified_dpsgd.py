"""Non-amplified IID Gaussian DP-SGD with standard SGD Momentum."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.optim import (
    SGDMomentumState,
    init_sgd_momentum_state,
    sgd_momentum_step,
)
from dp_muon.privacy import (
    PrivacyCalibration,
    make_clipped_gradient_query,
    sample_iid_gaussian_noise,
)


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedDPSGDState:
  """Checkpointable DP-SGD state; velocity only ever sees noisy gradients."""

  params: PyTree
  momentum_state: SGDMomentumState
  rng_key: jax.Array

  def tree_flatten(self):
    return (self.params, self.momentum_state, self.rng_key), None

  @classmethod
  def tree_unflatten(
      cls, aux_data: None, children: tuple[PyTree, SGDMomentumState, jax.Array]
  ):
    params, momentum_state, rng_key = children
    return cls(params=params, momentum_state=momentum_state, rng_key=rng_key)


def _finite_scalar(value: object, name: str, *, nonnegative: bool = False) -> float:
  array = np.asarray(value)
  if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
    raise ValueError(f"{name} must be a finite scalar")
  result = float(array)
  if not math.isfinite(result) or (nonnegative and result < 0):
    raise ValueError(f"{name} must be a finite{' non-negative' if nonnegative else ''} scalar")
  return result


def _validate_state(state: NonAmplifiedDPSGDState) -> None:
  if not isinstance(state, NonAmplifiedDPSGDState):
    raise TypeError("state must be a NonAmplifiedDPSGDState")
  if not isinstance(state.momentum_state, SGDMomentumState):
    raise TypeError("state.momentum_state must be an SGDMomentumState")


# Backwards-compatible test seam; the implementation now lives in privacy.
_sample_iid_gaussian_noise = sample_iid_gaussian_noise


def init_nonamplified_dpsgd_state(
    params: PyTree, rng_key: jax.Array
) -> NonAmplifiedDPSGDState:
  """Initializes a private SGD Momentum state from model parameters."""
  return NonAmplifiedDPSGDState(
      params=params,
      momentum_state=init_sgd_momentum_state(params),
      rng_key=rng_key,
  )


def make_nonamplified_dpsgd_train_step(
    loss_fn: Callable[..., Any],
    calibration: PrivacyCalibration,
    momentum: float,
    learning_rate: float,
    microbatch_size: int | None = None,
) -> Callable[[NonAmplifiedDPSGDState, Any], NonAmplifiedDPSGDState]:
  """Builds ``clip -> normalize -> IID noise -> Momentum -> SGD``.

  ``calibration.iid_noise_std`` is one per-step IID noise standard deviation;
  it must have been calibrated for the full fixed-cycle transcript separately.
  """
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  if not isinstance(calibration, PrivacyCalibration):
    raise TypeError("calibration must be a PrivacyCalibration")
  _finite_scalar(calibration.iid_noise_std, "calibration.iid_noise_std", nonnegative=True)
  _finite_scalar(learning_rate, "learning_rate")
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )

  def train_step(
      state: NonAmplifiedDPSGDState, batch: Any
  ) -> NonAmplifiedDPSGDState:
    _validate_state(state)
    clean_query = clipped_query(state.params, batch)
    # Tie public closure values to the traced state, keeping direct jitted use
    # of this closure equivalent to eager execution.
    step = state.momentum_state.step
    noise_std = jnp.asarray(calibration.iid_noise_std)
    runtime_noise_std = noise_std + (
        jnp.asarray(step, dtype=noise_std.dtype) * jnp.zeros_like(noise_std)
    )
    momentum_value = jnp.asarray(momentum)
    runtime_momentum = momentum_value + (
        jnp.asarray(step, dtype=momentum_value.dtype) * jnp.zeros_like(momentum_value)
    )
    noise, new_key = _sample_iid_gaussian_noise(
        state.rng_key, clean_query, runtime_noise_std
    )
    noisy_query = jax.tree_util.tree_map(
        lambda query_leaf, noise_leaf: query_leaf + noise_leaf,
        clean_query,
        noise,
    )
    velocity, new_momentum_state = sgd_momentum_step(
        state.momentum_state, noisy_query, runtime_momentum
    )
    new_params = jax.tree_util.tree_map(
        lambda parameter, velocity_leaf: parameter - learning_rate * velocity_leaf,
        state.params,
        velocity,
    )
    return NonAmplifiedDPSGDState(
        params=new_params,
        momentum_state=new_momentum_state,
        rng_key=new_key,
    )

  return train_step


__all__ = [
    "NonAmplifiedDPSGDState",
    "init_nonamplified_dpsgd_state",
    "make_nonamplified_dpsgd_train_step",
]
