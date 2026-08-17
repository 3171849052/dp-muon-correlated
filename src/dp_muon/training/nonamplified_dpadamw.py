"""Non-amplified IID DP-AdamW: one private gradient, all-parameter AdamW."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.privacy import (
    PrivacyCalibration,
    make_clipped_gradient_query,
    sample_iid_gaussian_noise,
)


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedDPAdamWState:
  """Checkpointable parameters, one Optax AdamW state, and DP RNG."""

  params: PyTree
  optimizer_state: Any
  rng_key: jax.Array
  step: jax.Array

  def tree_flatten(self):
    return (self.params, self.optimizer_state, self.rng_key, self.step), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[PyTree, Any, jax.Array, jax.Array]):
    del aux_data
    params, optimizer_state, rng_key, step = children
    return cls(params=params, optimizer_state=optimizer_state, rng_key=rng_key, step=step)


def _finite_scalar(value: object, name: str, *, positive: bool = False,
                   nonnegative: bool = False) -> float:
  array = np.asarray(value)
  if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
    raise ValueError(f"{name} must be a finite scalar")
  result = float(array)
  if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
    raise ValueError(f"{name} must be a valid finite scalar")
  return result


def make_nonamplified_dpadamw_optimizer(
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
  """Builds a plain Optax AdamW transformation for all parameters."""
  _finite_scalar(learning_rate, "learning_rate", positive=True)
  for value, name in ((beta1, "beta1"), (beta2, "beta2")):
    _finite_scalar(value, name, nonnegative=True)
    if not 0 <= value < 1:
      raise ValueError(f"{name} must be in [0, 1)")
  _finite_scalar(eps, "eps", positive=True)
  _finite_scalar(weight_decay, "weight_decay", nonnegative=True)
  return optax.adamw(
      learning_rate=learning_rate,
      b1=beta1,
      b2=beta2,
      eps=eps,
      weight_decay=weight_decay,
  )


def init_nonamplified_dpadamw_state(
    params: PyTree,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> NonAmplifiedDPAdamWState:
  """Initializes the Optax AdamW optimizer state."""
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax GradientTransformation")
  return NonAmplifiedDPAdamWState(
      params=params, optimizer_state=optimizer.init(params), rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
  )


# Kept as a local seam for tests while implemented centrally in privacy.
_sample_iid_gaussian_noise = sample_iid_gaussian_noise


def make_nonamplified_dpadamw_train_step(
    loss_fn: Callable[..., Any],
    calibration: PrivacyCalibration,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    microbatch_size: int | None = None,
) -> tuple[
    Callable[[NonAmplifiedDPAdamWState, Any], NonAmplifiedDPAdamWState],
    optax.GradientTransformation,
]:
  """Builds ``clip -> one full-tree IID noise -> AdamW -> apply_updates``.

  The returned optimizer is required only to initialize a state.  No parameter
  group is clipped, calibrated, or noised separately: the optimizer receives
  exactly the same already-private PyTree.
  """
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  if not isinstance(calibration, PrivacyCalibration):
    raise TypeError("calibration must be a PrivacyCalibration")
  _finite_scalar(calibration.iid_noise_std, "calibration.iid_noise_std", nonnegative=True)
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay,
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )

  def train_step(
      state: NonAmplifiedDPAdamWState, batch: Any
  ) -> NonAmplifiedDPAdamWState:
    if not isinstance(state, NonAmplifiedDPAdamWState):
      raise TypeError("state must be a NonAmplifiedDPAdamWState")
    clipped_grad = clipped_query(state.params, batch)
    # This is the unique privacy mechanism invocation for this logical batch.
    noise, new_key = _sample_iid_gaussian_noise(
        state.rng_key, clipped_grad, jnp.asarray(calibration.iid_noise_std)
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation, clipped_grad, noise
    )
    updates, optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    return NonAmplifiedDPAdamWState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=optimizer_state,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    )

  return train_step, optimizer


__all__ = [
    "NonAmplifiedDPAdamWState",
    "init_nonamplified_dpadamw_state",
    "make_nonamplified_dpadamw_optimizer",
    "make_nonamplified_dpadamw_train_step",
]