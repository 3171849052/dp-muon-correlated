"""Explicit Scale-Then-Privatize AdamW optimizer state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class STPAdamWState:
  """Checkpointable full AdamW state used by the STP query."""

  count: jax.Array
  m: PyTree
  v: PyTree

  def tree_flatten(self):
    return (self.count, self.m, self.v), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]):
    del aux_data
    return cls(*children)


def _finite_scalar(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
  array = np.asarray(value)
  if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
    raise ValueError(f"{name} must be a finite scalar")
  result = float(array)
  if (
      not math.isfinite(result)
      or (positive and result <= 0)
      or (nonnegative and result < 0)
  ):
    raise ValueError(f"{name} must be a valid finite scalar")
  return result


def _validate_state(state: STPAdamWState) -> None:
  if not isinstance(state, STPAdamWState):
    raise TypeError("state must be an STPAdamWState")
  count = jnp.asarray(state.count)
  if count.shape != () or not jnp.issubdtype(count.dtype, jnp.integer):
    raise ValueError("state.count must be an integer scalar")
  if not isinstance(count, jax.core.Tracer) and not bool(count >= 0):
    raise ValueError("state.count must be non-negative")
  if jax.tree_util.tree_structure(state.m) != jax.tree_util.tree_structure(state.v):
    raise ValueError("state.m and state.v must have identical PyTree structure")


@dataclass(frozen=True)
class STPAdamW:
  """Functional AdamW with an explicit previous-private-V preconditioner.

  ``scale(state)`` only reads the already-private Adam state at the start of
  the step.  ``update`` consumes the unscaled private gradient and then
  updates the complete ``m``/``v`` state with standard AdamW recurrences.
  """

  learning_rate: float
  beta1: float = 0.9
  beta2: float = 0.999
  eps: float = 1e-8
  scale_eps: float = 1e-8
  weight_decay: float = 0.0

  def __post_init__(self) -> None:
    _finite_scalar(self.learning_rate, "learning_rate", positive=True)
    for value, name in ((self.beta1, "beta1"), (self.beta2, "beta2")):
      value = _finite_scalar(value, name, nonnegative=True)
      if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    _finite_scalar(self.eps, "eps", positive=True)
    _finite_scalar(self.scale_eps, "scale_eps", positive=True)
    _finite_scalar(self.weight_decay, "weight_decay", nonnegative=True)

  def init(self, params: PyTree) -> STPAdamWState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return STPAdamWState(
        count=jnp.array(0, dtype=jnp.int32),
        m=zeros,
        v=zeros,
    )

  def previous_v_hat(self, state: STPAdamWState) -> PyTree:
    """Returns ``v_{t-1} / (1 - beta2^(t-1))``, with ``v_hat_0 == 0``."""
    _validate_state(state)
    count = state.count

    def correct(value: jax.Array) -> jax.Array:
      beta2 = jnp.asarray(self.beta2, dtype=value.dtype)
      denominator = jnp.where(
          count == 0,
          jnp.ones((), dtype=value.dtype),
          1.0 - beta2 ** count,
      )
      return jnp.where(count == 0, jnp.zeros_like(value), value / denominator)

    return jax.tree_util.tree_map(correct, state.v)

  def scale(self, state: STPAdamWState) -> PyTree:
    """Returns the elementwise STP scale computed from the previous state."""
    v_hat_prev = self.previous_v_hat(state)
    return jax.tree_util.tree_map(
        lambda value: 1.0 / (jnp.sqrt(value) + self.scale_eps), v_hat_prev
    )

  def update(
      self,
      gradients: PyTree,
      state: STPAdamWState,
      params: PyTree,
      *,
      learning_rate: float | jax.Array | None = None,
  ) -> tuple[PyTree, STPAdamWState]:
    """Updates ``m``, ``v`` and returns decoupled-AdamW parameter updates."""
    _validate_state(state)
    count = state.count + jnp.array(1, dtype=state.count.dtype)
    m = jax.tree_util.tree_map(
        lambda previous, gradient: self.beta1 * previous
        + (1.0 - self.beta1) * gradient,
        state.m,
        gradients,
    )
    v = jax.tree_util.tree_map(
        lambda previous, gradient: self.beta2 * previous
        + (1.0 - self.beta2) * gradient**2,
        state.v,
        gradients,
    )
    correction1 = 1.0 - jnp.asarray(self.beta1) ** count
    correction2 = 1.0 - jnp.asarray(self.beta2) ** count
    rate = jnp.asarray(
        self.learning_rate if learning_rate is None else learning_rate
    )
    updates = jax.tree_util.tree_map(
        lambda first, second, parameter: -rate * (
            (first / correction1)
            / (jnp.sqrt(second / correction2) + self.eps)
            + self.weight_decay * parameter
        ),
        m,
        v,
        params,
    )
    return updates, STPAdamWState(count=count, m=m, v=v)


__all__ = ["STPAdamW", "STPAdamWState"]
