"""State-preserving frozen-preconditioner AdamW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


PyTree = Any


def _adam_state(value: Any) -> Any:
  """Find Optax's real ``ScaleByAdamState`` inside an optimizer state."""
  if all(hasattr(value, name) for name in ("count", "mu", "nu")):
    return value
  if isinstance(value, (tuple, list)):
    for item in value:
      try:
        return _adam_state(item)
      except TypeError:
        pass
  raise TypeError("optimizer_state does not contain an Optax ScaleByAdamState")


def p_star_from_optax(optimizer_state: Any, *, beta2: float, eps: float) -> PyTree:
  """Computes ``1 / (sqrt(nu / (1-beta2**count)) + eps)`` from Optax state."""
  adam = _adam_state(optimizer_state)
  count = jnp.asarray(adam.count)
  if not isinstance(count, jax.core.Tracer) and int(count) < 1:
    raise ValueError("p_star can only be frozen after an AdamW update")
  correction = 1.0 - jnp.asarray(beta2) ** count
  return jax.tree_util.tree_map(
      lambda nu: 1.0 / (jnp.sqrt(nu / correction) + eps), adam.nu
  )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FrozenPAdamWState:
  """Global Adam count/momentum with immutable second moment and ``p_star``."""

  count: jax.Array
  mu: PyTree
  frozen_nu: PyTree
  p_star: PyTree

  def tree_flatten(self):
    return (self.count, self.mu, self.frozen_nu, self.p_star), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def freeze_optax_adamw(
    optimizer_state: Any, *, beta2: float, eps: float
) -> FrozenPAdamWState:
  """Freezes an updated Optax AdamW state without resetting any Adam fields."""
  adam = _adam_state(optimizer_state)
  return FrozenPAdamWState(
      count=adam.count,
      mu=adam.mu,
      frozen_nu=adam.nu,
      p_star=p_star_from_optax(optimizer_state, beta2=beta2, eps=eps),
  )


@dataclass(frozen=True)
class FrozenPAdamW:
  """Phase-II AdamW recurrence with a fixed coordinate preconditioner."""

  learning_rate: float
  beta1: float = 0.9
  weight_decay: float = 0.0

  def update(
      self, gradients: PyTree, state: FrozenPAdamWState, params: PyTree
  ) -> tuple[PyTree, FrozenPAdamWState]:
    count = state.count + jnp.array(1, dtype=state.count.dtype)
    mu = jax.tree_util.tree_map(
        lambda old, grad: self.beta1 * old + (1.0 - self.beta1) * grad,
        state.mu,
        gradients,
    )
    correction = 1.0 - jnp.asarray(self.beta1) ** count
    updates = jax.tree_util.tree_map(
        lambda moment, p, param: -self.learning_rate * (
            p * moment / correction + self.weight_decay * param
        ),
        mu,
        state.p_star,
        params,
    )
    return updates, FrozenPAdamWState(
        count=count,
        mu=mu,
        frozen_nu=state.frozen_nu,
        p_star=state.p_star,
    )


__all__ = [
    "FrozenPAdamW",
    "FrozenPAdamWState",
    "freeze_optax_adamw",
    "p_star_from_optax",
]
