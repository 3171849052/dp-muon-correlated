"""AdamW with an externally estimated, segment-frozen second moment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PublicVAdamWState:
  """Private first moment/count and the currently frozen public V."""

  count: jax.Array
  mu: PyTree
  public_v_hat: PyTree
  public_v_set: jax.Array

  def tree_flatten(self):
    return (self.count, self.mu, self.public_v_hat, self.public_v_set), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]):
    del aux_data
    return cls(*children)


def _validate_matching_tree(params: PyTree, v_hat: PyTree) -> None:
  if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(v_hat):
    raise ValueError("public v_hat must have the same tree structure as parameters")
  for parameter, value in zip(
      jax.tree_util.tree_leaves(params),
      jax.tree_util.tree_leaves(v_hat),
      strict=True,
  ):
    if parameter.shape != value.shape:
      raise ValueError("public v_hat leaves must match parameter shapes")
    if parameter.dtype != value.dtype:
      raise ValueError("public v_hat leaves must match parameter dtypes")


@dataclass(frozen=True)
class PublicVAdamW:
  """Functional Frozen-(V) AdamW optimizer.

  Private gradients update only ``mu`` and ``count``.  ``set_public_v`` is the
  sole operation that replaces V and it preserves both private state fields.
  """

  learning_rate: float
  beta1: float = 0.9
  eps: float = 1e-8
  weight_decay: float = 0.0

  def __post_init__(self) -> None:
    if not self.learning_rate > 0:
      raise ValueError("learning_rate must be positive")
    if not 0.0 <= self.beta1 < 1.0:
      raise ValueError("beta1 must be in [0, 1)")
    if not self.eps > 0:
      raise ValueError("eps must be positive")
    if self.weight_decay < 0:
      raise ValueError("weight_decay must be non-negative")

  def init(self, params: PyTree) -> PublicVAdamWState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return PublicVAdamWState(
        count=jnp.array(0, dtype=jnp.int32),
        mu=zeros,
        public_v_hat=zeros,
        public_v_set=jnp.array(False),
    )

  def set_public_v(
      self, state: PublicVAdamWState, v_hat: PyTree, params: PyTree
  ) -> PublicVAdamWState:
    """Injects bias-corrected public V without resetting private momentum."""
    if not isinstance(state, PublicVAdamWState):
      raise TypeError("state must be a PublicVAdamWState")
    _validate_matching_tree(params, v_hat)
    for value in jax.tree_util.tree_leaves(v_hat):
      if not isinstance(value, jax.core.Tracer) and not bool(
          jnp.all(jnp.isfinite(value) & (value >= 0))
      ):
        raise ValueError("public v_hat must be finite and non-negative")
    return PublicVAdamWState(
        count=state.count,
        mu=state.mu,
        public_v_hat=v_hat,
        public_v_set=jnp.array(True),
    )

  def update(
      self,
      gradients: PyTree,
      state: PublicVAdamWState,
      params: PyTree,
      *,
      learning_rate: float | jax.Array | None = None,
  ) -> tuple[PyTree, PublicVAdamWState]:
    """Updates first moment and parameters while leaving V unchanged."""
    if not isinstance(state, PublicVAdamWState):
      raise TypeError("state must be a PublicVAdamWState")
    if not isinstance(state.public_v_set, jax.core.Tracer) and not bool(state.public_v_set):
      raise ValueError("set_public_v must be called before a private update")
    count = state.count + jnp.array(1, dtype=state.count.dtype)
    mu = jax.tree_util.tree_map(
        lambda moment, gradient: self.beta1 * moment + (1.0 - self.beta1) * gradient,
        state.mu,
        gradients,
    )
    correction = 1.0 - jnp.asarray(self.beta1) ** count
    rate = jnp.asarray(self.learning_rate if learning_rate is None else learning_rate)
    updates = jax.tree_util.tree_map(
        lambda moment, v_hat, parameter: -rate * (
            (moment / correction) / (jnp.sqrt(v_hat) + self.eps)
            + self.weight_decay * parameter
        ),
        mu,
        state.public_v_hat,
        params,
    )
    return updates, PublicVAdamWState(
        count=count,
        mu=mu,
        public_v_hat=state.public_v_hat,
        public_v_set=state.public_v_set,
    )


__all__ = ["PublicVAdamW", "PublicVAdamWState"]
