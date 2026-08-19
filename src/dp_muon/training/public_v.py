"""Public-only Adam second-moment estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PublicVState:
  """Uncorrected public second moment and its independent update counter."""

  v: PyTree
  t_v: jax.Array

  def tree_flatten(self):
    return (self.v, self.t_v), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]):
    del aux_data
    return cls(*children)


@dataclass(frozen=True)
class PublicVEstimator:
  """Updates public V without owning model parameters or optimizer state."""

  loss_fn: Callable[[PyTree, Any], jax.Array]
  beta2: float = 0.999
  eps: float = 1e-8

  def __post_init__(self) -> None:
    if not callable(self.loss_fn):
      raise TypeError("loss_fn must be callable")
    if not 0.0 <= self.beta2 < 1.0:
      raise ValueError("beta2 must be in [0, 1)")
    if not self.eps > 0:
      raise ValueError("eps must be positive")

  def init(self, params: PyTree) -> PublicVState:
    return PublicVState(
        v=jax.tree_util.tree_map(jnp.zeros_like, params),
        t_v=jnp.array(0, dtype=jnp.int32),
    )

  def update(self, state: PublicVState, params: PyTree, public_batch: Any) -> PublicVState:
    """Consumes exactly one public batch and increments only ``t_v``."""
    gradient = jax.grad(self.loss_fn)(params, public_batch)
    v = jax.tree_util.tree_map(
        lambda old, value: self.beta2 * old + (1.0 - self.beta2) * jnp.square(value),
        state.v,
        gradient,
    )
    return PublicVState(v=v, t_v=state.t_v + jnp.array(1, dtype=state.t_v.dtype))

  def update_batches(
      self, state: PublicVState, params: PyTree, public_batches: Iterable[Any]
  ) -> PublicVState:
    for batch in public_batches:
      state = self.update(state, params, batch)
    return state

  def bias_corrected_v(self, state: PublicVState) -> PyTree:
    if not isinstance(state.t_v, jax.core.Tracer) and int(state.t_v) < 1:
      raise ValueError("at least one public V update is required")
    correction = 1.0 - jnp.asarray(self.beta2) ** state.t_v
    return jax.tree_util.tree_map(lambda value: value / correction, state.v)

  def preconditioner(self, state: PublicVState) -> PyTree:
    return jax.tree_util.tree_map(
        lambda value: 1.0 / (jnp.sqrt(value) + self.eps),
        self.bias_corrected_v(state),
    )


def public_preconditioner_rms(v_hat: PyTree, eps: float) -> jax.Array:
  """Returns the exact parameter-axis RMS scale for a shared temporal strategy.

  With V frozen, every parameter coordinate has the same temporal workload up
  to a constant diagonal scale.  Mean squared error therefore factors into
  this RMS scale times the existing two-dimensional temporal objective.
  """
  leaves = jax.tree_util.tree_leaves(v_hat)
  if not leaves:
    raise ValueError("v_hat must have at least one leaf")
  squared_sum = sum(
      jnp.sum(jnp.square(1.0 / (jnp.sqrt(value) + eps))) for value in leaves
  )
  return jnp.sqrt(squared_sum / sum(value.size for value in leaves))


__all__ = ["PublicVEstimator", "PublicVState", "public_preconditioner_rms"]
