"""Standard SGD with heavy-ball momentum for arbitrary JAX PyTrees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SGDMomentumState:
  """Velocity and checkpoint metadata for SGD Momentum."""

  velocity: PyTree
  step: jax.Array

  def tree_flatten(self):
    return (self.velocity, self.step), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[PyTree, jax.Array]):
    velocity, step = children
    return cls(velocity=velocity, step=step)


def _require_floating_tree(tree: PyTree, name: str) -> None:
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    raise ValueError(f"{name} must contain at least one floating array leaf")
  for leaf in leaves:
    if not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
      raise ValueError(f"{name} leaves must be floating arrays")


def _validated_momentum(momentum: float | jax.Array) -> float | jax.Array:
  value = jnp.asarray(momentum)
  if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.number):
    raise ValueError("momentum must be a finite scalar in [0, 1)")
  if isinstance(value, jax.core.Tracer):
    return value
  result = float(value)
  if not bool(jnp.isfinite(value)) or not 0.0 <= result < 1.0:
    raise ValueError("momentum must be a finite scalar in [0, 1)")
  return result


def init_sgd_momentum_state(template: PyTree) -> SGDMomentumState:
  """Initializes zero velocity with the structure and dtypes of ``template``."""
  _require_floating_tree(template, "template")
  return SGDMomentumState(
      velocity=jax.tree_util.tree_map(
          lambda leaf: jnp.zeros_like(jnp.asarray(leaf)), template
      ),
      step=jnp.array(0, dtype=jnp.int32),
  )


def sgd_momentum_step(
    state: SGDMomentumState, gradient: PyTree, momentum: float | jax.Array
) -> tuple[PyTree, SGDMomentumState]:
  """Advances ``v_t = beta * v_(t-1) + g_t`` and returns ``v_t``.

  The returned velocity is the unscaled update direction; callers apply their
  learning rate as ``params - learning_rate * velocity``.  At momentum zero,
  the returned direction is exactly the input gradient, i.e. ordinary SGD.
  """
  if not isinstance(state, SGDMomentumState):
    raise TypeError("state must be an SGDMomentumState")
  beta = _validated_momentum(momentum)
  _require_floating_tree(state.velocity, "state.velocity")
  _require_floating_tree(gradient, "gradient")
  if jax.tree_util.tree_structure(state.velocity) != jax.tree_util.tree_structure(gradient):
    raise ValueError("gradient and state.velocity must have identical PyTree structure")
  step = jnp.asarray(state.step)
  if step.shape != () or not jnp.issubdtype(step.dtype, jnp.integer):
    raise ValueError("state.step must be an integer scalar")

  def advance(old_velocity: jax.Array, gradient_leaf: jax.Array) -> jax.Array:
    old = jnp.asarray(old_velocity)
    grad = jnp.asarray(gradient_leaf)
    if old.shape != grad.shape:
      raise ValueError("gradient leaf shapes must match state velocity leaf shapes")
    if old.dtype != grad.dtype:
      raise ValueError("gradient leaf dtypes must match state velocity leaf dtypes")
    if not isinstance(beta, jax.core.Tracer) and beta == 0.0:
      return grad
    return (jnp.asarray(beta, dtype=old.dtype) * old + grad).astype(old.dtype)

  velocity = jax.tree_util.tree_map(advance, state.velocity, gradient)
  return velocity, SGDMomentumState(
      velocity=velocity,
      step=step + jnp.array(1, dtype=step.dtype),
  )


__all__ = [
    "SGDMomentumState",
    "init_sgd_momentum_state",
    "sgd_momentum_step",
]
