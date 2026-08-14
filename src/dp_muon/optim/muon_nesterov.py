"""The linear momentum-plus-Nesterov portion of Muon.

This module intentionally stops at the post-Nesterov, pre-Q update.  It does
not apply a learning rate or any of Muon's matrix-specific operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MuonNesterovState:
  """Streaming state for EMA momentum followed by Nesterov lookahead."""

  momentum: PyTree
  step: jax.Array

  def tree_flatten(self):
    return (self.momentum, self.step), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[PyTree, jax.Array]):
    momentum, step = children
    return cls(momentum=momentum, step=step)


def _require_floating_tree(tree: PyTree, name: str) -> list[jax.Array]:
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    raise ValueError(f"{name} must contain at least one floating array leaf")
  arrays = []
  for leaf in leaves:
    array = jnp.asarray(leaf)
    if not jnp.issubdtype(array.dtype, jnp.floating):
      raise ValueError(f"{name} leaves must be floating arrays")
    arrays.append(array)
  return arrays


def _validated_momentum(momentum: float | jax.Array) -> float | jax.Array:
  value = jnp.asarray(momentum)
  if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.number):
    raise ValueError("momentum must be a finite scalar in [0, 1)")
  if isinstance(value, jax.core.Tracer):
    return value
  value_float = float(value)
  if not bool(jnp.isfinite(value)) or not 0.0 <= value_float < 1.0:
    raise ValueError("momentum must be a finite scalar in [0, 1)")
  return value_float


def init_muon_nesterov_state(template: PyTree) -> MuonNesterovState:
  """Initializes zero EMA momentum for a floating gradient PyTree."""
  _require_floating_tree(template, "template")
  return MuonNesterovState(
      momentum=jax.tree_util.tree_map(lambda leaf: jnp.zeros_like(jnp.asarray(leaf)), template),
      step=jnp.array(0, dtype=jnp.int32),
  )


def muon_nesterov_step(
    state: MuonNesterovState, gradient: PyTree, momentum: float | jax.Array
) -> tuple[PyTree, MuonNesterovState]:
  """Returns ``U_t`` and advances ``M_t = beta M_(t-1) + (1-beta) G_t``.

  ``state.step`` is checkpoint metadata only.  The returned update is exactly
  ``U_t = (1-beta) G_t + beta M_t`` and has no learning-rate scaling.
  """
  if not isinstance(state, MuonNesterovState):
    raise TypeError("state must be a MuonNesterovState")
  beta = _validated_momentum(momentum)
  _require_floating_tree(state.momentum, "state.momentum")
  _require_floating_tree(gradient, "gradient")
  if jax.tree_util.tree_structure(state.momentum) != jax.tree_util.tree_structure(gradient):
    raise ValueError("gradient and state.momentum must have identical PyTree structure")
  step = jnp.asarray(state.step)
  if step.shape != () or not jnp.issubdtype(step.dtype, jnp.integer):
    raise ValueError("state.step must be an integer scalar")

  def advance_momentum(old_momentum: jax.Array, gradient_leaf: jax.Array):
    old = jnp.asarray(old_momentum)
    grad = jnp.asarray(gradient_leaf)
    if old.shape != grad.shape:
      raise ValueError("gradient leaf shapes must match state momentum leaf shapes")
    if old.dtype != grad.dtype:
      raise ValueError("gradient leaf dtypes must match state momentum leaf dtypes")
    leaf_beta = jnp.asarray(beta, dtype=old.dtype)
    new_momentum = leaf_beta * old + (1.0 - leaf_beta) * grad
    return new_momentum.astype(old.dtype)

  new_momentum = jax.tree_util.tree_map(advance_momentum, state.momentum, gradient)

  def compute_update(gradient_leaf: jax.Array, momentum_leaf: jax.Array):
    grad = jnp.asarray(gradient_leaf)
    new = jnp.asarray(momentum_leaf)
    leaf_beta = jnp.asarray(beta, dtype=grad.dtype)
    return ((1.0 - leaf_beta) * grad + leaf_beta * new).astype(grad.dtype)

  update = jax.tree_util.tree_map(compute_update, gradient, new_momentum)
  return update, MuonNesterovState(
      momentum=new_momentum,
      step=step + jnp.array(1, dtype=step.dtype),
  )


__all__ = [
    "MuonNesterovState",
    "init_muon_nesterov_state",
    "muon_nesterov_step",
]
