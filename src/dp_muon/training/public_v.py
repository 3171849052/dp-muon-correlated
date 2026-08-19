"""Public batch-gradient second-moment estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp


PyTree = Any


@dataclass(frozen=True)
class PublicVEstimator:
  """Averages squared mean-loss gradients across public batches."""

  loss_fn: Callable[[PyTree, Any], jax.Array]
  eps: float = 1e-8

  def __post_init__(self) -> None:
    if not callable(self.loss_fn):
      raise TypeError("loss_fn must be callable")
    if not self.eps > 0:
      raise ValueError("eps must be positive")

  def squared_batch_gradient(
      self, params: PyTree, public_batch: Any
  ) -> PyTree:
    """Returns the squared gradient of one public batch's mean loss."""
    leaves = jax.tree_util.tree_leaves(public_batch)
    if not leaves:
      raise ValueError("public batch must have at least one leaf")
    if any(value.ndim < 1 for value in leaves):
      raise ValueError("public batch leaves must have a leading example axis")
    batch_size = leaves[0].shape[0]
    if batch_size < 1 or any(value.shape[0] != batch_size for value in leaves):
      raise ValueError("public batch leaves must have the same non-empty leading axis")

    gradient = jax.grad(self.loss_fn)(params, public_batch)
    return jax.tree_util.tree_map(jnp.square, gradient)

  def estimate_with_count(
      self,
      params: PyTree,
      public_batches: Iterable[Any],
      *,
      batch_estimate: Callable[[PyTree, Any], PyTree] | None = None,
  ) -> tuple[PyTree, int]:
    """Returns the equal-batch average and total diagnostic example count."""
    estimate_batch = (
        self.squared_batch_gradient if batch_estimate is None else batch_estimate
    )
    squared_sum = None
    batch_count = 0
    example_count = 0
    for public_batch in public_batches:
      leaves = jax.tree_util.tree_leaves(public_batch)
      if not leaves or leaves[0].ndim < 1:
        raise ValueError("public batch must have a leading example axis")
      squared_gradient = estimate_batch(params, public_batch)
      squared_sum = (
          squared_gradient
          if squared_sum is None
          else jax.tree_util.tree_map(jnp.add, squared_sum, squared_gradient)
      )
      batch_count += 1
      example_count += int(leaves[0].shape[0])
    if squared_sum is None or batch_count < 1:
      raise ValueError("each segment requires at least one public batch")
    return (
        jax.tree_util.tree_map(lambda value: value / batch_count, squared_sum),
        example_count,
    )

  def estimate(self, params: PyTree, public_batches: Iterable[Any]) -> PyTree:
    """Returns ``mean_b(square(grad(mean_loss_on_batch_b)))``."""
    v_hat, _ = self.estimate_with_count(params, public_batches)
    return v_hat

  def preconditioner(self, v_hat: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda value: 1.0 / (jnp.sqrt(value) + self.eps), v_hat
    )


def public_preconditioner_rms(v_hat: PyTree, eps: float) -> jax.Array:
  """Returns the parameter-axis RMS scale of the frozen preconditioner."""
  leaves = jax.tree_util.tree_leaves(v_hat)
  if not leaves:
    raise ValueError("v_hat must have at least one leaf")
  squared_sum = sum(
      jnp.sum(jnp.square(1.0 / (jnp.sqrt(value) + eps))) for value in leaves
  )
  return jnp.sqrt(squared_sum / sum(value.size for value in leaves))


__all__ = ["PublicVEstimator", "public_preconditioner_rms"]
