"""Direct public per-example second-moment estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp


PyTree = Any


@dataclass(frozen=True)
class PublicVEstimator:
  """Estimates ``E[g_i**2]`` from public examples at fixed parameters."""

  loss_fn: Callable[[PyTree, Any], jax.Array]
  eps: float = 1e-8

  def __post_init__(self) -> None:
    if not callable(self.loss_fn):
      raise TypeError("loss_fn must be callable")
    if not self.eps > 0:
      raise ValueError("eps must be positive")

  def squared_gradient_sum(
      self, params: PyTree, public_batch: Any
  ) -> tuple[PyTree, jax.Array]:
    """Returns ``sum_i g_i**2`` and the example count for one batch."""
    leaves = jax.tree_util.tree_leaves(public_batch)
    if not leaves:
      raise ValueError("public batch must have at least one leaf")
    if any(value.ndim < 1 for value in leaves):
      raise ValueError("public batch leaves must have a leading example axis")
    batch_size = leaves[0].shape[0]
    if batch_size < 1 or any(value.shape[0] != batch_size for value in leaves):
      raise ValueError("public batch leaves must have the same non-empty leading axis")

    def per_example_loss(parameters: PyTree, example: Any) -> jax.Array:
      singleton_batch = jax.tree_util.tree_map(
          lambda value: jnp.expand_dims(value, axis=0), example
      )
      return self.loss_fn(parameters, singleton_batch)

    def accumulate(squared_sum: PyTree, example: Any) -> tuple[PyTree, None]:
      gradient = jax.grad(per_example_loss)(params, example)
      return (
          jax.tree_util.tree_map(
              lambda total, value: total + jnp.square(value),
              squared_sum,
              gradient,
          ),
          None,
      )

    squared_sum, _ = jax.lax.scan(
        accumulate,
        jax.tree_util.tree_map(jnp.zeros_like, params),
        public_batch,
    )
    return squared_sum, jnp.asarray(batch_size, dtype=jnp.int32)

  def estimate_with_count(
      self,
      params: PyTree,
      public_batches: Iterable[Any],
      *,
      batch_estimate: Callable[[PyTree, Any], tuple[PyTree, jax.Array]] | None = None,
  ) -> tuple[PyTree, int]:
    """Accumulates a segment-local direct estimate across public batches."""
    estimate_batch = (
        self.squared_gradient_sum if batch_estimate is None else batch_estimate
    )
    squared_sum = None
    example_count = 0
    for public_batch in public_batches:
      batch_sum, batch_count = estimate_batch(params, public_batch)
      squared_sum = (
          batch_sum
          if squared_sum is None
          else jax.tree_util.tree_map(jnp.add, squared_sum, batch_sum)
      )
      example_count += int(batch_count)
    if squared_sum is None or example_count < 1:
      raise ValueError("each segment requires at least one public example")
    return (
        jax.tree_util.tree_map(lambda value: value / example_count, squared_sum),
        example_count,
    )

  def estimate(self, params: PyTree, public_batches: Iterable[Any]) -> PyTree:
    """Returns the direct public ``mean_i(g_i**2)`` estimate."""
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
