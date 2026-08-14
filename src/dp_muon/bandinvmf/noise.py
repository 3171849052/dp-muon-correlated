"""Streaming correlated Gaussian noise for a BandInvMF noising matrix.

``noising_coef`` directly parameterizes the lower-triangular Toeplitz matrix
``D = C^-1``.  The state deliberately stores latent iid noise, rather than
previous correlated outputs, so this is an FIR filter and not a recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BandInvMFNoiseState:
  """Latent-noise ring buffer used by the streaming BandInvMF filter.

  ``buffer`` has the same tree structure as a gradient, with every leaf shaped
  ``(bandwidth, *gradient_leaf.shape)``.  ``cursor`` points at the slot that
  will receive the next latent noise.  ``step`` is checkpoint/debug metadata.
  """

  buffer: PyTree
  cursor: jax.Array
  step: jax.Array
  bandwidth: int

  def tree_flatten(self):
    return (self.buffer, self.cursor, self.step), self.bandwidth

  @classmethod
  def tree_unflatten(cls, bandwidth: int, children: tuple[PyTree, jax.Array, jax.Array]):
    buffer, cursor, step = children
    return cls(buffer=buffer, cursor=cursor, step=step, bandwidth=bandwidth)


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


def _is_tracer(value: object) -> bool:
  """Whether value is a JAX tracing value rather than a concrete input."""
  return isinstance(value, jax.core.Tracer)


def _validated_coef(noising_coef: jax.Array) -> jax.Array:
  """Performs shape/dtype checks always and value checks for eager inputs.

  Finiteness is necessarily a runtime numerical property for a dynamic JIT
  argument.  It is checked at eager API boundaries, while traced values remain
  on the pure JAX execution path (where Python boolean conversion is invalid).
  """
  coef = jnp.asarray(noising_coef)
  if coef.ndim != 1 or coef.shape[0] == 0:
    raise ValueError("noising_coef must be a non-empty one-dimensional array")
  if not jnp.issubdtype(coef.dtype, jnp.floating):
    raise ValueError("noising_coef must be a floating array")
  if not _is_tracer(coef) and not bool(jnp.all(jnp.isfinite(coef))):
    raise ValueError("noising_coef must contain only finite values")
  return coef


def _validated_iid_noise_std(iid_noise_std: float | jax.Array) -> jax.Array:
  std = jnp.asarray(iid_noise_std)
  if std.ndim != 0 or not jnp.issubdtype(std.dtype, jnp.number):
    raise ValueError("iid_noise_std must be a finite scalar")
  if not _is_tracer(std) and (not bool(jnp.isfinite(std)) or not bool(std >= 0)):
    raise ValueError("iid_noise_std must be finite and non-negative")
  return std


def _validate_state(state: BandInvMFNoiseState, bandwidth: int) -> None:
  if not isinstance(state, BandInvMFNoiseState):
    raise TypeError("state must be a BandInvMFNoiseState")
  if state.bandwidth != bandwidth:
    raise ValueError("state bandwidth must equal len(noising_coef)")
  if not isinstance(state.bandwidth, Integral) or state.bandwidth < 1:
    raise ValueError("state bandwidth must be a positive integer")
  buffers = _require_floating_tree(state.buffer, "state.buffer")
  if any(buffer.ndim < 1 or buffer.shape[0] != bandwidth for buffer in buffers):
    raise ValueError("every state buffer leaf must have leading dimension bandwidth")
  cursor = jnp.asarray(state.cursor)
  step = jnp.asarray(state.step)
  if cursor.shape != () or not jnp.issubdtype(cursor.dtype, jnp.integer):
    raise ValueError("state.cursor must be an integer scalar")
  if step.shape != () or not jnp.issubdtype(step.dtype, jnp.integer):
    raise ValueError("state.step must be an integer scalar")
  if not _is_tracer(cursor) and not bool((cursor >= 0) & (cursor < bandwidth)):
    raise ValueError("state.cursor must be in [0, bandwidth)")


def init_bandinv_noise_state(template: PyTree, bandwidth: int) -> BandInvMFNoiseState:
  """Creates an all-zero latent-noise ring buffer for ``template``."""
  if not isinstance(bandwidth, Integral) or bandwidth < 1:
    raise ValueError("bandwidth must be a positive integer")
  _require_floating_tree(template, "template")
  buffer = jax.tree_util.tree_map(
      lambda leaf: jnp.zeros((bandwidth, *jnp.asarray(leaf).shape), dtype=jnp.asarray(leaf).dtype),
      template,
  )
  return BandInvMFNoiseState(
      buffer=buffer,
      cursor=jnp.array(0, dtype=jnp.int32),
      step=jnp.array(0, dtype=jnp.int32),
      bandwidth=int(bandwidth),
  )


def filter_latent_noise(
    state: BandInvMFNoiseState, latent_noise: PyTree, noising_coef: jax.Array
) -> tuple[PyTree, BandInvMFNoiseState]:
  """Filters supplied iid latent noise without sampling any random values."""
  coef = _validated_coef(noising_coef)
  bandwidth = coef.shape[0]
  _validate_state(state, bandwidth)
  _require_floating_tree(latent_noise, "latent_noise")
  if jax.tree_util.tree_structure(latent_noise) != jax.tree_util.tree_structure(state.buffer):
    raise ValueError("latent_noise and state.buffer must have identical PyTree structure")

  cursor = state.cursor
  indices = jnp.mod(cursor - jnp.arange(bandwidth, dtype=cursor.dtype), bandwidth)

  def write_buffer(buffer_leaf: jax.Array, latent_leaf: jax.Array) -> jax.Array:
    latent = jnp.asarray(latent_leaf)
    if buffer_leaf.shape[1:] != latent.shape:
      raise ValueError("latent_noise leaf shapes must match state buffer leaf shapes")
    if buffer_leaf.dtype != latent.dtype:
      raise ValueError("latent_noise leaf dtypes must match state buffer leaf dtypes")
    return buffer_leaf.at[cursor].set(latent)

  new_buffer = jax.tree_util.tree_map(write_buffer, state.buffer, latent_noise)

  def filter_buffer(buffer_leaf: jax.Array) -> jax.Array:
    # Keep the gradient/state dtype even when the fitted strategy is float64.
    leaf_coef = coef.astype(buffer_leaf.dtype)
    correlated = jnp.tensordot(leaf_coef, buffer_leaf[indices], axes=1)
    return correlated.astype(buffer_leaf.dtype)

  correlated_noise = jax.tree_util.tree_map(filter_buffer, new_buffer)
  new_state = BandInvMFNoiseState(
      buffer=new_buffer,
      cursor=jnp.mod(cursor + 1, bandwidth),
      step=state.step + jnp.array(1, dtype=state.step.dtype),
      bandwidth=bandwidth,
  )
  return correlated_noise, new_state


def sample_bandinv_noise(
    key: jax.Array,
    state: BandInvMFNoiseState,
    noising_coef: jax.Array,
    iid_noise_std: float | jax.Array,
) -> tuple[PyTree, BandInvMFNoiseState, jax.Array]:
  """Samples iid latent Gaussian noise, then applies ``filter_latent_noise``."""
  std = _validated_iid_noise_std(iid_noise_std)
  coef = _validated_coef(noising_coef)
  _validate_state(state, coef.shape[0])
  leaves, tree_def = jax.tree_util.tree_flatten(state.buffer)
  new_key, sample_key = jax.random.split(key)
  leaf_keys = jax.random.split(sample_key, len(leaves))
  latent_leaves = [
      jax.random.normal(leaf_key, buffer.shape[1:], dtype=buffer.dtype)
      * jnp.asarray(std, dtype=buffer.dtype)
      for leaf_key, buffer in zip(leaf_keys, leaves, strict=True)
  ]
  latent_noise = jax.tree_util.tree_unflatten(tree_def, latent_leaves)
  correlated_noise, new_state = filter_latent_noise(state, latent_noise, coef)
  return correlated_noise, new_state, new_key


__all__ = [
    "BandInvMFNoiseState",
    "filter_latent_noise",
    "init_bandinv_noise_state",
    "sample_bandinv_noise",
]
