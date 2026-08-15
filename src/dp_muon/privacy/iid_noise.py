"""IID Gaussian sampling shared by non-amplified private trainers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


PyTree = Any


def sample_iid_gaussian_noise(
    key: jax.Array, template: PyTree, noise_std: float | jax.Array
) -> tuple[PyTree, jax.Array]:
  """Samples one IID Gaussian PyTree matching ``template``.

  The key is split once for the complete mechanism, then deterministically
  split into one independent key per leaf.  Callers must add this returned
  tree to their complete clipped query before any optimizer partitioning.
  """
  leaves, treedef = jax.tree_util.tree_flatten(template)
  key, sample_key = jax.random.split(key)
  leaf_keys = jax.random.split(sample_key, len(leaves))
  noise_leaves = [
      jax.random.normal(leaf_key, jnp.asarray(leaf).shape, jnp.asarray(leaf).dtype)
      * jnp.asarray(noise_std, dtype=jnp.asarray(leaf).dtype)
      for leaf_key, leaf in zip(leaf_keys, leaves, strict=True)
  ]
  return jax.tree_util.tree_unflatten(treedef, noise_leaves), key


__all__ = ["sample_iid_gaussian_noise"]
