"""Parameter labels for the ViT-Tiny Muon/AdamW split."""

from __future__ import annotations

from typing import Any

import jax


PyTree = Any
MUON = "muon"
ADAMW = "adamw"
_ATTENTION = frozenset(("query", "key", "value", "out"))
_MLP = frozenset(("dense0", "dense1"))


def _path_names(path: tuple[jax.tree_util.KeyEntry, ...]) -> tuple[Any, ...]:
  """Converts JAX key entries to their dictionary/sequence values."""
  names = []
  for entry in path:
    if isinstance(entry, jax.tree_util.DictKey):
      names.append(entry.key)
    elif isinstance(entry, jax.tree_util.SequenceKey):
      names.append(entry.idx)
    else:
      names.append(str(entry))
  return tuple(names)


def is_muon_parameter_path(path: tuple[jax.tree_util.KeyEntry, ...]) -> bool:
  """Whether a flattened parameter path is one of the 72 block kernels."""
  names = _path_names(path)
  if len(names) != 5 or names[0] != "blocks" or not isinstance(names[1], int):
    return False
  _, _, family, layer, leaf = names
  return leaf == "kernel" and (
      (family == "attention" and layer in _ATTENTION)
      or (family == "mlp" and layer in _MLP)
  )


def vit_muon_parameter_labels(params: PyTree) -> PyTree:
  """Labels exactly Transformer attention/MLP kernels as ``muon``.

  It is intentionally path based, rather than rank based: patch embeddings,
  the classifier head, all vectors, and every future non-block matrix remain
  on AdamW.
  """
  return jax.tree_util.tree_map_with_path(
      lambda path, _: MUON if is_muon_parameter_path(path) else ADAMW, params
  )


def count_muon_parameters(params: PyTree) -> int:
  """Returns the number of leaves selected for Muon."""
  return sum(label == MUON for label in jax.tree_util.tree_leaves(
      vit_muon_parameter_labels(params)
  ))


__all__ = [
    "ADAMW",
    "MUON",
    "count_muon_parameters",
    "is_muon_parameter_path",
    "vit_muon_parameter_labels",
]
