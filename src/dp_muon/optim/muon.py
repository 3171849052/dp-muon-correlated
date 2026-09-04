"""Classic-Nesterov Muon Optax transformations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


ParameterPath = tuple[str | int, ...]


class PreQSVDState(NamedTuple):
  """Optax state containing only the latest pre-Q singular values.

  A named tuple is a regular JAX PyTree node.  Keeping only this small array
  makes the analysis hook safe to use in a long training run without retaining
  the full pre-Q matrix in the optimizer state.
  """

  singular_values: jax.Array


def _tree_value(tree: Any, path: ParameterPath) -> Any:
  value = tree
  for entry in path:
    value = value[entry]
  return value


def make_pre_q_svd_hook(parameter_path: ParameterPath) -> optax.GradientTransformation:
  """Capture one Muon pre-Q spectrum during the optimizer update.

  The hook is intended to be placed immediately after Muon's temporal
  momentum/Nesterov transform.  It forwards updates unchanged and stores only
  the exact SVD spectrum of the selected rank-two update in its Optax state.
  """
  if not isinstance(parameter_path, tuple) or not parameter_path:
    raise ValueError("parameter_path must be a non-empty tuple")
  if any(not isinstance(entry, (str, int)) or isinstance(entry, bool)
         for entry in parameter_path):
    raise ValueError("parameter_path entries must be strings or integers")

  def _matrix(tree: Any) -> jax.Array:
    matrix = _tree_value(tree, parameter_path)
    if isinstance(matrix, optax.MaskedNode):
      raise ValueError("pre-Q parameter path is not in the Muon partition")
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2 or not jnp.issubdtype(matrix.dtype, jnp.floating):
      raise ValueError("pre-Q parameter must be a floating rank-two matrix")
    return matrix

  def init_fn(params: Any) -> PreQSVDState:
    matrix = _matrix(params)
    return PreQSVDState(
        jnp.zeros((min(matrix.shape),), dtype=jnp.result_type(matrix, jnp.float32))
    )

  def update_fn(updates: Any, state: PreQSVDState, params: Any = None):
    del params
    if not isinstance(state, PreQSVDState):
      raise TypeError("state must be a PreQSVDState")
    matrix = _matrix(updates)
    # JAX/Optax returns singular values in descending order.  Sorting here is
    # an explicit invariant for the analysis artifact and costs no matrix
    # retention or singular-vector computation.
    singular_values = jnp.flip(
        jnp.sort(jnp.linalg.svd(matrix, compute_uv=False)), axis=0
    )
    return updates, PreQSVDState(singular_values)

  return optax.GradientTransformation(init_fn, update_fn)


def extract_pre_q_singular_values(optimizer_state: Any) -> jax.Array:
  """Find the spectrum emitted by :func:`make_pre_q_svd_hook` in Optax state."""
  if isinstance(optimizer_state, PreQSVDState):
    return optimizer_state.singular_values
  if isinstance(optimizer_state, Mapping):
    for value in optimizer_state.values():
      try:
        return extract_pre_q_singular_values(value)
      except ValueError:
        pass
  elif isinstance(optimizer_state, tuple):
    for value in optimizer_state:
      try:
        return extract_pre_q_singular_values(value)
      except ValueError:
        pass
  raise ValueError("optimizer state does not contain a pre-Q SVD hook")


def classic_nesterov_momentum(
    momentum: float = 0.95,
) -> optax.GradientTransformation:
  """Returns ``M_t=(1-b)G_t+bM_(t-1)``, ``U_t=(1-b)G_t+bM_t``.

  ``optax.trace(..., nesterov=True)`` has exactly this convention after the
  preceding ``scale(1-b)``.  It deliberately contains no bias correction.
  """
  if not 0.0 <= float(momentum) < 1.0:
    raise ValueError("momentum must be in [0, 1)")
  return optax.chain(
      optax.scale(1.0 - momentum),
      optax.trace(decay=momentum, nesterov=True),
  )


def _cast_tree(dtype: jnp.dtype) -> optax.GradientTransformation:
  """A tiny stateless Optax transform kept compatible with Optax 0.2.8."""
  def init_fn(params: Any) -> tuple[()]:
    del params
    return ()

  def update_fn(updates: Any, state: tuple[()], params: Any = None):
    del params
    return jax.tree_util.tree_map(
        lambda update: jnp.asarray(update, dtype=dtype), updates
    ), state

  return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_consistent_rms(
    consistent_rms: float,
) -> optax.GradientTransformation:
  """Muon's Moonlight RMS scaling for the selected rank-two matrices."""
  def init_fn(params: Any) -> tuple[()]:
    del params
    return ()

  def update_fn(updates: Any, state: tuple[()], params: Any = None):
    del params
    def scale(update: Any):
      matrix = jnp.asarray(update)
      if matrix.ndim != 2:
        raise ValueError("Muon parameters must be rank-two matrices")
      return matrix * (consistent_rms * math.sqrt(max(matrix.shape)))
    return jax.tree_util.tree_map(scale, updates), state

  return optax.GradientTransformation(init_fn, update_fn)


def muon_post_nesterov_transform(
    *,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    use_bf16_ns: bool = True,
) -> optax.GradientTransformation:
  """Returns Muon's matrix-only post-Nesterov ``Q`` transformation.

  This is deliberately the exact tail used by :func:`muon_transform`: it has
  no temporal momentum, weight decay, or learning-rate scaling.  Keeping it
  public gives analyses a single authoritative implementation of BF16,
  Frobenius Newton--Schulz, and consistent-RMS scaling.
  """
  if ns_steps < 1:
    raise ValueError("ns_steps must be positive")
  if consistent_rms <= 0:
    raise ValueError("consistent_rms must be positive")
  ns_path: list[optax.GradientTransformation] = []
  if use_bf16_ns:
    ns_path.append(_cast_tree(jnp.bfloat16))
  ns_path.append(optax.contrib.scale_by_muon(
      beta=0.0,
      nesterov=False,
      adaptive=False,
      ns_steps=ns_steps,
      preconditioning="frobenius",
      mu_dtype=jnp.bfloat16 if use_bf16_ns else None,
      weight_dimension_numbers=lambda tree: jax.tree_util.tree_map(
          lambda _: optax.contrib.MuonDimensionNumbers(), tree
      ),
  ))
  if use_bf16_ns:
    ns_path.append(_cast_tree(jnp.float32))
  ns_path.append(_scale_by_consistent_rms(consistent_rms))
  return optax.chain(*ns_path)


def muon_transform(
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    use_bf16_ns: bool = True,
    pre_q_hook: optax.GradientTransformation | None = None,
) -> optax.GradientTransformation:
  """Classic-Nesterov Muon with five-step Frobenius Newton--Schulz.

  ``scale_by_muon(beta=0)`` is used solely as Optax's public NS primitive;
  temporal semantics are wholly supplied by ``classic_nesterov_momentum``.
  This avoids the bias-corrected temporal path of ``optax.contrib.muon``.
  """
  if learning_rate <= 0:
    raise ValueError("learning_rate must be positive")
  if weight_decay < 0:
    raise ValueError("weight_decay must be non-negative")
  if ns_steps < 1:
    raise ValueError("ns_steps must be positive")
  if consistent_rms <= 0:
    raise ValueError("consistent_rms must be positive")
  if pre_q_hook is not None and not isinstance(
      pre_q_hook, optax.GradientTransformation
  ):
    raise TypeError("pre_q_hook must be an Optax GradientTransformation")
  transforms = [classic_nesterov_momentum(momentum)]
  if pre_q_hook is not None:
    transforms.append(pre_q_hook)
  transforms.extend([
      muon_post_nesterov_transform(
          ns_steps=ns_steps,
          consistent_rms=consistent_rms,
          use_bf16_ns=use_bf16_ns,
      ),
      optax.add_decayed_weights(weight_decay),
      optax.scale(-learning_rate),
  ])
  return optax.chain(*transforms)


__all__ = [
    "classic_nesterov_momentum",
    "extract_pre_q_singular_values",
    "make_pre_q_svd_hook",
    "muon_post_nesterov_transform",
    "muon_transform",
    "PreQSVDState",
]
