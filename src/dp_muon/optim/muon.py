"""Classic-Nesterov Muon Optax transformations."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import optax


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


def muon_transform(
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    use_bf16_ns: bool = True,
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
  # On the supported JAX GPU stack BF16 arithmetic is available.  Casting
  # around NS keeps model/master updates in their original dtype while using
  # the requested low-precision NS path.  CPU-only setups can opt out.
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
  return optax.chain(
      classic_nesterov_momentum(momentum),
      *ns_path,
      _scale_by_consistent_rms(consistent_rms),
      optax.add_decayed_weights(weight_decay),
      optax.scale(-learning_rate),
  )


__all__ = ["classic_nesterov_momentum", "muon_transform"]
