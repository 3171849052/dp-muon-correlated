"""Inspectable, cumulative post-Nesterov Muon ``Q`` stages.

The implementation intentionally delegates Newton--Schulz to Optax 0.2.8,
the same primitive used by :mod:`dp_muon.optim.muon`.  It is for analysis of a
single rank-two pre-``Q`` matrix, not an optimizer transformation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import optax
from optax.contrib._muon import orthogonalize_via_newton_schulz

from .muon import muon_post_nesterov_transform


STAGES = ("linear", "bf16", "norm", "ns", "scale")
_NS_COEFFS = jnp.asarray((3.4445, -4.7750, 2.0315), dtype=jnp.float32)


def _matrix(value: jax.Array) -> jax.Array:
  matrix = jnp.asarray(value)
  if matrix.ndim != 2:
    raise ValueError("Muon Q expects one rank-two matrix")
  if not jnp.issubdtype(matrix.dtype, jnp.floating):
    raise ValueError("Muon Q expects a floating matrix")
  return matrix


def _orthogonalize(matrix: jax.Array, *, ns_steps: int) -> jax.Array:
  """Uses Optax's Frobenius preconditioning and NS implementation verbatim."""
  return orthogonalize_via_newton_schulz(
      matrix,
      _NS_COEFFS,
      ns_steps=ns_steps,
      preconditioning="frobenius",
      dimension_numbers=optax.contrib.MuonDimensionNumbers(),
  )


def muon_q_stages(
    matrix: jax.Array,
    *,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    use_bf16_ns: bool = True,
) -> Mapping[str, jax.Array]:
  """Returns the cumulative ``linear,bf16,norm,ns,scale`` Muon Q stages.

  ``norm`` is Optax's Frobenius preconditioner (zero NS iterations), and
  ``ns`` is the normalised BF16 input run through Optax's standard NS path.
  ``scale`` is calculated by the exact reusable post-Nesterov transform from
  :mod:`muon`, so its numerical semantics stay locked to production Muon.
  """
  if ns_steps < 1:
    raise ValueError("ns_steps must be positive")
  if consistent_rms <= 0 or not math.isfinite(consistent_rms):
    raise ValueError("consistent_rms must be finite and positive")
  linear = _matrix(matrix)
  bf16 = linear.astype(jnp.bfloat16) if use_bf16_ns else linear
  norm = _orthogonalize(bf16, ns_steps=0)
  ns = _orthogonalize(bf16, ns_steps=ns_steps)
  transform = muon_post_nesterov_transform(
      ns_steps=ns_steps,
      consistent_rms=consistent_rms,
      use_bf16_ns=use_bf16_ns,
  )
  scale, _ = transform.update(linear, transform.init(linear))
  return {"linear": linear, "bf16": bf16, "norm": norm, "ns": ns, "scale": scale}


def muon_q(
    matrix: jax.Array,
    *,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    use_bf16_ns: bool = True,
) -> jax.Array:
  """Returns the full production-equivalent post-Nesterov ``Q`` stage."""
  return muon_q_stages(
      matrix,
      ns_steps=ns_steps,
      consistent_rms=consistent_rms,
      use_bf16_ns=use_bf16_ns,
  )["scale"]


__all__ = ["STAGES", "muon_q", "muon_q_stages"]
