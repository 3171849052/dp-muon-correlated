"""Fixed-normalization, globally clipped gradient queries.

This module intentionally delegates all per-example clipping mathematics to
``jax_privacy.clipping.clipped_grad``.  It neither samples noise nor exposes
per-example diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from jax_privacy import clipping


def _positive_finite(name: str, value: float) -> float:
  """Validates public, static clipping configuration."""
  try:
    value = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be a finite positive number") from error
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f"{name} must be a finite positive number")
  return value


def make_clipped_gradient_query(
    loss_fn: Callable[..., Any],
    *,
    clip_norm: float,
    normalize_by: float,
    batch_argnums: int | tuple[int, ...] = 1,
    keep_batch_dim: bool = True,
) -> clipping.BoundedSensitivityCallable:
  """Builds ``sum_i clip_L(grad_i) / B0`` using JAX Privacy.

  ``clip_norm`` is passed as a scalar ``l2_clip_norm``, which makes
  :func:`jax_privacy.clipping.clipped_grad` clip the L2 norm over the complete
  parameter PyTree.  ``normalize_by`` is a fixed public constant and is never
  inferred from the runtime batch size.  The returned callable emits only the
  aggregated gradient; per-example values, norms, and auxiliary diagnostics
  are deliberately disabled.
  """
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  clip_norm = _positive_finite("clip_norm", clip_norm)
  normalize_by = _positive_finite("normalize_by", normalize_by)
  return clipping.clipped_grad(
      loss_fn,
      argnums=0,
      has_aux=False,
      l2_clip_norm=clip_norm,
      normalize_by=normalize_by,
      batch_argnums=batch_argnums,
      keep_batch_dim=keep_batch_dim,
      return_values=False,
      return_grad_norms=False,
  )


__all__ = ["make_clipped_gradient_query"]
