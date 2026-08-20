"""Exact p-aware linear Phase-II workload for frozen-p AdamW."""
from __future__ import annotations

from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


def frozen_p_time_workload(
    horizon: int, *, tau: int, beta1: float, learning_rate: float,
    weight_decay: float,
) -> np.ndarray:
  """Map Phase-II gradient perturbations to parameter perturbations.

  Row zero uses Adam count ``tau + 1``. The sign is included, so the returned
  matrix directly predicts ``delta_theta``.
  """
  if horizon < 1 or tau < 1:
    raise ValueError("horizon and tau must be positive")
  if not 0 <= beta1 < 1 or learning_rate <= 0 or weight_decay < 0:
    raise ValueError("invalid AdamW scalar configuration")
  a = np.zeros((horizon, horizon), dtype=np.float64)
  rho = 1.0 - learning_rate * weight_decay
  for source in range(horizon):
    dm = 0.0
    dtheta = 0.0
    for step in range(horizon):
      dm = beta1 * dm + ((1.0 - beta1) if step == source else 0.0)
      dtheta = rho * dtheta - learning_rate * dm / (1.0 - beta1 ** (tau + step + 1))
      a[step, source] = dtheta
  return a


def apply_frozen_p_workload(time_workload: np.ndarray, perturbations: PyTree,
                            p_star: PyTree) -> PyTree:
  """Apply ``A_time tensor Diag(p_star)`` to a time-major PyTree."""
  matrix = jnp.asarray(time_workload)
  return jax.tree_util.tree_map(
      lambda noise, p: jnp.einsum("ts,s...->t...", matrix, noise) * p,
      perturbations, p_star)


def p_weighted_squared_norm(trajectory: PyTree, p_star: PyTree | None = None) -> float:
  """Squared Frobenius norm, optionally with explicit coordinate p weighting."""
  if p_star is None:
    leaves = jax.tree_util.tree_leaves(trajectory)
  else:
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda value, p: value * p, trajectory, p_star))
  return float(sum(np.sum(np.asarray(leaf, dtype=np.float64) ** 2) for leaf in leaves))


__all__ = ["apply_frozen_p_workload", "frozen_p_time_workload", "p_weighted_squared_norm"]
