"""Small, side-effect-free diagnostics for the real AdamW state."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

PyTree = Any

def compute_p_tree(optimizer_state: Any, *, beta2: float, eps: float, step: int | None = None) -> PyTree:
  """Compute p from Optax's *actual* Adam second moment (``nu``).

  Optax stores the uncorrected ``nu`` and ``count`` in ``ScaleByAdamState``.
  ``step`` is optional for tests and is otherwise read from the optimizer state.
  """
  def find(value):
    found = getattr(value, "nu", None)
    if found is not None:
      return found, getattr(value, "count", None)
    if isinstance(value, (tuple, list)):
      for item in value:
        result = find(item)
        if result[0] is not None:
          return result
    return None, None
  nu, state_count = find(optimizer_state)
  if nu is None:
    raise TypeError("optimizer_state must expose Optax Adam second moment 'nu'")
  count = int(step if step is not None else np.asarray(state_count))
  if count < 1:
    raise ValueError("p_t is defined after the first optimizer step")
  correction = 1.0 - float(beta2) ** count
  return jax.tree_util.tree_map(lambda v: 1.0 / (jnp.sqrt(v / correction) + eps), nu)

def _flat(tree: PyTree) -> np.ndarray:
  return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in jax.tree_util.tree_leaves(tree)])

@dataclass(frozen=True)
class PDiagnostics:
  step: int
  p_mean: float
  p_median: float
  p_p10: float
  p_p25: float
  p_p75: float
  p_p90: float
  p_rms: float
  relative_change: float

def p_tree_statistics(p_tree: PyTree, previous: PyTree | None = None, *, step: int) -> tuple[PDiagnostics, PyTree]:
  values = _flat(p_tree)
  old = None if previous is None else _flat(previous)
  relative = 0.0 if old is None else float(np.linalg.norm(values - old) / max(np.linalg.norm(old), 1e-30))
  row = PDiagnostics(int(step), float(np.mean(values)), float(np.median(values)),
      float(np.percentile(values, 10)), float(np.percentile(values, 25)),
      float(np.percentile(values, 75)), float(np.percentile(values, 90)),
      float(np.sqrt(np.mean(values * values))), relative)
  return row, p_tree

def diagnostics_row_dict(row: PDiagnostics) -> dict[str, float | int]:
  return asdict(row)

__all__ = ["PDiagnostics", "compute_p_tree", "p_tree_statistics", "diagnostics_row_dict"]
