"""Hybrid IID/BandInvMF plans and full-transcript privacy calibration."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import optimization, toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy, fit_bandinv_strategy
from dp_muon.privacy import PrivacyCalibration, calibrate_nonamplified_bandinv

from exp5.workload import frozen_p_time_workload


def block_lengths(horizon: int, block_size: int | None) -> tuple[int, ...]:
  if horizon < 1:
    raise ValueError("horizon must be positive")
  if block_size is None:
    return (horizon,)
  if block_size < 1:
    raise ValueError("block_size must be positive")
  full, remainder = divmod(horizon, block_size)
  return (block_size,) * full + ((remainder,) if remainder else ())


def _lower_toeplitz(coef: np.ndarray, horizon: int) -> np.ndarray:
  coef = np.asarray(coef, dtype=np.float64)
  offsets = np.arange(horizon)[:, None] - np.arange(horizon)[None, :]
  padded = np.pad(coef, (0, max(0, horizon - len(coef))))
  return np.where(offsets >= 0, padded[np.maximum(offsets, 0)], 0.0)


def hybrid_strategy_matrix(tau: int, strategies: tuple[BandInvMFStrategy, ...]) -> np.ndarray:
  """Materialize ``C_hybrid = blockdiag(I_tau, C_1, ...)``."""
  if tau < 1:
    raise ValueError("tau must be positive")
  blocks = [np.eye(tau)] + [
      _lower_toeplitz(np.asarray(strategy.strategy_coef), strategy.horizon)
      for strategy in strategies]
  total = sum(block.shape[0] for block in blocks)
  result = np.zeros((total, total), dtype=np.float64)
  offset = 0
  for block in blocks:
    length = block.shape[0]
    result[offset:offset + length, offset:offset + length] = block
    offset += length
  return result


def hybrid_noising_matrix(tau: int, strategies: tuple[BandInvMFStrategy, ...]) -> np.ndarray:
  """Materialize ``D_hybrid = blockdiag(I_tau, D_1, ...)``."""
  if tau < 1:
    raise ValueError("tau must be positive")
  blocks = [np.eye(tau)] + [
      _lower_toeplitz(np.asarray(strategy.noising_coef), strategy.horizon)
      for strategy in strategies]
  total = sum(block.shape[0] for block in blocks)
  result = np.zeros((total, total), dtype=np.float64)
  offset = 0
  for block in blocks:
    length = block.shape[0]
    result[offset:offset + length, offset:offset + length] = block
    offset += length
  return result


def hybrid_sensitivity_squared(
    strategy_matrix: np.ndarray, *, min_sep: int, max_participations: int,
    block_sizes: tuple[int, ...] | None = None,
) -> float:
  """Exact ``max ||C x_pi||^2`` under the original global contract.

  This enumerates only legal participation tuples and is intended for the
  short Exp5 horizon/cap (cap is five in the CIFAR contract). It deliberately
  treats boundaries as ordinary global time positions.
  """
  c = np.asarray(strategy_matrix, dtype=np.float64)
  if c.ndim != 2 or c.shape[0] != c.shape[1]:
    raise ValueError("strategy_matrix must be square")
  if min_sep < 1 or max_participations < 1:
    raise ValueError("participation constraints must be positive")
  n = c.shape[1]
  sizes = (n,) if block_sizes is None else tuple(block_sizes)
  if any(size < 1 for size in sizes) or sum(sizes) != n:
    raise ValueError("block_sizes must be positive and cover the matrix")
  # Because C is block diagonal, local energies add. The DP carries the last
  # absolute participation so separation remains global across every boundary.
  states: dict[tuple[int, int], float] = {(-min_sep, 0): 0.0}
  offset = 0
  for size in sizes:
    block = c[offset:offset + size, offset:offset + size]
    gram = block.T @ block
    option_map: dict[tuple[int, int, int], float] = {}
    local_cap = min(max_participations, 1 + (size - 1) // min_sep)
    for count in range(1, local_cap + 1):
      compressed_n = size - (count - 1) * (min_sep - 1)
      for compressed in combinations(range(compressed_n), count):
        columns = tuple(value + i * (min_sep - 1)
                        for i, value in enumerate(compressed))
        energy = float(np.sum(gram[np.ix_(columns, columns)]))
        key = (columns[0], columns[-1], count)
        option_map[key] = max(option_map.get(key, float("-inf")), energy)
    updated = dict(states)
    for (last, used), old_energy in states.items():
      for (first, local_last, count), energy in option_map.items():
        absolute_first = offset + first
        if used + count <= max_participations and absolute_first - last >= min_sep:
          key = (offset + local_last, used + count)
          updated[key] = max(updated.get(key, float("-inf")), old_energy + energy)
    states = updated
    offset += size
  return max(states.values())


@dataclass(frozen=True)
class HybridPlan:
  tau: int
  horizon: int
  block_size: int | None
  block_lengths: tuple[int, ...]
  strategies: tuple[BandInvMFStrategy, ...]
  sensitivity_squared: float
  calibration: PrivacyCalibration


def _fit_rectangular_block(
    workload: np.ndarray, *, length: int, bandwidth: int, min_sep: int,
    max_participations: int, max_optimizer_steps: int, reduction: str,
) -> BandInvMFStrategy:
  """Fit one block against all full-workload rows affected by its columns."""
  matrix = jnp.asarray(workload)
  if matrix.ndim != 2 or matrix.shape[1] != length:
    raise ValueError("rectangular block workload has the wrong shape")
  reduction_fn = {"mean": jnp.mean, "max": jnp.max,
                  "last": lambda values: values[-1]}.get(reduction)
  if reduction_fn is None:
    raise ValueError("reduction must be mean, max, or last")
  initial = toeplitz.banded_inverse_square_root_noising_coefs(bandwidth)
  row = jnp.arange(length)[:, None]
  column = jnp.arange(length)[None, :]
  lag = row - column

  def product(coef):
    d = jnp.where((lag >= 0) & (lag < bandwidth), coef[jnp.clip(lag, 0)], 0.)
    return matrix @ d

  def loss(coef):
    error = reduction_fn(jnp.sum(product(coef) ** 2, axis=1))
    sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
        n=length, noising_coef=coef, min_sep=min(min_sep, length),
        max_participations=max_participations, use_matrix_upper_bound=False)
    return error * sensitivity

  noising = optimization.optimize(loss, initial, max_optimizer_steps=max_optimizer_steps)
  noising = noising / noising[0]
  strategy = toeplitz.inverse_coef(noising, length)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=length, noising_coef=noising, min_sep=min(min_sep, length),
      max_participations=max_participations)
  objective = reduction_fn(jnp.sum(product(noising) ** 2, axis=1)) * sensitivity
  return BandInvMFStrategy(
      length, bandwidth, min(min_sep, length), max_participations, None,
      noising, strategy, sensitivity, objective, workload_matrix=matrix)


def p_aware_hybrid_objective(
    plan: HybridPlan, p_star, *, beta1: float, learning_rate: float,
    weight_decay: float, reduction: str = "mean",
) -> float:
  """Evaluate the complete Phase-II objective with explicit p weighting."""
  phase_horizon = plan.horizon - plan.tau
  a = frozen_p_time_workload(
      phase_horizon, tau=plan.tau, beta1=beta1,
      learning_rate=learning_rate, weight_decay=weight_decay)
  d = hybrid_noising_matrix(1, plan.strategies)[1:, 1:]
  errors = np.sum((a @ d) ** 2, axis=1)
  reduce = {"mean": np.mean, "max": np.max,
            "last": lambda values: values[-1]}.get(reduction)
  if reduce is None:
    raise ValueError("reduction must be mean, max, or last")
  p_energy = sum(float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
                 for leaf in jax.tree_util.tree_leaves(p_star))
  return float(reduce(errors) * p_energy * plan.sensitivity_squared)


def share_conservative_calibration(plans: dict[str, HybridPlan]) -> dict[str, HybridPlan]:
  """Give paired conditions one noise scale calibrated to worst sensitivity.

  Every returned mechanism is bounded by the same final epsilon/delta, and
  identical base draws in the IID prefix become identical actual draws.
  """
  if not plans:
    raise ValueError("plans must not be empty")
  from dataclasses import replace
  first = next(iter(plans.values())).calibration
  if any((p.calibration.epsilon, p.calibration.delta, p.calibration.adjacency,
          p.calibration.clip_norm, p.calibration.normalize_by) !=
         (first.epsilon, first.delta, first.adjacency, first.clip_norm,
          first.normalize_by) for p in plans.values()):
    raise ValueError("paired plans must share one privacy/query contract")
  worst = max(p.sensitivity_squared for p in plans.values())
  common = calibrate_nonamplified_bandinv(
      epsilon=first.epsilon, delta=first.delta, clip_norm=first.clip_norm,
      normalize_by=first.normalize_by, adjacency=first.adjacency,
      sensitivity_squared=worst)
  return {name: replace(plan, calibration=common) for name, plan in plans.items()}


def fit_hybrid_plan(
    *, horizon: int, tau: int, block_size: int | None, bandwidth: int,
    min_sep: int, max_participations: int, learning_rate: float, beta1: float,
    weight_decay: float, epsilon: float, delta: float, clip_norm: float,
    normalize_by: float, adjacency: str, max_optimizer_steps: int,
    reduction: str = "mean",
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> HybridPlan:
  """Fit Phase-II blocks, then calibrate the complete hybrid transcript once."""
  if not 1 <= tau < horizon:
    raise ValueError("tau must lie in [1, horizon)")
  lengths = block_lengths(horizon - tau, block_size)
  strategies = []
  phase_workload = frozen_p_time_workload(
      horizon - tau, tau=tau, beta1=beta1, learning_rate=learning_rate,
      weight_decay=weight_decay)
  offset = 0
  for length in lengths:
    # Rows after this block remain part of its parameter-trajectory workload.
    rectangular = np.abs(phase_workload[offset:, offset:offset + length])
    if fit_strategy is fit_bandinv_strategy:
      strategy = _fit_rectangular_block(
          rectangular, length=length, bandwidth=min(bandwidth, length),
          min_sep=min_sep, max_participations=max_participations,
          max_optimizer_steps=max_optimizer_steps, reduction=reduction)
    else:
      # Test/custom adapters retain the repository's public square API.
      strategy = fit_strategy(
          length, min(bandwidth, length), min(min_sep, length),
          max_participations=max_participations,
          workload_matrix=rectangular[:length],
          max_optimizer_steps=max_optimizer_steps, reduction=reduction)
    strategies.append(strategy)
    offset += length
  strategies_tuple = tuple(strategies)
  sensitivity = hybrid_sensitivity_squared(
      hybrid_strategy_matrix(tau, strategies_tuple), min_sep=min_sep,
      max_participations=max_participations, block_sizes=(tau, *lengths))
  calibration = calibrate_nonamplified_bandinv(
      epsilon=epsilon, delta=delta, clip_norm=clip_norm,
      normalize_by=normalize_by, adjacency=adjacency,
      sensitivity_squared=sensitivity)
  return HybridPlan(tau, horizon, block_size, lengths, strategies_tuple,
                    sensitivity, calibration)


__all__ = [
    "HybridPlan", "block_lengths", "fit_hybrid_plan", "hybrid_noising_matrix",
    "hybrid_sensitivity_squared", "hybrid_strategy_matrix",
    "p_aware_hybrid_objective", "share_conservative_calibration",
]
