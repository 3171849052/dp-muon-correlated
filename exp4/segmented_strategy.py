"""Exact privacy calibration and execution helpers for segmented BandInvMF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from dp_muon.bandinvmf import (
    BandInvMFNoiseState, BandInvMFStrategy, fit_bandinv_strategy,
    init_bandinv_noise_state, sample_bandinv_noise,
)
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.privacy import (
    PrivacyCalibration, calibrate_nonamplified_bandinv, make_clipped_gradient_query,
)
from dp_muon.training.nonamplified_bandinv_dpadamw import NonAmplifiedBandInvDPAdamWState
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer


@dataclass(frozen=True)
class SegmentedPlan:
  condition: str
  block_lengths: tuple[int, ...]
  strategies: tuple[BandInvMFStrategy, ...]
  sensitivity_squared: float
  calibration: PrivacyCalibration


def block_lengths(horizon: int, block_size: int) -> tuple[int, ...]:
  if horizon < 1 or block_size < 1:
    raise ValueError("horizon and block_size must be positive")
  full, remainder = divmod(horizon, block_size)
  return (block_size,) * full + ((remainder,) if remainder else ())


def _dense_strategy(coef: Any) -> jax.Array:
  coef = jnp.asarray(coef)
  n = coef.shape[0]
  row, column = jnp.arange(n)[:, None], jnp.arange(n)[None, :]
  return jnp.where(row >= column, coef[row - column], 0.0)


def global_segmented_sensitivity_squared(
    strategies: tuple[BandInvMFStrategy, ...], *, min_sep: int,
    max_participations: int | None,
) -> float:
  """Exact max energy over the original full-horizon participation contract.

  Exp4 blocks never exceed ``min_sep`` (97 and 16 versus 97), hence a legal
  record occurs at most once per block. Dynamic programming retains the last
  global participation and count, and considers every column in every block.
  """
  if min_sep < 1:
    raise ValueError("min_sep must be positive")
  if any(strategy.horizon > min_sep for strategy in strategies):
    raise ValueError("exact Exp4 DP requires every block length <= min_sep")
  horizon = sum(strategy.horizon for strategy in strategies)
  cap = horizon if max_participations is None else max_participations
  states: dict[tuple[int, int], float] = {(-min_sep, 0): 0.0}
  offset = 0
  for strategy in strategies:
    matrix = _dense_strategy(strategy.strategy_coef)
    column_energy = [float(jnp.sum(matrix[:, index] ** 2)) for index in range(strategy.horizon)]
    updated = dict(states)  # no participation in this block
    for (last, count), energy in states.items():
      if count >= cap:
        continue
      for local, contribution in enumerate(column_energy):
        position = offset + local
        if position - last >= min_sep:
          key = (position, count + 1)
          updated[key] = max(updated.get(key, float("-inf")), energy + contribution)
    states = updated
    offset += strategy.horizon
  return max(states.values())


def fit_segmented_plan(*, horizon: int, block_size: int, bandwidth: int, min_sep: int,
    max_participations: int | None, max_optimizer_steps: int, reduction: str,
    learning_rate: float, weight_decay: float, epsilon: float, delta: float,
    clip_norm: float, normalize_by: float, adjacency: str,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy) -> SegmentedPlan:
  lengths = block_lengths(horizon, block_size)
  fitted: dict[int, BandInvMFStrategy] = {}
  for length in sorted(set(lengths)):
    fitted[length] = fit_strategy(
        length, min(bandwidth, length), min_sep=min(min_sep, length),
        max_participations=max_participations,
        max_optimizer_steps=max_optimizer_steps, reduction=reduction,
        workload_coef=decayed_prefix_sum_workload_coef(
            length, learning_rate, weight_decay),
    )
  strategies = tuple(fitted[length] for length in lengths)
  sensitivity_squared = global_segmented_sensitivity_squared(
      strategies, min_sep=min_sep, max_participations=max_participations)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=epsilon, delta=delta, clip_norm=clip_norm,
      normalize_by=normalize_by, adjacency=adjacency,
      sensitivity_squared=sensitivity_squared)
  return SegmentedPlan(
      f"seg{block_size}", lengths, strategies, sensitivity_squared, calibration)


def coefficient_schedule(plan: SegmentedPlan) -> jax.Array:
  bandwidth = max(strategy.bandwidth for strategy in plan.strategies)
  rows = []
  for strategy in plan.strategies:
    coef = jnp.asarray(strategy.noising_coef)
    coef = jnp.pad(coef, (0, bandwidth - coef.shape[0]))
    rows.extend([coef] * strategy.horizon)
  return jnp.stack(rows)


def reset_noise_state(params: Any, *, bandwidth: int) -> BandInvMFNoiseState:
  """Create empty block-local FIR memory; no optimizer field is involved."""
  return init_bandinv_noise_state(params, bandwidth)


def segment_starts(plan: SegmentedPlan) -> frozenset[int]:
  starts, total = [], 0
  for length in plan.block_lengths[:-1]:
    total += length
    starts.append(total)
  return frozenset(starts)


def begin_segment(state: NonAmplifiedBandInvDPAdamWState, logical_step: int,
                  plan: SegmentedPlan) -> NonAmplifiedBandInvDPAdamWState:
  if logical_step not in segment_starts(plan):
    return state
  fresh = reset_noise_state(state.params, bandwidth=state.noise_state.bandwidth)
  return NonAmplifiedBandInvDPAdamWState(
      state.params, state.optimizer_state, fresh, state.rng_key, state.step)


def make_segmented_train_step(loss_fn: Callable[..., Any], plan: SegmentedPlan,
    *, learning_rate: float, beta1: float, beta2: float, eps: float,
    weight_decay: float, microbatch_size: int | None = None):
  """One clipped query and AdamW update with block-local BandInvMF memory."""
  schedule = coefficient_schedule(plan)
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate, beta1=beta1, beta2=beta2,
      eps=eps, weight_decay=weight_decay)
  query = make_clipped_gradient_query(
      loss_fn, clip_norm=plan.calibration.clip_norm,
      normalize_by=plan.calibration.normalize_by, batch_argnums=1,
      keep_batch_dim=True, microbatch_size=microbatch_size)

  def train_step(state: NonAmplifiedBandInvDPAdamWState, batch: Any):
    clipped = query(state.params, batch)
    coef = schedule[state.step]
    noise, noise_state, key = sample_bandinv_noise(
        state.rng_key, state.noise_state, coef, plan.calibration.iid_noise_std)
    private = jax.tree_util.tree_map(lambda gradient, perturbation: gradient + perturbation,
                                    clipped, noise)
    updates, optimizer_state = optimizer.update(private, state.optimizer_state, state.params)
    return NonAmplifiedBandInvDPAdamWState(
        optax.apply_updates(state.params, updates), optimizer_state, noise_state,
        key, state.step + jnp.array(1, state.step.dtype))
  return train_step, optimizer


__all__ = [
    "SegmentedPlan", "begin_segment", "block_lengths", "coefficient_schedule",
    "fit_segmented_plan", "global_segmented_sensitivity_squared",
    "make_segmented_train_step", "reset_noise_state", "segment_starts",
]
