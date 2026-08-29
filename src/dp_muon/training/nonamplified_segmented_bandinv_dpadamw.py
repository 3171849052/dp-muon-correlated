"""Segmented BandInvMF-correlated non-amplified DP-AdamW.

This module is the production version of the segmented prototype.  The
segment boundary is a boundary of the correlated-noise mechanism only:
AdamW's parameters and optimizer state are deliberately continuous over the
whole transcript.

The first implementation intentionally requires ``segment_length <= min_sep``.
Under that restriction a record can participate at most once in each diagonal
block, so the exact global sensitivity is a small dynamic program over the
original full-horizon participation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    fit_bandinv_strategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.privacy import PrivacyCalibration, calibrate_nonamplified_bandinv, make_clipped_gradient_query

from .nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer


PyTree = Any


@dataclass(frozen=True)
class SegmentedPlan:
  """Fitted per-length strategies and one global privacy calibration.

  ``runtime_bandwidth`` is the maximum fitted bandwidth.  Strategies for
  shorter blocks are padded with zero coefficients at execution time.  This
  is mathematically identical to using their shorter FIR and keeps all
  segment states JIT-compatible.
  """

  condition: str
  block_lengths: tuple[int, ...]
  strategies: tuple[BandInvMFStrategy, ...]
  sensitivity_squared: float
  calibration: PrivacyCalibration
  global_min_sep: int | None = None
  max_participations: int | None = None
  runtime_bandwidth: int | None = None

  @property
  def horizon(self) -> int:
    return sum(self.block_lengths)

  @property
  def segment_length(self) -> int:
    return max(self.block_lengths)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SegmentedBandInvDPAdamWState:
  """Checkpointable state with separate global and segment-local counters.

  ``step`` is the global AdamW training step.  ``noise_state.step`` is the
  local FIR step within the current segment and is reset to zero at a segment
  boundary.  ``rng_root_key`` is immutable; ``rng_key`` is the current
  segment's advancing key.
  """

  params: PyTree
  optimizer_state: Any
  noise_state: BandInvMFNoiseState
  rng_root_key: jax.Array
  rng_key: jax.Array
  step: jax.Array
  segment_index: jax.Array
  segment_start: jax.Array

  def tree_flatten(self):
    return (
        self.params,
        self.optimizer_state,
        self.noise_state,
        self.rng_root_key,
        self.rng_key,
        self.step,
        self.segment_index,
        self.segment_start,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]):
    del aux_data
    (
        params,
        optimizer_state,
        noise_state,
        rng_root_key,
        rng_key,
        step,
        segment_index,
        segment_start,
    ) = children
    return cls(
        params=params,
        optimizer_state=optimizer_state,
        noise_state=noise_state,
        rng_root_key=rng_root_key,
        rng_key=rng_key,
        step=step,
        segment_index=segment_index,
        segment_start=segment_start,
    )


def block_lengths(horizon: int, segment_length: int) -> tuple[int, ...]:
  """Returns ``[L, L, ..., remainder]`` for a positive horizon."""
  if (
      not isinstance(horizon, Integral)
      or isinstance(horizon, bool)
      or horizon < 1
      or not isinstance(segment_length, Integral)
      or isinstance(segment_length, bool)
      or segment_length < 1
  ):
    raise ValueError("horizon and segment_length must be positive integers")
  full, remainder = divmod(int(horizon), int(segment_length))
  return (int(segment_length),) * full + ((remainder,) if remainder else ())


def _dense_strategy(strategy_coef: Any) -> np.ndarray:
  coef = np.asarray(strategy_coef)
  if coef.ndim != 1:
    raise ValueError("strategy_coef must be one-dimensional")
  n = coef.shape[0]
  rows, columns = np.indices((n, n))
  return np.where(rows >= columns, coef[rows - columns], 0.0)


def global_segmented_sensitivity_squared(
    strategies: tuple[BandInvMFStrategy, ...] | list[BandInvMFStrategy],
    *,
    min_sep: int,
    max_participations: int | None,
) -> float:
  """Computes exact block-diagonal sensitivity under the global contract.

  Each block is a separate diagonal mechanism.  Since every block length is
  at most ``min_sep``, a legal record appears at most once in a block.  The DP
  retains the last global appearance and the number of appearances and sums
  the squared column energy of the selected block-diagonal columns.
  """
  if not isinstance(min_sep, Integral) or isinstance(min_sep, bool) or min_sep < 1:
    raise ValueError("min_sep must be a positive integer")
  if max_participations is not None and (
      not isinstance(max_participations, Integral)
      or isinstance(max_participations, bool)
      or max_participations < 1
  ):
    raise ValueError("max_participations must be a positive integer when supplied")
  if not strategies:
    raise ValueError("strategies must be non-empty")

  expected_offset = 0
  prepared: list[tuple[int, list[float]]] = []
  for strategy in strategies:
    if not isinstance(strategy, BandInvMFStrategy):
      raise TypeError("strategies must contain BandInvMFStrategy values")
    if strategy.horizon < 1 or strategy.horizon > min_sep:
      raise ValueError("exact segmented sensitivity requires every block length <= min_sep")
    if strategy.strategy_coef is None:
      raise ValueError("strategy.strategy_coef must be present")
    matrix = _dense_strategy(strategy.strategy_coef)
    if matrix.shape != (strategy.horizon, strategy.horizon):
      raise ValueError("strategy.strategy_coef must have shape (strategy.horizon,)")
    energies = [float(np.sum(matrix[:, index] ** 2)) for index in range(strategy.horizon)]
    prepared.append((expected_offset, energies))
    expected_offset += strategy.horizon

  cap = expected_offset if max_participations is None else int(max_participations)
  states: dict[tuple[int, int], float] = {(-int(min_sep), 0): 0.0}
  for offset, energies in prepared:
    updated = dict(states)  # A record may skip the complete block.
    for (last, count), energy in states.items():
      if count >= cap:
        continue
      for local, contribution in enumerate(energies):
        position = offset + local
        if position - last >= min_sep:
          key = (position, count + 1)
          updated[key] = max(
              updated.get(key, float("-inf")), energy + contribution
          )
    states = updated
  result = max(states.values())
  if not np.isfinite(result) or result <= 0:
    raise ValueError("global segmented sensitivity must be finite and positive")
  return float(result)


def fit_segmented_plan(
    *,
    horizon: int,
    bandwidth: int,
    min_sep: int,
    max_participations: int | None,
    max_optimizer_steps: int,
    reduction: str,
    learning_rate: float,
    weight_decay: float,
    epsilon: float,
    delta: float,
    clip_norm: float,
    normalize_by: float,
    adjacency: str,
    segment_length: int | None = None,
    block_size: int | None = None,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> SegmentedPlan:
  """Fits one strategy per unique segment length and calibrates once globally."""
  if segment_length is not None and block_size is not None and segment_length != block_size:
    raise ValueError("segment_length and block_size must agree")
  chosen_length = segment_length if segment_length is not None else block_size
  if chosen_length is None:
    raise ValueError("segment_length is required")
  if chosen_length > min_sep:
    raise ValueError("the first segmented implementation requires segment_length <= min_sep")
  lengths = block_lengths(horizon, chosen_length)
  fitted: dict[int, BandInvMFStrategy] = {}
  for length in sorted(set(lengths)):
    fitted[length] = fit_strategy(
        length,
        min(bandwidth, length),
        min_sep=min(min_sep, length),
        max_participations=max_participations,
        max_optimizer_steps=max_optimizer_steps,
        reduction=reduction,
        workload_coef=decayed_prefix_sum_workload_coef(
            length, learning_rate, weight_decay
        ),
    )
    if not isinstance(fitted[length], BandInvMFStrategy):
      raise TypeError("fit_strategy must return a BandInvMFStrategy")
  strategies = tuple(fitted[length] for length in lengths)
  sensitivity_squared = global_segmented_sensitivity_squared(
      strategies, min_sep=min_sep, max_participations=max_participations
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=epsilon,
      delta=delta,
      clip_norm=clip_norm,
      normalize_by=normalize_by,
      adjacency=adjacency,  # type: ignore[arg-type]
      sensitivity_squared=sensitivity_squared,
  )
  return SegmentedPlan(
      condition=f"seg{chosen_length}",
      block_lengths=lengths,
      strategies=strategies,
      sensitivity_squared=sensitivity_squared,
      calibration=calibration,
      global_min_sep=int(min_sep),
      max_participations=max_participations,
      runtime_bandwidth=max(strategy.bandwidth for strategy in strategies),
  )


def segmented_prefix_sensitivity_squared(
    plan: SegmentedPlan, prefix_steps: int
) -> float:
  """Returns sensitivity of a transcript prefix at the fixed global scale."""
  if not isinstance(prefix_steps, Integral) or isinstance(prefix_steps, bool):
    raise ValueError("prefix_steps must be an integer")
  if not 1 <= prefix_steps <= plan.horizon:
    raise ValueError("prefix_steps must lie in [1, plan.horizon]")
  remaining = int(prefix_steps)
  prefix_strategies: list[BandInvMFStrategy] = []
  global_min_sep = plan.global_min_sep or min(
      strategy.min_sep for strategy in plan.strategies
  )
  for strategy, length in zip(plan.strategies, plan.block_lengths, strict=True):
    if remaining <= 0:
      break
    local_length = min(length, remaining)
    if local_length == length:
      prefix_strategies.append(strategy)
    else:
      prefix_strategies.append(
          replace(
              strategy,
              horizon=local_length,
              bandwidth=min(strategy.bandwidth, local_length),
              min_sep=min(global_min_sep, local_length),
              noising_coef=strategy.noising_coef[:local_length],
              strategy_coef=strategy.strategy_coef[:local_length],
              workload_coef=(
                  None
                  if strategy.workload_coef is None
                  else strategy.workload_coef[:local_length]
              ),
          )
      )
    remaining -= local_length
  return global_segmented_sensitivity_squared(
      prefix_strategies,
      min_sep=global_min_sep,
      max_participations=plan.max_participations,
  )


def epsilon_spent_for_segmented_prefix(
    plan: SegmentedPlan, prefix_steps: int
) -> float:
  """Accounts a prefix using the one calibration for the full block diagonal."""
  if prefix_steps == plan.horizon:
    return plan.calibration.epsilon
  sensitivity_squared = segmented_prefix_sensitivity_squared(plan, prefix_steps)
  mu = (
      plan.calibration.query_sensitivity
      * np.sqrt(sensitivity_squared)
      / plan.calibration.iid_noise_std
  )
  epsilon = float(gdp.eps_from_mu(mu=float(mu), delta=plan.calibration.delta))
  if not np.isfinite(epsilon) or epsilon < 0:
    raise RuntimeError("Opacus GDP conversion returned an invalid epsilon")
  return epsilon


def coefficient_schedule(plan: SegmentedPlan) -> jax.Array:
  """Returns one zero-padded noising-coefficient vector per global step."""
  if not plan.strategies or len(plan.strategies) != len(plan.block_lengths):
    raise ValueError("plan must contain one strategy per block")
  bandwidth = plan.runtime_bandwidth or max(strategy.bandwidth for strategy in plan.strategies)
  rows = []
  for length, strategy in zip(plan.block_lengths, plan.strategies, strict=True):
    if strategy.horizon != length or strategy.noising_coef.shape != (strategy.bandwidth,):
      raise ValueError("each strategy must match its block length and bandwidth")
    coef = jnp.asarray(strategy.noising_coef)
    rows.extend([jnp.pad(coef, (0, bandwidth - strategy.bandwidth))] * length)
  return jnp.stack(rows)


def segment_starts(plan: SegmentedPlan) -> frozenset[int]:
  starts: list[int] = []
  total = 0
  for length in plan.block_lengths[:-1]:
    total += length
    starts.append(total)
  return frozenset(starts)


def _segment_start_schedule(plan: SegmentedPlan) -> tuple[jax.Array, jax.Array]:
  starts = [0]
  for length in plan.block_lengths[:-1]:
    starts.append(starts[-1] + length)
  segment_ids = np.repeat(np.arange(len(plan.block_lengths), dtype=np.int32), plan.block_lengths)
  return jnp.asarray(segment_ids), jnp.asarray(starts, dtype=jnp.int32)


def _segment_key(root_key: jax.Array, segment_index: jax.Array) -> jax.Array:
  """Uses the original key for segment zero and fold-in streams thereafter."""
  return jax.lax.cond(
      segment_index == 0,
      lambda _: root_key,
      lambda index: jax.random.fold_in(root_key, index),
      segment_index,
  )


def _validate_state(state: SegmentedBandInvDPAdamWState, plan: SegmentedPlan) -> None:
  if not isinstance(state, SegmentedBandInvDPAdamWState):
    raise TypeError("state must be a SegmentedBandInvDPAdamWState")
  if not isinstance(state.noise_state, BandInvMFNoiseState):
    raise TypeError("state.noise_state must be a BandInvMFNoiseState")
  bandwidth = plan.runtime_bandwidth or max(strategy.bandwidth for strategy in plan.strategies)
  if state.noise_state.bandwidth != bandwidth:
    raise ValueError("state.noise_state has the wrong runtime bandwidth")
  if not isinstance(state.step, jax.core.Tracer):
    step = int(jnp.asarray(state.step))
    if not 0 <= step <= plan.horizon:
      raise ValueError("state.step must lie within the plan horizon")
    segment_start = int(jnp.asarray(state.segment_start))
    if segment_start > step:
      raise ValueError("segment_start must not exceed global step")


def init_segmented_bandinv_dpadamw_state(
    params: PyTree,
    plan: SegmentedPlan,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> SegmentedBandInvDPAdamWState:
  """Initializes AdamW once and an empty FIR state for segment zero."""
  if not isinstance(plan, SegmentedPlan):
    raise TypeError("plan must be a SegmentedPlan")
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax GradientTransformation")
  bandwidth = plan.runtime_bandwidth or max(strategy.bandwidth for strategy in plan.strategies)
  return SegmentedBandInvDPAdamWState(
      params=params,
      optimizer_state=optimizer.init(params),
      noise_state=init_bandinv_noise_state(params, bandwidth),
      rng_root_key=rng_key,
      rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
      segment_index=jnp.array(0, dtype=jnp.int32),
      segment_start=jnp.array(0, dtype=jnp.int32),
  )


def begin_segment(
    state: SegmentedBandInvDPAdamWState,
    logical_step: int,
    plan: SegmentedPlan,
) -> SegmentedBandInvDPAdamWState:
  """Eager boundary helper, useful for callers that orchestrate segments."""
  _validate_state(state, plan)
  if not isinstance(logical_step, Integral) or isinstance(logical_step, bool):
    raise ValueError("logical_step must be an integer")
  if logical_step not in segment_starts(plan):
    return state
  boundaries = [0]
  for length in plan.block_lengths[:-1]:
    boundaries.append(boundaries[-1] + length)
  segment_index = boundaries.index(int(logical_step))
  bandwidth = plan.runtime_bandwidth or max(strategy.bandwidth for strategy in plan.strategies)
  return replace(
      state,
      noise_state=init_bandinv_noise_state(state.params, bandwidth),
      rng_key=jax.random.fold_in(state.rng_root_key, segment_index),
      segment_index=jnp.array(segment_index, dtype=state.segment_index.dtype),
      segment_start=jnp.array(logical_step, dtype=state.segment_start.dtype),
  )


def make_segmented_bandinv_dpadamw_train_step(
    loss_fn: Callable[..., Any],
    plan: SegmentedPlan,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    microbatch_size: int | None = None,
) -> tuple[Callable[[SegmentedBandInvDPAdamWState, Any], SegmentedBandInvDPAdamWState], optax.GradientTransformation]:
  """Builds the same clipped-gradient -> noise -> standard AdamW pipeline."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  if not isinstance(plan, SegmentedPlan) or not isinstance(plan.calibration, PrivacyCalibration):
    raise TypeError("plan must contain a PrivacyCalibration")
  schedule = coefficient_schedule(plan)
  segment_ids, segment_starts_array = _segment_start_schedule(plan)
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay,
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=plan.calibration.clip_norm,
      normalize_by=plan.calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )

  def train_step(
      state: SegmentedBandInvDPAdamWState, batch: Any
  ) -> SegmentedBandInvDPAdamWState:
    _validate_state(state, plan)
    target_segment = segment_ids[state.step]
    target_start = segment_starts_array[target_segment]
    fresh_noise_state = init_bandinv_noise_state(
        state.params,
        plan.runtime_bandwidth
        or max(strategy.bandwidth for strategy in plan.strategies),
    )
    transitioned = SegmentedBandInvDPAdamWState(
        params=state.params,
        optimizer_state=state.optimizer_state,
        noise_state=fresh_noise_state,
        rng_root_key=state.rng_root_key,
        rng_key=_segment_key(state.rng_root_key, target_segment),
        step=state.step,
        segment_index=target_segment,
        segment_start=target_start,
    )
    state = jax.lax.cond(
        state.segment_index == target_segment,
        lambda _: state,
        lambda _: transitioned,
        operand=None,
    )
    clipped_grad = clipped_query(state.params, batch)
    correlated_noise, new_noise_state, new_key = sample_bandinv_noise(
        state.rng_key,
        state.noise_state,
        schedule[state.step],
        plan.calibration.iid_noise_std,
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation,
        clipped_grad,
        correlated_noise,
    )
    # Keep this call structurally identical to the continuous naive trainer.
    updates, new_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    return SegmentedBandInvDPAdamWState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=new_optimizer_state,
        noise_state=new_noise_state,
        rng_root_key=state.rng_root_key,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
        segment_index=state.segment_index,
        segment_start=state.segment_start,
    )

  return train_step, optimizer


# Concise aliases make the public algorithm name easier to discover while the
# longer names remain explicit in checkpoints and driver wiring.
NonAmplifiedSegmentedBandInvDPAdamWState = SegmentedBandInvDPAdamWState
init_nonamplified_segmented_bandinv_dpadamw_state = init_segmented_bandinv_dpadamw_state
make_nonamplified_segmented_bandinv_dpadamw_train_step = make_segmented_bandinv_dpadamw_train_step
make_segmented_train_step = make_segmented_bandinv_dpadamw_train_step


def reset_noise_state(params: PyTree, *, bandwidth: int) -> BandInvMFNoiseState:
  """Returns an empty FIR state without touching any AdamW state."""
  return init_bandinv_noise_state(params, bandwidth)


__all__ = [
    "SegmentedBandInvDPAdamWState",
    "NonAmplifiedSegmentedBandInvDPAdamWState",
    "SegmentedPlan",
    "begin_segment",
    "block_lengths",
    "coefficient_schedule",
    "fit_segmented_plan",
    "global_segmented_sensitivity_squared",
    "segmented_prefix_sensitivity_squared",
    "epsilon_spent_for_segmented_prefix",
    "init_segmented_bandinv_dpadamw_state",
    "init_nonamplified_segmented_bandinv_dpadamw_state",
    "make_segmented_bandinv_dpadamw_train_step",
    "make_nonamplified_segmented_bandinv_dpadamw_train_step",
    "make_segmented_train_step",
    "reset_noise_state",
    "segment_starts",
]
