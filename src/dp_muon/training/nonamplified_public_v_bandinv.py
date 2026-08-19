"""Segmented Public-(V) + Frozen AdamW + BandInvMF private training math."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import toeplitz
from opacus.accountants.analysis import gdp

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    fit_bandinv_strategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.optim import (
    PublicVAdamW,
    PublicVAdamWState,
    public_v_adamw_segment_workload_matrix,
)
from dp_muon.privacy import PrivacyCalibration, make_clipped_gradient_query

from .public_v import PublicVEstimator, public_preconditioner_rms


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PublicVBandInvAdamWState:
  """Checkpointable public, optimizer, and current segment noise state."""

  params: PyTree
  optimizer_state: PublicVAdamWState
  noise_state: BandInvMFNoiseState
  noise_root_key: jax.Array
  rng_key: jax.Array
  step: jax.Array
  segment_index: jax.Array
  segment_start: jax.Array
  segment_end: jax.Array
  segment_min_sep: jax.Array
  segment_max_participations: jax.Array
  noising_coef: jax.Array
  iid_noise_std: jax.Array
  segment_sensitivity_squared: jax.Array

  def tree_flatten(self):
    return (
        self.params,
        self.optimizer_state,
        self.noise_state,
        self.noise_root_key,
        self.rng_key,
        self.step,
        self.segment_index,
        self.segment_start,
        self.segment_end,
        self.segment_min_sep,
        self.segment_max_participations,
        self.noising_coef,
        self.iid_noise_std,
        self.segment_sensitivity_squared,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]):
    del aux_data
    return cls(*children)


@dataclass(frozen=True)
class PublicVSegmentInfo:
  """Non-private diagnostics for one freshly configured segment."""

  segment_index: int
  start_step: int
  length: int
  workload_matrix: np.ndarray
  strategy: BandInvMFStrategy
  preconditioner_rms: float
  iid_noise_std: float
  public_v_num_examples: int


def init_public_v_bandinv_adamw_state(
    params: PyTree,
    *,
    optimizer: PublicVAdamW,
    noise_root_key: jax.Array,
    bandwidth: int,
) -> PublicVBandInvAdamWState:
  """Initializes persistent state before the first public estimation pass."""
  if not isinstance(bandwidth, Integral) or bandwidth < 1:
    raise ValueError("bandwidth must be positive")
  return PublicVBandInvAdamWState(
      params=params,
      optimizer_state=optimizer.init(params),
      noise_state=init_bandinv_noise_state(params, int(bandwidth)),
      noise_root_key=noise_root_key,
      rng_key=jax.random.fold_in(noise_root_key, 0),
      step=jnp.array(0, dtype=jnp.int32),
      segment_index=jnp.array(-1, dtype=jnp.int32),
      segment_start=jnp.array(0, dtype=jnp.int32),
      segment_end=jnp.array(0, dtype=jnp.int32),
      segment_min_sep=jnp.array(1, dtype=jnp.int32),
      segment_max_participations=jnp.array(1, dtype=jnp.int32),
      noising_coef=jnp.zeros((int(bandwidth),), dtype=jnp.float32),
      iid_noise_std=jnp.array(0.0, dtype=jnp.float32),
      segment_sensitivity_squared=jnp.array(1.0, dtype=jnp.float32),
  )


def begin_public_v_segment(
    state: PublicVBandInvAdamWState,
    public_batches: Iterable[Any],
    *,
    estimator: PublicVEstimator,
    optimizer: PublicVAdamW,
    segment_index: int,
    segment_length: int,
    global_min_sep: int,
    bandwidth: int,
    num_segments: int,
    global_noise_multiplier: float,
    query_sensitivity: float,
    learning_rates: float | np.ndarray,
    reduction: str,
    max_optimizer_steps: int,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
    public_v_batch_estimate: (
        Callable[[PyTree, Any], tuple[PyTree, jax.Array]] | None
    ) = None,
) -> tuple[PublicVBandInvAdamWState, PublicVSegmentInfo]:
  """Directly estimates public V, fits A_time, freezes V, and resets noise."""
  start_step = int(state.step)
  if segment_length < 1 or num_segments < 1 or global_min_sep < 1:
    raise ValueError("segment and participation dimensions must be positive")
  if segment_index < 0 or segment_index >= num_segments:
    raise ValueError("segment_index is outside the configured run")
  if segment_index != int(state.segment_index) + 1:
    raise ValueError("segments must begin in consecutive order")
  if start_step != int(state.segment_end):
    raise ValueError("the previous segment must finish before the next begins")

  v_hat, public_v_num_examples = estimator.estimate_with_count(
      state.params,
      public_batches,
      batch_estimate=public_v_batch_estimate,
  )
  optimizer_state = optimizer.set_public_v(state.optimizer_state, v_hat, state.params)
  # Frozen V acts on the parameter axis; this RMS is diagnostic only and must
  # not affect the temporal BandInvMF workload or its fitted strategy.
  preconditioner_rms = float(public_preconditioner_rms(v_hat, optimizer.eps))
  workload = public_v_adamw_segment_workload_matrix(
      segment_length,
      optimizer.beta1,
      learning_rates,
      optimizer.weight_decay,
      first_moment_start_step=int(optimizer_state.count),
  )
  local_max_participations = 1 + (segment_length - 1) // global_min_sep
  local_bandwidth = min(int(bandwidth), segment_length)
  strategy = fit_strategy(
      segment_length,
      local_bandwidth,
      int(global_min_sep),
      max_participations=local_max_participations,
      workload_matrix=workload,
      max_optimizer_steps=max_optimizer_steps,
      reduction=reduction,
  )

  # Each independent segment receives global_mu^2 / num_segments.  Its noise
  # scale therefore includes sqrt(num_segments * segment_sensitivity_squared).
  sensitivity_squared = float(strategy.sensitivity_squared)
  iid_noise_std = (
      float(global_noise_multiplier)
      * float(query_sensitivity)
      * math.sqrt(num_segments * sensitivity_squared)
  )
  rng_key = jax.random.fold_in(state.noise_root_key, segment_index)
  new_state = PublicVBandInvAdamWState(
      params=state.params,
      optimizer_state=optimizer_state,
      noise_state=init_bandinv_noise_state(state.params, strategy.bandwidth),
      noise_root_key=state.noise_root_key,
      rng_key=rng_key,
      step=state.step,
      segment_index=jnp.array(segment_index, dtype=jnp.int32),
      segment_start=jnp.array(start_step, dtype=jnp.int32),
      segment_end=jnp.array(start_step + segment_length, dtype=jnp.int32),
      segment_min_sep=jnp.array(global_min_sep, dtype=jnp.int32),
      segment_max_participations=jnp.array(local_max_participations, dtype=jnp.int32),
      noising_coef=jnp.asarray(strategy.noising_coef),
      iid_noise_std=jnp.asarray(iid_noise_std),
      segment_sensitivity_squared=jnp.asarray(strategy.sensitivity_squared),
  )
  return new_state, PublicVSegmentInfo(
      segment_index=segment_index,
      start_step=start_step,
      length=segment_length,
      workload_matrix=np.asarray(workload),
      strategy=strategy,
      preconditioner_rms=preconditioner_rms,
      iid_noise_std=iid_noise_std,
      public_v_num_examples=public_v_num_examples,
  )


def make_public_v_bandinv_adamw_train_step(
    loss_fn: Callable[..., Any],
    clipping_calibration: PrivacyCalibration,
    optimizer: PublicVAdamW,
    *,
    microbatch_size: int | None = None,
) -> Callable[[PublicVBandInvAdamWState, Any], PublicVBandInvAdamWState]:
  """Builds global clip -> BandInvMF noise -> Frozen-(V) AdamW."""
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=clipping_calibration.clip_norm,
      normalize_by=clipping_calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )

  def train_step(
      state: PublicVBandInvAdamWState, batch: Any
  ) -> PublicVBandInvAdamWState:
    # V is intentionally absent from this query: private per-example gradients
    # retain the repository's original global L2 clipping geometry.
    clipped_grad = clipped_query(state.params, batch)
    correlated_noise, noise_state, rng_key = sample_bandinv_noise(
        state.rng_key,
        state.noise_state,
        state.noising_coef,
        state.iid_noise_std,
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, noise: gradient + noise,
        clipped_grad,
        correlated_noise,
    )
    updates, optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    params = jax.tree_util.tree_map(
        lambda parameter, update: parameter + update,
        state.params,
        updates,
    )
    return PublicVBandInvAdamWState(
        params=params,
        optimizer_state=optimizer_state,
        noise_state=noise_state,
        noise_root_key=state.noise_root_key,
        rng_key=rng_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
        segment_index=state.segment_index,
        segment_start=state.segment_start,
        segment_end=state.segment_end,
        segment_min_sep=state.segment_min_sep,
        segment_max_participations=state.segment_max_participations,
        noising_coef=state.noising_coef,
        iid_noise_std=state.iid_noise_std,
        segment_sensitivity_squared=state.segment_sensitivity_squared,
    )

  return train_step


class SegmentedBandInvPrivacyAccountant:
  """GDP prefix accounting for independent, equally allocated segments."""

  def __init__(self, *, num_segments: int, global_mu: float, delta: float):
    if num_segments < 1 or global_mu <= 0 or not 0 < delta < 1:
      raise ValueError("invalid segmented accountant configuration")
    self.num_segments = int(num_segments)
    self.global_mu = float(global_mu)
    self.delta = float(delta)
    self._current: PublicVBandInvAdamWState | None = None

  def set_current_state(self, state: PublicVBandInvAdamWState) -> None:
    self._current = state

  def epsilon_spent(self, prefix_steps: int) -> float:
    if self._current is None:
      raise ValueError("current segment mechanism has not been registered")
    state = self._current
    segment_index = int(state.segment_index)
    start, end = int(state.segment_start), int(state.segment_end)
    if not start < prefix_steps <= end:
      raise ValueError("prefix_steps is outside the current segment")
    local_prefix = prefix_steps - start
    segment_length = end - start
    if local_prefix == segment_length:
      prefix_sensitivity_squared = float(state.segment_sensitivity_squared)
    else:
      prefix_sensitivity_squared = float(
          toeplitz.compute_banded_inverse_sensitivity_squared(
              n=local_prefix,
              noising_coef=state.noising_coef[:local_prefix],
              min_sep=int(state.segment_min_sep),
              max_participations=int(state.segment_max_participations),
          )
      )
    fraction = prefix_sensitivity_squared / float(state.segment_sensitivity_squared)
    mu_squared = (
        self.global_mu**2 / self.num_segments * (segment_index + fraction)
    )
    return float(gdp.eps_from_mu(mu=math.sqrt(mu_squared), delta=self.delta))


__all__ = [
    "PublicVBandInvAdamWState",
    "PublicVSegmentInfo",
    "SegmentedBandInvPrivacyAccountant",
    "begin_public_v_segment",
    "init_public_v_bandinv_adamw_state",
    "make_public_v_bandinv_adamw_train_step",
]
