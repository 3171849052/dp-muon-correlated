"""End-to-end non-amplified linear BandInvMF training step.

This module composes the project's clipping, streaming BandInvMF noise, and
EMA-then-Nesterov components.  The sole momentum state receives the *noisy*
query; no clean parallel momentum path is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.optim import (
    MuonNesterovState,
    fixed_lr_nesterov_trajectory_workload_coef,
    init_muon_nesterov_state,
    muon_nesterov_step,
)
from dp_muon.privacy import (
    ParticipationSpec,
    PrivacyCalibration,
    calibrate_nonamplified_bandinv,
    make_clipped_gradient_query,
    validate_participation_spec_against_strategy,
)


PyTree = Any

# Strategy artefacts commonly originate in float32, so comparisons should
# tolerate that representation while still rejecting a differently fitted run.
_RTOL = 1e-5
_ATOL = 1e-7


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedBandInvState:
  """Checkpointable state for the non-amplified linear trainer.

  ``nesterov_state.step`` and ``noise_state.step`` advance together.  There is
  deliberately no top-level step counter, clean momentum state, or transient
  query/noise diagnostic in this state.
  """

  params: PyTree
  nesterov_state: MuonNesterovState
  noise_state: BandInvMFNoiseState
  rng_key: jax.Array

  def tree_flatten(self):
    return (self.params, self.nesterov_state, self.noise_state, self.rng_key), None

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: None,
      children: tuple[PyTree, MuonNesterovState, BandInvMFNoiseState, jax.Array],
  ):
    params, nesterov_state, noise_state, rng_key = children
    return cls(
        params=params,
        nesterov_state=nesterov_state,
        noise_state=noise_state,
        rng_key=rng_key,
    )


def _concrete_finite_scalar(value: object, name: str) -> float:
  """Returns a public scalar configuration value, rejecting dynamic inputs."""
  array = np.asarray(value)
  if (
      array.ndim != 0
      or not np.issubdtype(array.dtype, np.number)
      or np.issubdtype(array.dtype, np.complexfloating)
  ):
    raise ValueError(f"{name} must be a finite scalar")
  try:
    result = float(array)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be a finite scalar") from error
  if not math.isfinite(result):
    raise ValueError(f"{name} must be a finite scalar")
  return result


def _concrete_finite_vector(value: object, name: str) -> np.ndarray:
  array = np.asarray(value)
  if (
      array.ndim != 1
      or array.size == 0
      or not np.issubdtype(array.dtype, np.number)
      or np.issubdtype(array.dtype, np.complexfloating)
  ):
    raise ValueError(f"{name} must be a non-empty one-dimensional finite array")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be a non-empty one-dimensional finite array")
  return array


def _require_strategy_bandwidth(strategy: BandInvMFStrategy) -> int:
  """Checks that the fitted coefficient metadata is internally coherent."""
  if not isinstance(strategy.horizon, Integral) or strategy.horizon < 1:
    raise ValueError("strategy.horizon must be a positive integer")
  if not isinstance(strategy.bandwidth, Integral) or strategy.bandwidth < 1:
    raise ValueError("strategy.bandwidth must be a positive integer")
  noising_coef = _concrete_finite_vector(strategy.noising_coef, "strategy.noising_coef")
  if not np.issubdtype(noising_coef.dtype, np.floating):
    raise ValueError("strategy.noising_coef must be a floating array")
  if noising_coef.size != strategy.bandwidth:
    raise ValueError("strategy.bandwidth must equal len(strategy.noising_coef)")
  if strategy.bandwidth > strategy.horizon:
    raise ValueError("strategy.bandwidth must not exceed strategy.horizon")
  return int(strategy.bandwidth)


def _steps_are_equal(left: jax.Array, right: jax.Array) -> bool | None:
  """Checks eager checkpoint metadata, leaving traced steps to JAX execution."""
  left_array, right_array = jnp.asarray(left), jnp.asarray(right)
  if isinstance(left_array, jax.core.Tracer) or isinstance(right_array, jax.core.Tracer):
    return None
  return bool(jnp.array_equal(left_array, right_array))


def _validate_state(state: NonAmplifiedBandInvState) -> None:
  if not isinstance(state, NonAmplifiedBandInvState):
    raise TypeError("state must be a NonAmplifiedBandInvState")
  if not isinstance(state.nesterov_state, MuonNesterovState):
    raise TypeError("state.nesterov_state must be a MuonNesterovState")
  if not isinstance(state.noise_state, BandInvMFNoiseState):
    raise TypeError("state.noise_state must be a BandInvMFNoiseState")
  equal_steps = _steps_are_equal(state.nesterov_state.step, state.noise_state.step)
  if equal_steps is False:
    raise ValueError("nesterov_state.step must equal noise_state.step")


def validate_nonamplified_bandinv_setup(
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    momentum: float,
    learning_rate: float,
) -> None:
  """Fails fast unless runtime training settings match fitted DP artefacts.

  The validation deliberately reuses the M1 workload/sensitivity helpers and
  M4 participation validator instead of duplicating their mathematics.
  """
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  if not isinstance(calibration, PrivacyCalibration):
    raise TypeError("calibration must be a PrivacyCalibration")
  if not isinstance(participation_spec, ParticipationSpec):
    raise TypeError("participation_spec must be a ParticipationSpec")

  _require_strategy_bandwidth(strategy)
  expected_workload = np.asarray(
      fixed_lr_nesterov_trajectory_workload_coef(
          strategy.horizon, momentum, learning_rate
      )
  )
  fitted_workload = _concrete_finite_vector(
      strategy.workload_coef, "strategy.workload_coef"
  )
  if fitted_workload.shape != expected_workload.shape or not np.allclose(
      fitted_workload, expected_workload, rtol=_RTOL, atol=_ATOL
  ):
    raise ValueError(
        "strategy.workload_coef must equal the fixed-LR Nesterov trajectory "
        "workload for runtime momentum and learning_rate"
    )

  strategy_sensitivity_squared = _concrete_finite_scalar(
      strategy.sensitivity_squared, "strategy.sensitivity_squared"
  )
  if strategy_sensitivity_squared <= 0:
    raise ValueError("strategy.sensitivity_squared must be positive")
  expected_calibration = calibrate_nonamplified_bandinv(
      epsilon=calibration.epsilon,
      delta=calibration.delta,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      adjacency=calibration.adjacency,
      sensitivity_squared=strategy_sensitivity_squared,
  )
  for field in (
      "query_sensitivity",
      "matrix_sensitivity",
      "total_sensitivity",
      "mu",
      "noise_multiplier",
      "iid_noise_std",
  ):
    actual = _concrete_finite_scalar(
        getattr(calibration, field), f"calibration.{field}"
    )
    expected = _concrete_finite_scalar(
        getattr(expected_calibration, field), f"expected calibration.{field}"
    )
    if not np.isclose(actual, expected, rtol=_RTOL, atol=_ATOL):
      raise ValueError(
          f"calibration.{field} must match calibrate_nonamplified_bandinv"
      )

  validate_participation_spec_against_strategy(participation_spec, strategy)


def init_nonamplified_bandinv_state(
    params: PyTree, strategy: BandInvMFStrategy, rng_key: jax.Array
) -> NonAmplifiedBandInvState:
  """Initializes aligned Nesterov and BandInvMF states from ``params``."""
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  bandwidth = _require_strategy_bandwidth(strategy)
  nesterov_state = init_muon_nesterov_state(params)
  noise_state = init_bandinv_noise_state(params, bandwidth)
  return NonAmplifiedBandInvState(
      params=params,
      nesterov_state=nesterov_state,
      noise_state=noise_state,
      rng_key=rng_key,
  )


def make_nonamplified_bandinv_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    momentum: float,
    learning_rate: float,
) -> Callable[[NonAmplifiedBandInvState, Any], NonAmplifiedBandInvState]:
  """Builds a JIT-compatible private step for ``loss_fn(params, batch)``."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  validate_nonamplified_bandinv_setup(
      strategy, calibration, participation_spec, momentum, learning_rate
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
  )

  def train_step(
      state: NonAmplifiedBandInvState, batch: Any
  ) -> NonAmplifiedBandInvState:
    _validate_state(state)
    query = clipped_query(state.params, batch)
    # M2/M5 validate concrete values at eager API boundaries and intentionally
    # accept traced configuration arguments.  Make each validated setup value
    # a zero-valued dependency of the streaming state, so a caller may wrap
    # this closure directly in ``jax.jit`` without changing any mathematics.
    step = state.noise_state.step
    noising_coef = jnp.asarray(strategy.noising_coef)
    runtime_noising_coef = noising_coef + (
        jnp.asarray(step, dtype=noising_coef.dtype) * jnp.zeros_like(noising_coef)
    )
    iid_noise_std = jnp.asarray(calibration.iid_noise_std)
    runtime_iid_noise_std = iid_noise_std + (
        jnp.asarray(step, dtype=iid_noise_std.dtype) * jnp.zeros_like(iid_noise_std)
    )
    momentum_value = jnp.asarray(momentum)
    runtime_momentum = momentum_value + (
        jnp.asarray(step, dtype=momentum_value.dtype) * jnp.zeros_like(momentum_value)
    )
    query_noise, new_noise_state, new_key = sample_bandinv_noise(
        state.rng_key,
        state.noise_state,
        runtime_noising_coef,
        runtime_iid_noise_std,
    )
    private_query = jax.tree_util.tree_map(
        lambda query_leaf, noise_leaf: query_leaf + noise_leaf,
        query,
        query_noise,
    )
    update, new_nesterov_state = muon_nesterov_step(
        state.nesterov_state, private_query, runtime_momentum
    )
    new_params = jax.tree_util.tree_map(
        lambda parameter, update_leaf: parameter - learning_rate * update_leaf,
        state.params,
        update,
    )
    return NonAmplifiedBandInvState(
        params=new_params,
        nesterov_state=new_nesterov_state,
        noise_state=new_noise_state,
        rng_key=new_key,
    )

  return train_step


__all__ = [
    "NonAmplifiedBandInvState",
    "init_nonamplified_bandinv_state",
    "make_nonamplified_bandinv_train_step",
    "validate_nonamplified_bandinv_setup",
]
