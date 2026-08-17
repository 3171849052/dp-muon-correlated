"""BandInvMF-correlated DP-AdamW: one global private gradient, all-parameter AdamW.

The data flow is::

  per-example grad
  -> global L2 clipping
  -> normalize_by = logical_batch_size
  -> clipped_grad
  -> one full-tree BandInvMF correlated noise  (the sole private mechanism)
  -> private_grad = clipped_grad + correlated_noise
  -> AdamW(all params)
  -> apply_updates

BandInvMF only generates the correlated private gradient; AdamW is a pure
post-processing step.  This is intentionally a naive prefix-sum BandInvMF
baseline with no AdamW-aware workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.privacy import (
    ParticipationSpec,
    PrivacyCalibration,
    make_clipped_gradient_query,
)

from .nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer
from .nonamplified_linear import validate_nonamplified_bandinv_privacy_setup


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedBandInvDPAdamWState:
  """Checkpointable state for one private BandInvMF DP-AdamW transcript.

  ``state.step == noise_state.step`` is enforced at the eager API boundary.
  """

  params: PyTree
  optimizer_state: Any
  noise_state: BandInvMFNoiseState
  rng_key: jax.Array
  step: jax.Array

  def tree_flatten(self):
    return (
        self.params,
        self.optimizer_state,
        self.noise_state,
        self.rng_key,
        self.step,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]):
    del aux_data
    params, optimizer_state, noise_state, rng_key, step = children
    return cls(
        params=params,
        optimizer_state=optimizer_state,
        noise_state=noise_state,
        rng_key=rng_key,
        step=step,
    )


def _steps_are_equal(left: jax.Array, right: jax.Array) -> bool | None:
  left_array, right_array = jnp.asarray(left), jnp.asarray(right)
  if isinstance(left_array, jax.core.Tracer) or isinstance(right_array, jax.core.Tracer):
    return None
  return bool(jnp.array_equal(left_array, right_array))


def _validate_state(state: NonAmplifiedBandInvDPAdamWState) -> None:
  if not isinstance(state, NonAmplifiedBandInvDPAdamWState):
    raise TypeError("state must be a NonAmplifiedBandInvDPAdamWState")
  if not isinstance(state.noise_state, BandInvMFNoiseState):
    raise TypeError("state.noise_state must be a BandInvMFNoiseState")
  if _steps_are_equal(state.step, state.noise_state.step) is False:
    raise ValueError("state.step must equal noise_state.step")


def init_nonamplified_bandinv_dpadamw_state(
    params: PyTree,
    strategy: BandInvMFStrategy,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> NonAmplifiedBandInvDPAdamWState:
  """Initialises the all-parameter AdamW state and full-tree BandInvMF noise."""
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax GradientTransformation")
  return NonAmplifiedBandInvDPAdamWState(
      params=params,
      optimizer_state=optimizer.init(params),
      noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
  )


def make_nonamplified_bandinv_dpadamw_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    microbatch_size: int | None = None,
) -> tuple[
    Callable[[NonAmplifiedBandInvDPAdamWState, Any], NonAmplifiedBandInvDPAdamWState],
    optax.GradientTransformation,
]:
  """Builds ``global clip -> full-tree BandInvMF -> AdamW -> apply_updates``.

  The returned optimizer is required only to initialise a state.  The private
  gradient PyTree is passed to AdamW as-is; no parameter group is clipped,
  calibrated, or noised separately.
  """
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  validate_nonamplified_bandinv_privacy_setup(
      strategy, calibration, participation_spec
  )
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay,
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )

  def train_step(
      state: NonAmplifiedBandInvDPAdamWState, batch: Any
  ) -> NonAmplifiedBandInvDPAdamWState:
    _validate_state(state)
    clipped_grad = clipped_query(state.params, batch)
    # This is the sole private mechanism invocation for the logical batch.
    step = state.noise_state.step
    noising_coef = jnp.asarray(strategy.noising_coef)
    runtime_noising_coef = noising_coef + (
        jnp.asarray(step, dtype=noising_coef.dtype) * jnp.zeros_like(noising_coef)
    )
    iid_noise_std = jnp.asarray(calibration.iid_noise_std)
    runtime_iid_noise_std = iid_noise_std + (
        jnp.asarray(step, dtype=iid_noise_std.dtype) * jnp.zeros_like(iid_noise_std)
    )
    correlated_noise, new_noise_state, new_key = sample_bandinv_noise(
        state.rng_key,
        state.noise_state,
        runtime_noising_coef,
        runtime_iid_noise_std,
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation,
        clipped_grad,
        correlated_noise,
    )
    updates, new_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    return NonAmplifiedBandInvDPAdamWState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=new_optimizer_state,
        noise_state=new_noise_state,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    )

  return train_step, optimizer


__all__ = [
    "NonAmplifiedBandInvDPAdamWState",
    "init_nonamplified_bandinv_dpadamw_state",
    "make_nonamplified_bandinv_dpadamw_train_step",
]