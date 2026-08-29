"""BandInvMF-correlated DP-AdamW with Scale-Then-Privatize (STP).

The per-step data flow is::

  previous private Adam V
  -> elementwise STP scale
  -> per-example scaled gradient
  -> global L2 clipping and normalize_by
  -> one full-tree BandInvMF correlated noise
  -> private scaled gradient
  -> undo STP scale
  -> explicit private AdamW (m, v, bias correction)

The BandInvMF mechanism remains the same naive prefix-sum mechanism as the
ordinary correlated DP-AdamW baseline.  The scale is computed only from the
already-private optimizer state at the beginning of each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.optim import STPAdamW, STPAdamWState
from dp_muon.privacy import (
    ParticipationSpec,
    PrivacyCalibration,
    make_clipped_gradient_query,
)

from .nonamplified_linear import validate_nonamplified_bandinv_privacy_setup


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedBandInvSTPDPAdamWState:
  """Checkpointable STP AdamW and BandInvMF transcript state."""

  params: PyTree
  optimizer_state: STPAdamWState
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


def _validate_state(state: NonAmplifiedBandInvSTPDPAdamWState) -> None:
  if not isinstance(state, NonAmplifiedBandInvSTPDPAdamWState):
    raise TypeError("state must be a NonAmplifiedBandInvSTPDPAdamWState")
  if not isinstance(state.optimizer_state, STPAdamWState):
    raise TypeError("state.optimizer_state must be an STPAdamWState")
  if not isinstance(state.noise_state, BandInvMFNoiseState):
    raise TypeError("state.noise_state must be a BandInvMFNoiseState")
  if _steps_are_equal(state.step, state.noise_state.step) is False:
    raise ValueError("state.step must equal noise_state.step")
  if _steps_are_equal(state.step, state.optimizer_state.count) is False:
    raise ValueError("state.step must equal optimizer_state.count")


def init_nonamplified_bandinv_stp_dpadamw_state(
    params: PyTree,
    strategy: BandInvMFStrategy,
    rng_key: jax.Array,
    optimizer: STPAdamW,
) -> NonAmplifiedBandInvSTPDPAdamWState:
  """Initializes explicit AdamW ``m/v/count`` and BandInvMF noise state."""
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  if not isinstance(optimizer, STPAdamW):
    raise TypeError("optimizer must be an STPAdamW")
  return NonAmplifiedBandInvSTPDPAdamWState(
      params=params,
      optimizer_state=optimizer.init(params),
      noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
  )


def make_nonamplified_bandinv_stp_dpadamw_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    scale_eps: float = 1e-8,
    weight_decay: float = 0.0,
    microbatch_size: int | None = None,
) -> tuple[
    Callable[
        [NonAmplifiedBandInvSTPDPAdamWState, Any],
        NonAmplifiedBandInvSTPDPAdamWState,
    ],
    STPAdamW,
]:
  """Builds the complete STP scaled-query and correlated AdamW update."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  validate_nonamplified_bandinv_privacy_setup(
      strategy, calibration, participation_spec
  )
  optimizer = STPAdamW(
      learning_rate=learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      scale_eps=scale_eps,
      weight_decay=weight_decay,
  )

  def train_step(
      state: NonAmplifiedBandInvSTPDPAdamWState, batch: Any
  ) -> NonAmplifiedBandInvSTPDPAdamWState:
    _validate_state(state)
    scale = optimizer.scale(state.optimizer_state)
    # clipped_grad delegates all clipping mathematics to jax_privacy.  The
    # factory is created here so the dynamic, state-derived scale is captured
    # by the pre_clipping_transform during eager execution and JIT tracing.
    clipped_scaled_query = make_clipped_gradient_query(
        loss_fn,
        clip_norm=calibration.clip_norm,
        normalize_by=calibration.normalize_by,
        batch_argnums=1,
        keep_batch_dim=True,
        microbatch_size=microbatch_size,
        pre_clipping_transform=lambda gradient: jax.tree_util.tree_map(
            lambda factor, value: factor * value, scale, gradient
        ),
    )
    clipped_scaled_grad = clipped_scaled_query(state.params, batch)

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
    private_scaled_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation,
        clipped_scaled_grad,
        correlated_noise,
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, factor: gradient / factor,
        private_scaled_grad,
        scale,
    )
    updates, new_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    return NonAmplifiedBandInvSTPDPAdamWState(
        params=jax.tree_util.tree_map(
            lambda parameter, update: parameter + update,
            state.params,
            updates,
        ),
        optimizer_state=new_optimizer_state,
        noise_state=new_noise_state,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    )

  return train_step, optimizer


__all__ = [
    "NonAmplifiedBandInvSTPDPAdamWState",
    "init_nonamplified_bandinv_stp_dpadamw_state",
    "make_nonamplified_bandinv_stp_dpadamw_train_step",
]
