"""BandInvMF-correlated DP-Muon with the standard Muon/AdamW partition.

One global clipped gradient query is noised as a complete parameter PyTree.
Only after that single private mechanism does Optax route leaves to Muon or
AdamW.  This is intentionally not a nonlinear-aware BandInvMF optimizer.
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

from .nonamplified_dpmuon import make_nonamplified_dpmuon_optimizer
from .nonamplified_linear import validate_nonamplified_bandinv_privacy_setup


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedBandInvDPMuonState:
  """Checkpointable state for one private BandInvMF transcript."""

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


def _validate_state(state: NonAmplifiedBandInvDPMuonState) -> None:
  if not isinstance(state, NonAmplifiedBandInvDPMuonState):
    raise TypeError("state must be a NonAmplifiedBandInvDPMuonState")
  if not isinstance(state.noise_state, BandInvMFNoiseState):
    raise TypeError("state.noise_state must be a BandInvMFNoiseState")
  if _steps_are_equal(state.step, state.noise_state.step) is False:
    raise ValueError("state.step must equal noise_state.step")


def init_nonamplified_bandinv_dpmuon_state(
    params: PyTree,
    strategy: BandInvMFStrategy,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> NonAmplifiedBandInvDPMuonState:
  """Initializes the one partitioned optimizer and full-tree noise state."""
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax.GradientTransformation")
  return NonAmplifiedBandInvDPMuonState(
      params=params,
      optimizer_state=optimizer.init(params),
      noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
  )


def make_nonamplified_bandinv_dpmuon_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    muon_learning_rate: float,
    muon_weight_decay: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    adamw_learning_rate: float,
    adamw_beta1: float = 0.9,
    adamw_beta2: float = 0.999,
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.0,
    microbatch_size: int | None = None,
    use_bf16_ns: bool = True,
) -> tuple[
    Callable[[NonAmplifiedBandInvDPMuonState, Any], NonAmplifiedBandInvDPMuonState],
    optax.GradientTransformation,
]:
  """Builds ``global clip -> full-tree BandInvMF -> Muon/AdamW partition``."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  validate_nonamplified_bandinv_privacy_setup(
      strategy, calibration, participation_spec
  )
  optimizer = make_nonamplified_dpmuon_optimizer(
      muon_learning_rate=muon_learning_rate,
      muon_weight_decay=muon_weight_decay,
      momentum=momentum,
      ns_steps=ns_steps,
      consistent_rms=consistent_rms,
      adamw_learning_rate=adamw_learning_rate,
      adamw_beta1=adamw_beta1,
      adamw_beta2=adamw_beta2,
      adamw_eps=adamw_eps,
      adamw_weight_decay=adamw_weight_decay,
      use_bf16_ns=use_bf16_ns,
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
      state: NonAmplifiedBandInvDPMuonState, batch: Any
  ) -> NonAmplifiedBandInvDPMuonState:
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
    return NonAmplifiedBandInvDPMuonState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=new_optimizer_state,
        noise_state=new_noise_state,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    )

  return train_step, optimizer


__all__ = [
    "NonAmplifiedBandInvDPMuonState",
    "init_nonamplified_bandinv_dpmuon_state",
    "make_nonamplified_bandinv_dpmuon_train_step",
]
