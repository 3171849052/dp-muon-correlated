"""Frozen-p DP-AdamW with one continuous Phase-II BandInvMF mechanism."""

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
from dp_muon.optim import (
    FrozenPAdamW,
    FrozenPAdamWState,
    freeze_optax_adamw,
)
from dp_muon.privacy import (
    ParticipationSpec,
    PrivacyCalibration,
    make_clipped_gradient_query,
    sample_iid_gaussian_noise,
)

from .nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedFrozenPBandInvDPAdamWState:
  """Checkpointable hybrid state with a single post-switch noise stream."""

  params: PyTree
  optimizer_state: Any
  frozen_state: FrozenPAdamWState
  noise_state: BandInvMFNoiseState
  rng_key: jax.Array
  step: jax.Array
  switch_step: int

  def tree_flatten(self):
    return (
        self.params,
        self.optimizer_state,
        self.frozen_state,
        self.noise_state,
        self.rng_key,
        self.step,
    ), self.switch_step

  @classmethod
  def tree_unflatten(cls, switch_step, children):
    params, optimizer_state, frozen_state, noise_state, rng_key, step = children
    return cls(
        params=params,
        optimizer_state=optimizer_state,
        frozen_state=frozen_state,
        noise_state=noise_state,
        rng_key=rng_key,
        step=step,
        switch_step=switch_step,
    )


def _empty_frozen_state(params: PyTree) -> FrozenPAdamWState:
  zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
  ones = jax.tree_util.tree_map(jnp.ones_like, params)
  return FrozenPAdamWState(
      count=jnp.array(0, dtype=jnp.int32),
      mu=zeros,
      frozen_nu=zeros,
      p_star=ones,
  )


def init_nonamplified_frozen_p_bandinv_dpadamw_state(
    params: PyTree,
    strategy: BandInvMFStrategy,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
    *,
    switch_step: int,
) -> NonAmplifiedFrozenPBandInvDPAdamWState:
  """Initializes AdamW and an untouched continuous Phase-II noise buffer."""
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax GradientTransformation")
  if not isinstance(switch_step, int) or isinstance(switch_step, bool) or switch_step < 1:
    raise ValueError("switch_step must be a positive integer")
  return NonAmplifiedFrozenPBandInvDPAdamWState(
      params=params,
      optimizer_state=optimizer.init(params),
      frozen_state=_empty_frozen_state(params),
      noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
      switch_step=switch_step,
  )


# Local seams make the two mechanism phases independently testable.
_sample_iid_gaussian_noise = sample_iid_gaussian_noise


def _validate_strategy(
    strategy: BandInvMFStrategy, participation_spec: ParticipationSpec, switch_step: int
) -> None:
  if strategy.horizon != participation_spec.horizon - switch_step:
    raise ValueError("strategy horizon must equal the Phase-II horizon")
  if strategy.max_participations != participation_spec.max_participations:
    raise ValueError("strategy max_participations must match the global contract")
  if strategy.bandwidth < 1 or strategy.bandwidth > strategy.horizon:
    raise ValueError("strategy bandwidth is invalid")
  if strategy.min_sep != min(participation_spec.min_sep, strategy.horizon):
    raise ValueError("strategy min_sep must match the global contract on Phase II")
  if not 1 <= switch_step < participation_spec.horizon:
    raise ValueError("switch_step must lie in [1, horizon)")


def make_nonamplified_frozen_p_bandinv_dpadamw_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    switch_step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    microbatch_size: int | None = None,
) -> tuple[
    Callable[
        [NonAmplifiedFrozenPBandInvDPAdamWState, Any],
        NonAmplifiedFrozenPBandInvDPAdamWState,
    ],
    optax.GradientTransformation,
]:
  """Builds IID warmup -> state-preserving frozen-p -> continuous BandInvMF."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  if not isinstance(calibration, PrivacyCalibration):
    raise TypeError("calibration must be a PrivacyCalibration")
  _validate_strategy(strategy, participation_spec, switch_step)
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay,
  )
  frozen_optimizer = FrozenPAdamW(
      learning_rate=learning_rate,
      beta1=beta1,
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

  def warmup_step(
      state: NonAmplifiedFrozenPBandInvDPAdamWState, clipped_grad: PyTree
  ) -> NonAmplifiedFrozenPBandInvDPAdamWState:
    noise, new_key = _sample_iid_gaussian_noise(
        state.rng_key, clipped_grad, jnp.asarray(calibration.iid_noise_std)
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation,
        clipped_grad,
        noise,
    )
    updates, new_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    new_step = state.step + jnp.array(1, dtype=state.step.dtype)
    new_frozen_state = jax.lax.cond(
        new_step == state.switch_step,
        lambda _: freeze_optax_adamw(
            new_optimizer_state, beta2=beta2, eps=eps
        ),
        lambda _: state.frozen_state,
        operand=None,
    )
    return NonAmplifiedFrozenPBandInvDPAdamWState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=new_optimizer_state,
        frozen_state=new_frozen_state,
        noise_state=state.noise_state,
        rng_key=new_key,
        step=new_step,
        switch_step=state.switch_step,
    )

  def phase_step(
      state: NonAmplifiedFrozenPBandInvDPAdamWState, clipped_grad: PyTree
  ) -> NonAmplifiedFrozenPBandInvDPAdamWState:
    noising_coef = jnp.asarray(strategy.noising_coef)
    runtime_noising_coef = noising_coef + (
        jnp.asarray(state.step, dtype=noising_coef.dtype)
        * jnp.zeros_like(noising_coef)
    )
    correlated_noise, new_noise_state, new_key = sample_bandinv_noise(
        state.rng_key,
        state.noise_state,
        runtime_noising_coef,
        jnp.asarray(calibration.iid_noise_std),
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation,
        clipped_grad,
        correlated_noise,
    )
    updates, new_frozen_state = frozen_optimizer.update(
        private_grad, state.frozen_state, state.params
    )
    return NonAmplifiedFrozenPBandInvDPAdamWState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=state.optimizer_state,
        frozen_state=new_frozen_state,
        noise_state=new_noise_state,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
        switch_step=state.switch_step,
    )

  def train_step(
      state: NonAmplifiedFrozenPBandInvDPAdamWState, batch: Any
  ) -> NonAmplifiedFrozenPBandInvDPAdamWState:
    if not isinstance(state, NonAmplifiedFrozenPBandInvDPAdamWState):
      raise TypeError("state must be a NonAmplifiedFrozenPBandInvDPAdamWState")
    clipped_grad = clipped_query(state.params, batch)
    return jax.lax.cond(
        state.step < state.switch_step,
        lambda value: warmup_step(value[0], value[1]),
        lambda value: phase_step(value[0], value[1]),
        (state, clipped_grad),
    )

  return train_step, optimizer


__all__ = [
    "NonAmplifiedFrozenPBandInvDPAdamWState",
    "init_nonamplified_frozen_p_bandinv_dpadamw_state",
    "make_nonamplified_frozen_p_bandinv_dpadamw_train_step",
]
