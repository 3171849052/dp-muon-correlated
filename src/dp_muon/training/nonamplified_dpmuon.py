"""Non-amplified IID DP-Muon: one private gradient, two optimizer groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.optim import ADAMW, MUON, muon_transform, vit_muon_parameter_labels
from dp_muon.privacy import (
    PrivacyCalibration,
    make_clipped_gradient_query,
    sample_iid_gaussian_noise,
)


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NonAmplifiedDPMuonState:
  """Checkpointable parameters, one partitioned Optax state, and DP RNG."""

  params: PyTree
  optimizer_state: Any
  rng_key: jax.Array
  step: jax.Array

  def tree_flatten(self):
    return (self.params, self.optimizer_state, self.rng_key, self.step), None

  @classmethod
  def tree_unflatten(cls, aux_data: None, children: tuple[PyTree, Any, jax.Array, jax.Array]):
    del aux_data
    params, optimizer_state, rng_key, step = children
    return cls(params=params, optimizer_state=optimizer_state, rng_key=rng_key, step=step)


def _finite_scalar(value: object, name: str, *, positive: bool = False,
                   nonnegative: bool = False) -> float:
  array = np.asarray(value)
  if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
    raise ValueError(f"{name} must be a finite scalar")
  result = float(array)
  if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
    raise ValueError(f"{name} must be a valid finite scalar")
  return result


def make_nonamplified_dpmuon_optimizer(
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
    use_bf16_ns: bool = True,
) -> optax.GradientTransformation:
  """Builds a single Optax partition with distinct Muon and AdamW settings."""
  _finite_scalar(muon_learning_rate, "muon_learning_rate", positive=True)
  _finite_scalar(muon_weight_decay, "muon_weight_decay", nonnegative=True)
  _finite_scalar(momentum, "momentum", nonnegative=True)
  if not 0 <= momentum < 1:
    raise ValueError("momentum must be in [0, 1)")
  if isinstance(ns_steps, bool) or not isinstance(ns_steps, int) or ns_steps < 1:
    raise ValueError("ns_steps must be a positive integer")
  _finite_scalar(consistent_rms, "consistent_rms", positive=True)
  _finite_scalar(adamw_learning_rate, "adamw_learning_rate", positive=True)
  for value, name in ((adamw_beta1, "adamw_beta1"), (adamw_beta2, "adamw_beta2")):
    _finite_scalar(value, name, nonnegative=True)
    if not 0 <= value < 1:
      raise ValueError(f"{name} must be in [0, 1)")
  _finite_scalar(adamw_eps, "adamw_eps", positive=True)
  _finite_scalar(adamw_weight_decay, "adamw_weight_decay", nonnegative=True)
  return optax.partition(
      {
          MUON: muon_transform(
              learning_rate=muon_learning_rate,
              weight_decay=muon_weight_decay,
              momentum=momentum,
              ns_steps=ns_steps,
              consistent_rms=consistent_rms,
              use_bf16_ns=use_bf16_ns,
          ),
          ADAMW: optax.adamw(
              learning_rate=adamw_learning_rate,
              b1=adamw_beta1,
              b2=adamw_beta2,
              eps=adamw_eps,
              weight_decay=adamw_weight_decay,
          ),
      },
      vit_muon_parameter_labels,
  )


def init_nonamplified_dpmuon_state(
    params: PyTree,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> NonAmplifiedDPMuonState:
  """Initializes the one top-level partitioned optimizer state."""
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax GradientTransformation")
  return NonAmplifiedDPMuonState(
      params=params, optimizer_state=optimizer.init(params), rng_key=rng_key,
      step=jnp.array(0, dtype=jnp.int32),
  )


# Kept as a local seam for tests while implemented centrally in privacy.
_sample_iid_gaussian_noise = sample_iid_gaussian_noise


def make_nonamplified_dpmuon_train_step(
    loss_fn: Callable[..., Any],
    calibration: PrivacyCalibration,
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
    Callable[[NonAmplifiedDPMuonState, Any], NonAmplifiedDPMuonState],
    optax.GradientTransformation,
]:
  """Builds ``clip -> one full-tree IID noise -> Muon/AdamW partition``.

  The returned optimizer is required only to initialize a state.  No group is
  clipped, calibrated, or noised separately: the partition sees projections
  of exactly the same already-private PyTree.
  """
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  if not isinstance(calibration, PrivacyCalibration):
    raise TypeError("calibration must be a PrivacyCalibration")
  _finite_scalar(calibration.iid_noise_std, "calibration.iid_noise_std", nonnegative=True)
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
      state: NonAmplifiedDPMuonState, batch: Any
  ) -> NonAmplifiedDPMuonState:
    if not isinstance(state, NonAmplifiedDPMuonState):
      raise TypeError("state must be a NonAmplifiedDPMuonState")
    clipped_grad = clipped_query(state.params, batch)
    # This is the unique privacy mechanism invocation for this logical batch.
    noise, new_key = _sample_iid_gaussian_noise(
        state.rng_key, clipped_grad, jnp.asarray(calibration.iid_noise_std)
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation, clipped_grad, noise
    )
    updates, optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    return NonAmplifiedDPMuonState(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=optimizer_state,
        rng_key=new_key,
        step=state.step + jnp.array(1, dtype=state.step.dtype),
    )

  return train_step, optimizer


__all__ = [
    "NonAmplifiedDPMuonState",
    "init_nonamplified_dpmuon_state",
    "make_nonamplified_dpmuon_optimizer",
    "make_nonamplified_dpmuon_train_step",
]
