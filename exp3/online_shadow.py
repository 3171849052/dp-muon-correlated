"""Online shadow diagnostics embedded in a real BandInvMF DP-AdamW step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from dp_muon.bandinvmf import (
    BandInvMFNoiseState, BandInvMFStrategy, init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.privacy import ParticipationSpec, PrivacyCalibration, make_clipped_gradient_query
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer
from dp_muon.training.nonamplified_bandinv_dpadamw import NonAmplifiedBandInvDPAdamWState
from dp_muon.training.nonamplified_linear import validate_nonamplified_bandinv_privacy_setup

PyTree = Any


def aggregate_ratio(sum_j: float | jax.Array, sum_d: float | jax.Array) -> jax.Array:
  """Compute the required aggregate ``sum(J_t) / sum(D_t)`` ratio."""
  numerator, denominator = jnp.asarray(sum_j), jnp.asarray(sum_d)
  return jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(numerator))


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)


def _sqnorm(tree: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(x, dtype=jnp.float32) ** 2)
      for x in jax.tree_util.tree_leaves(tree)
  )


def _dot(left: PyTree, right: PyTree) -> jax.Array:
  return sum(jnp.sum(jnp.asarray(a, dtype=jnp.float32) * jnp.asarray(b, dtype=jnp.float32))
             for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OnlineShadowState(NonAmplifiedBandInvDPAdamWState):
  clean_m: PyTree
  clean_v: PyTree
  dp_m: PyTree
  dp_v: PyTree
  noise_m: PyTree
  d_linear: PyTree
  d_adamw: PyTree
  j_linear: jax.Array
  d_prefix_linear: jax.Array
  sum_j_linear: jax.Array
  sum_d_linear: jax.Array
  j_adamw: jax.Array
  d_prefix_adamw: jax.Array
  sum_j_adamw: jax.Array
  sum_d_adamw: jax.Array
  amplitude: jax.Array
  direction: jax.Array

  def tree_flatten(self):
    children = (self.params, self.optimizer_state, self.noise_state, self.rng_key, self.step,
                self.clean_m, self.clean_v, self.dp_m, self.dp_v, self.noise_m,
                self.d_linear, self.d_adamw, self.j_linear, self.d_prefix_linear,
                self.sum_j_linear, self.sum_d_linear, self.j_adamw, self.d_prefix_adamw,
                self.sum_j_adamw, self.sum_d_adamw, self.amplitude, self.direction)
    return children, None

  @classmethod
  def tree_unflatten(cls, aux, children):
    del aux
    return cls(*children)


def init_online_shadow_state(params: PyTree, strategy: BandInvMFStrategy,
                             rng_key: jax.Array, optimizer: optax.GradientTransformation) -> OnlineShadowState:
  zeros = _zeros_like(params)
  return OnlineShadowState(params, optimizer.init(params), init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key, jnp.array(0, jnp.int32), zeros, zeros, zeros, zeros, zeros, zeros, zeros,
      jnp.array(0., jnp.float32), jnp.array(0., jnp.float32), jnp.array(0., jnp.float32),
      jnp.array(0., jnp.float32), jnp.array(0., jnp.float32), jnp.array(0., jnp.float32),
      jnp.array(0., jnp.float32), jnp.array(0., jnp.float32), jnp.array(0., jnp.float32),
      jnp.array(0., jnp.float32))


def make_online_shadow_train_step(loss_fn: Callable[..., Any], strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration, participation_spec: ParticipationSpec, *, learning_rate: float,
    beta1: float = .9, beta2: float = .999, eps: float = 1e-8, weight_decay: float = .01,
    microbatch_size: int | None = None):
  """Builds a genuine DP-AdamW update plus scalar online diagnostics."""
  validate_nonamplified_bandinv_privacy_setup(strategy, calibration, participation_spec)
  optimizer = make_nonamplified_dpadamw_optimizer(learning_rate=learning_rate, beta1=beta1,
      beta2=beta2, eps=eps, weight_decay=weight_decay)
  clipped_query = make_clipped_gradient_query(loss_fn, clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by, batch_argnums=1, keep_batch_dim=True,
      microbatch_size=microbatch_size)

  def step_fn(state: OnlineShadowState, batch: Any) -> OnlineShadowState:
    g = clipped_query(state.params, batch)
    step = state.noise_state.step
    noising_coef = jnp.asarray(strategy.noising_coef)
    runtime_noising_coef = noising_coef + (
        jnp.asarray(step, dtype=noising_coef.dtype) * jnp.zeros_like(noising_coef)
    )
    iid_noise_std = jnp.asarray(calibration.iid_noise_std)
    runtime_iid_noise_std = iid_noise_std + (
        jnp.asarray(step, dtype=iid_noise_std.dtype) * jnp.zeros_like(iid_noise_std)
    )
    noise, new_noise, new_key = sample_bandinv_noise(state.rng_key, state.noise_state,
        runtime_noising_coef, runtime_iid_noise_std)
    private = jax.tree_util.tree_map(lambda a, b: a + b, g, noise)
    updates, new_opt = optimizer.update(private, state.optimizer_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    t = state.step + 1
    b1t, b2t = beta1 ** t, beta2 ** t
    cm = jax.tree_util.tree_map(lambda m, x: beta1*m + (1-beta1)*x, state.clean_m, g)
    cv = jax.tree_util.tree_map(lambda v, x: beta2*v + (1-beta2)*x*x, state.clean_v, g)
    dm = jax.tree_util.tree_map(lambda m, x: beta1*m + (1-beta1)*x, state.dp_m, private)
    dv = jax.tree_util.tree_map(lambda v, x: beta2*v + (1-beta2)*x*x, state.dp_v, private)
    nm = jax.tree_util.tree_map(lambda m, x: beta1*m + (1-beta1)*x, state.noise_m, noise)
    q0 = jax.tree_util.tree_map(lambda m, v: (m/(1-b1t))/(jnp.sqrt(v/(1-b2t))+eps), cm, cv)
    qdp = jax.tree_util.tree_map(lambda m, v: (m/(1-b1t))/(jnp.sqrt(v/(1-b2t))+eps), dm, dv)
    rlin = jax.tree_util.tree_map(lambda m: m/(1-b1t), nm)
    dq = jax.tree_util.tree_map(lambda a, b: a-b, qdp, q0)
    amp = jnp.sqrt(_sqnorm(dq)) / (jnp.sqrt(_sqnorm(rlin)) + 1e-12)
    direction = _dot(dq, rlin) / (jnp.sqrt(_sqnorm(dq))*jnp.sqrt(_sqnorm(rlin)) + 1e-12)
    xlin = jax.tree_util.tree_map(lambda x: -learning_rate*x, rlin)
    xdp = jax.tree_util.tree_map(lambda x: -learning_rate*x, dq)
    dlin = jax.tree_util.tree_map(lambda d, x: (1-learning_rate*weight_decay)*d+x, state.d_linear, xlin)
    dadam = jax.tree_util.tree_map(lambda d, x: (1-learning_rate*weight_decay)*d+x, state.d_adamw, xdp)
    jl, ja = _sqnorm(dlin), _sqnorm(dadam)
    dl_prefix = state.d_prefix_linear + _sqnorm(xlin)
    da_prefix = state.d_prefix_adamw + _sqnorm(xdp)
    return OnlineShadowState(new_params, new_opt, new_noise, new_key, t, cm, cv, dm, dv, nm,
        dlin, dadam, jl, dl_prefix, state.sum_j_linear+jl, state.sum_d_linear+dl_prefix,
        ja, da_prefix, state.sum_j_adamw+ja, state.sum_d_adamw+da_prefix, amp, direction)

  return step_fn, optimizer


__all__ = ["OnlineShadowState", "aggregate_ratio", "init_online_shadow_state", "make_online_shadow_train_step"]
