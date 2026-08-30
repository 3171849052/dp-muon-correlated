"""Single-query training state for Experiment 7.

Every step evaluates the clipped query once and samples BandInvMF once.  The
four second-moment shadows and the AdamBC shadow consume that same gradient
and that same correlated-noise realization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.privacy import ParticipationSpec, PrivacyCalibration, make_clipped_gradient_query
from dp_muon.training.nonamplified_linear import validate_nonamplified_bandinv_privacy_setup


PyTree = Any
Algorithm = Literal["baseline", "bc"]
DEFAULT_V_FLOOR = 1e-30


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def bandinv_marginal_variances(
    strategy: BandInvMFStrategy, iid_noise_std: float | jax.Array
) -> jax.Array:
  """Return exact per-coordinate marginal variances of streamed noise.

  The sampler applies the FIR coefficients ``strategy.noising_coef`` (the
  nonzero bands of ``C^{-1}``) to iid Gaussian latents.  Row ``t`` therefore
  contains only the first ``min(t + 1, bandwidth)`` coefficients.
  """
  coef = jnp.asarray(strategy.noising_coef)
  if coef.ndim != 1 or coef.shape != (strategy.bandwidth,):
    raise ValueError("strategy.noising_coef must match strategy.bandwidth")
  sigma = jnp.asarray(iid_noise_std, dtype=coef.dtype)
  if sigma.ndim != 0:
    raise ValueError("iid_noise_std must be scalar")
  cumulative = jnp.cumsum(jnp.square(coef))
  row_index = jnp.minimum(jnp.arange(strategy.horizon), strategy.bandwidth - 1)
  return jnp.square(sigma) * cumulative[row_index]


def update_bias_ema(
    previous: float | jax.Array, phi_t: float | jax.Array, beta2: float,
    step: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Update and debias the EMA of the time-varying marginal variance."""
  bias_v = beta2 * jnp.asarray(previous) + (1.0 - beta2) * jnp.asarray(phi_t)
  correction = 1.0 - beta2 ** jnp.asarray(step)
  return bias_v, bias_v / correction


def shadow_second_moment_inputs(g: PyTree, noise: PyTree) -> dict[str, PyTree]:
  """Build the four factorial inputs from one gradient/noise realization."""
  return {
      "00": jax.tree_util.tree_map(lambda x: x * x, g),
      "10": jax.tree_util.tree_map(lambda x, z: x * x + 2.0 * x * z, g, noise),
      "01": jax.tree_util.tree_map(lambda x, z: x * x + z * z, g, noise),
      "11": jax.tree_util.tree_map(lambda x, z: (x + z) * (x + z), g, noise),
  }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp7TrainState:
  params: PyTree
  noise_state: BandInvMFNoiseState
  rng_key: jax.Array
  step: jax.Array
  clean_m: PyTree
  dp_m: PyTree
  noise_m: PyTree
  v00: PyTree
  v10: PyTree
  v01: PyTree
  v11: PyTree
  bias_v: jax.Array
  phi_t: jax.Array
  last_noise: PyTree
  last_latent_noise: PyTree

  def tree_flatten(self):
    return (
        self.params, self.noise_state, self.rng_key, self.step,
        self.clean_m, self.dp_m, self.noise_m,
        self.v00, self.v10, self.v01, self.v11,
        self.bias_v, self.phi_t, self.last_noise, self.last_latent_noise,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)

  @property
  def clean_v(self) -> PyTree:
    """Compatibility alias used by Experiment 6 terminology."""
    return self.v00

  @property
  def dp_v(self) -> PyTree:
    """Compatibility alias used by Experiment 6 terminology."""
    return self.v11


def init_exp7_train_state(
    params: PyTree, strategy: BandInvMFStrategy, rng_key: jax.Array
) -> Exp7TrainState:
  zeros = _zeros_like(params)
  return Exp7TrainState(
      params=params,
      noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key=rng_key,
      step=jnp.asarray(0, jnp.int32),
      clean_m=zeros,
      dp_m=zeros,
      noise_m=zeros,
      v00=zeros,
      v10=zeros,
      v01=zeros,
      v11=zeros,
      bias_v=jnp.asarray(0.0, jnp.float32),
      phi_t=jnp.asarray(0.0, jnp.float32),
      last_noise=zeros,
      last_latent_noise=zeros,
  )


def make_exp7_train_step(
    loss_fn: Callable[..., Any],
    strategy: BandInvMFStrategy,
    calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    algorithm: Algorithm,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    v_floor: float = DEFAULT_V_FLOOR,
    microbatch_size: int | None = None,
) -> Callable[[Exp7TrainState, Any], Exp7TrainState]:
  """Create the baseline or BC real update with all shadows maintained online."""
  if algorithm not in ("baseline", "bc"):
    raise ValueError("algorithm must be 'baseline' or 'bc'")
  if not math.isfinite(v_floor) or v_floor < 0:
    raise ValueError("v_floor must be finite and non-negative")
  validate_nonamplified_bandinv_privacy_setup(strategy, calibration, participation_spec)
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )
  phi = bandinv_marginal_variances(strategy, calibration.iid_noise_std)

  def step_fn(state: Exp7TrainState, batch: Any) -> Exp7TrainState:
    g = clipped_query(state.params, batch)
    step0 = state.noise_state.step
    coef = jnp.asarray(strategy.noising_coef)
    runtime_coef = coef + jnp.asarray(step0, coef.dtype) * jnp.zeros_like(coef)
    sigma = jnp.asarray(calibration.iid_noise_std)
    runtime_sigma = sigma + jnp.asarray(step0, sigma.dtype) * jnp.zeros_like(sigma)
    noise, noise_state, rng_key = sample_bandinv_noise(
        state.rng_key, state.noise_state, runtime_coef, runtime_sigma
    )
    # sample_bandinv_noise writes exactly the newly drawn iid latent into the
    # old cursor slot.  Reading it here does not sample or filter again.
    latent = jax.tree_util.tree_map(
        lambda buffer: buffer[state.noise_state.cursor], noise_state.buffer
    )
    private = jax.tree_util.tree_map(lambda x, z: x + z, g, noise)
    t = state.step + jnp.asarray(1, state.step.dtype)
    clean_m = jax.tree_util.tree_map(
        lambda old, x: beta1 * old + (1.0 - beta1) * x, state.clean_m, g
    )
    dp_m = jax.tree_util.tree_map(
        lambda old, x: beta1 * old + (1.0 - beta1) * x, state.dp_m, private
    )
    noise_m = jax.tree_util.tree_map(
        lambda old, x: beta1 * old + (1.0 - beta1) * x, state.noise_m, noise
    )
    inputs = shadow_second_moment_inputs(g, noise)

    def ema(old, value):
      return jax.tree_util.tree_map(
          lambda a, b: beta2 * a + (1.0 - beta2) * b, old, value
      )

    v00 = ema(state.v00, inputs["00"])
    v10 = ema(state.v10, inputs["10"])
    v01 = ema(state.v01, inputs["01"])
    v11 = ema(state.v11, inputs["11"])
    phi_t = phi[state.step]
    bias_v, phi_hat = update_bias_ema(state.bias_v, phi_t, beta2, t)
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1 ** t), dp_m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2 ** t), v11)
    if algorithm == "bc":
      preconditioner = jax.tree_util.tree_map(
          lambda value: 1.0 / (jnp.sqrt(jnp.maximum(value - phi_hat, v_floor)) + eps),
          v_hat,
      )
    else:
      preconditioner = jax.tree_util.tree_map(
          lambda value: 1.0 / (jnp.sqrt(value) + eps), v_hat
      )
    params = jax.tree_util.tree_map(
        lambda parameter, moment, p: (
            (1.0 - learning_rate * weight_decay) * parameter
            - learning_rate * moment * p
        ),
        state.params, m_hat, preconditioner,
    )
    return Exp7TrainState(
        params=params,
        noise_state=noise_state,
        rng_key=rng_key,
        step=t,
        clean_m=clean_m,
        dp_m=dp_m,
        noise_m=noise_m,
        v00=v00,
        v10=v10,
        v01=v01,
        v11=v11,
        bias_v=bias_v,
        phi_t=phi_t,
        last_noise=noise,
        last_latent_noise=latent,
    )

  return step_fn


__all__ = [
    "Algorithm",
    "DEFAULT_V_FLOOR",
    "Exp7TrainState",
    "bandinv_marginal_variances",
    "init_exp7_train_state",
    "make_exp7_train_step",
    "shadow_second_moment_inputs",
    "update_bias_ema",
]
