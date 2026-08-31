"""Core single-query training and shadow math for Experiment 8.

The real update in this module is the existing naive, non-amplified
BandInvMF DP-AdamW update.  The diagnostic branch is deliberately kept out of
the parameter and optimizer updates: it receives the same clipped gradient,
but uses an independent RNG stream and maintains only shadow moment states.
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
    filter_latent_noise,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.privacy import (
    ParticipationSpec,
    PrivacyCalibration,
    make_clipped_gradient_query,
)
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    make_nonamplified_dpadamw_optimizer,
)
from dp_muon.training.nonamplified_linear import (
    validate_nonamplified_bandinv_privacy_setup,
)


PyTree = Any
BRANCHES = ("corr", "iid")
PATHS = ("P0", "P1", "P2", "P3")


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add(left: PyTree, right: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_scale(scale: Any, tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda value: scale * value, tree)


def _tree_square(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda value: value * value, tree)


def _tree_norm_sq(tree: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(leaf, jnp.float32) ** 2)
      for leaf in jax.tree_util.tree_leaves(tree)
  )


def _tree_dot(left: PyTree, right: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(a, jnp.float32) * jnp.asarray(b, jnp.float32))
      for a, b in zip(
          jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
      )
  )


def _normalized_tree_error(left: PyTree, right: PyTree) -> jax.Array:
  """Return a finite normalized error for two same-shaped pytrees."""
  difference = jax.tree_util.tree_map(lambda a, b: a - b, left, right)
  numerator = jnp.sqrt(_tree_norm_sq(difference))
  denominator = 1.0 + jnp.sqrt(_tree_norm_sq(right))
  return numerator / denominator


def _tree_ema(old: PyTree, value: PyTree, beta: float) -> PyTree:
  return jax.tree_util.tree_map(
      lambda previous, current: beta * previous + (1.0 - beta) * current,
      old,
      value,
  )


def _safe_bias_correction(beta: float, step: jax.Array) -> jax.Array:
  # step is always positive when called by advance_diagnostic_shadow.
  return 1.0 - beta ** step


def bandinv_marginal_variances(
    strategy: BandInvMFStrategy, iid_noise_std: float | jax.Array
) -> jax.Array:
  """Return ``phi_t = Var(xi_corr[t])`` for the streamed BandInvMF filter."""
  coef = jnp.asarray(strategy.noising_coef)
  if coef.ndim != 1 or coef.shape != (strategy.bandwidth,):
    raise ValueError("strategy.noising_coef must match strategy.bandwidth")
  sigma = jnp.asarray(iid_noise_std, dtype=coef.dtype)
  if sigma.ndim != 0:
    raise ValueError("iid_noise_std must be scalar")
  cumulative = jnp.cumsum(jnp.square(coef))
  row_index = jnp.minimum(jnp.arange(strategy.horizon), strategy.bandwidth - 1)
  return jnp.square(sigma) * cumulative[row_index]


def init_diagnostic_noise_state(
    params: PyTree, strategy: BandInvMFStrategy
) -> BandInvMFNoiseState:
  """Create state for the correlated diagnostic filter only.

  The IID control intentionally has no temporal/filter state.  Both branches
  still consume the same newly sampled standard-normal innovation each step.
  """
  return init_bandinv_noise_state(params, strategy.bandwidth)


def _standard_normal_tree(key: jax.Array, template: PyTree) -> PyTree:
  leaves, treedef = jax.tree_util.tree_flatten(template)
  keys = jax.random.split(key, len(leaves))
  values = [
      jax.random.normal(subkey, jnp.asarray(leaf).shape, dtype=jnp.asarray(leaf).dtype)
      for subkey, leaf in zip(keys, leaves, strict=True)
  ]
  return jax.tree_util.tree_unflatten(treedef, values)


def sample_paired_diagnostic_noise(
    key: jax.Array,
    corr_state: BandInvMFNoiseState,
    strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    phi_t: float | jax.Array,
) -> tuple[PyTree, PyTree, PyTree, BandInvMFNoiseState, jax.Array]:
  """Sample paired ``(xi_corr, xi_iid)`` from one standard-normal ``z_t``.

  Returns ``(xi_corr, xi_iid, z_t, new_corr_state, new_key)``.  Only the
  correlated branch advances a temporal filter state; the IID branch is
  constructed directly as ``sqrt(phi_t) * z_t``.
  """
  leaves = jax.tree_util.tree_leaves(corr_state.buffer)
  if not leaves:
    raise ValueError("diagnostic noise state must contain at least one leaf")
  z_key, new_key = jax.random.split(key)
  template = jax.tree_util.tree_map(
      lambda buffer: jnp.zeros(buffer.shape[1:], dtype=buffer.dtype), corr_state.buffer
  )
  z = _standard_normal_tree(z_key, template)
  # Make the coefficient dynamic inside the jitted train step.  This lets the
  # formal helper retain its eager validation while following its traced JAX
  # path here.
  coef = jnp.asarray(strategy.noising_coef)
  runtime_coef = coef + corr_state.step.astype(coef.dtype) * jnp.zeros_like(coef)
  xi_corr, xi_iid, new_corr_state = paired_diagnostic_noise_from_innovation(
      corr_state, z, strategy, iid_noise_std, phi_t, noising_coef=runtime_coef
  )
  return xi_corr, xi_iid, z, new_corr_state, new_key


def paired_diagnostic_noise_from_innovation(
    corr_state: BandInvMFNoiseState,
    z: PyTree,
    strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    phi_t: float | jax.Array,
    *,
    noising_coef: jax.Array | None = None,
) -> tuple[PyTree, PyTree, BandInvMFNoiseState]:
  """Construct both branches from supplied standard-normal innovations.

  The correlated branch deliberately delegates to the package's formal
  ``filter_latent_noise`` helper.  ``noising_coef`` is an internal escape hatch
  for the jitted caller to pass an equivalent traced coefficient; ordinary
  callers should leave it unset.
  """
  coefficient = (
      jnp.asarray(strategy.noising_coef)
      if noising_coef is None else jnp.asarray(noising_coef)
  )
  latent = jax.tree_util.tree_map(
      lambda value: jnp.asarray(iid_noise_std) * value, z
  )
  xi_corr, new_corr_state = filter_latent_noise(
      corr_state, latent, coefficient
  )
  iid_scale = jnp.sqrt(jnp.asarray(phi_t))
  xi_iid = jax.tree_util.tree_map(lambda value: iid_scale * value, z)
  return xi_corr, xi_iid, new_corr_state


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp8DiagnosticStep:
  """One-step shadow quantities retained for the online accumulator."""

  r: dict[str, PyTree]
  x: dict[str, dict[str, PyTree]]
  A: PyTree
  B: PyTree
  I: PyTree
  dq: PyTree
  xi: dict[str, PyTree]
  z: PyTree
  phi_t: jax.Array
  Phi_t: jax.Array
  reconstruction_error: jax.Array
  r_difference_error_corr: jax.Array
  r_difference_error_iid: jax.Array

  def tree_flatten(self):
    return (
        self.r, self.x, self.A, self.B, self.I, self.dq, self.xi, self.z,
        self.phi_t, self.Phi_t, self.reconstruction_error,
        self.r_difference_error_corr, self.r_difference_error_iid,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp8ShadowState:
  """Shared clean moments and branch-specific diagnostic moments."""

  clean_m: PyTree
  clean_v: PyTree
  corr_m: PyTree
  iid_m: PyTree
  corr_noise_m: PyTree
  iid_noise_m: PyTree
  corr_v: PyTree
  iid_v: PyTree
  bias_v: jax.Array
  step: jax.Array

  def tree_flatten(self):
    return (
        self.clean_m, self.clean_v, self.corr_m, self.iid_m,
        self.corr_noise_m, self.iid_noise_m, self.corr_v, self.iid_v,
        self.bias_v, self.step,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_diagnostic_shadow_state(params: PyTree) -> Exp8ShadowState:
  zeros = _zeros_like(params)
  return Exp8ShadowState(
      clean_m=zeros, clean_v=zeros, corr_m=zeros, iid_m=zeros,
      corr_noise_m=zeros, iid_noise_m=zeros, corr_v=zeros, iid_v=zeros,
      bias_v=jnp.asarray(0.0, jnp.float32), step=jnp.asarray(0, jnp.int32),
  )


def _zero_step(params: PyTree) -> Exp8DiagnosticStep:
  zero = _zeros_like(params)
  return Exp8DiagnosticStep(
      r={branch: zero for branch in BRANCHES},
      x={branch: {path: zero for path in PATHS} for branch in BRANCHES},
      A=zero, B=zero, I=zero, dq=zero,
      xi={branch: zero for branch in BRANCHES}, z=zero,
      phi_t=jnp.asarray(0.0, jnp.float32), Phi_t=jnp.asarray(0.0, jnp.float32),
      reconstruction_error=jnp.asarray(0.0, jnp.float32),
      r_difference_error_corr=jnp.asarray(0.0, jnp.float32),
      r_difference_error_iid=jnp.asarray(0.0, jnp.float32),
  )


def advance_diagnostic_shadow(
    state: Exp8ShadowState,
    g: PyTree,
    xi_corr: PyTree,
    xi_iid: PyTree,
    phi_t: float | jax.Array,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    eps: float,
) -> tuple[Exp8ShadowState, Exp8DiagnosticStep]:
  """Advance all four P0/P1/P2/P3 paths for both paired branches."""
  t = state.step + jnp.asarray(1, state.step.dtype)
  clean_m = _tree_ema(state.clean_m, g, beta1)
  clean_v = _tree_ema(state.clean_v, _tree_square(g), beta2)
  corr_m = _tree_ema(state.corr_m, _tree_add(g, xi_corr), beta1)
  iid_m = _tree_ema(state.iid_m, _tree_add(g, xi_iid), beta1)
  corr_noise_m = _tree_ema(state.corr_noise_m, xi_corr, beta1)
  iid_noise_m = _tree_ema(state.iid_noise_m, xi_iid, beta1)
  corr_v = _tree_ema(state.corr_v, _tree_square(_tree_add(g, xi_corr)), beta2)
  iid_v = _tree_ema(state.iid_v, _tree_square(_tree_add(g, xi_iid)), beta2)

  m_c = _tree_scale(1.0 / _safe_bias_correction(beta1, t), clean_m)
  m_corr = _tree_scale(1.0 / _safe_bias_correction(beta1, t), corr_m)
  m_iid = _tree_scale(1.0 / _safe_bias_correction(beta1, t), iid_m)
  r_from_difference = {
      "corr": jax.tree_util.tree_map(lambda a, b: a - b, m_corr, m_c),
      "iid": jax.tree_util.tree_map(lambda a, b: a - b, m_iid, m_c),
  }
  r = {
      "corr": _tree_scale(1.0 / _safe_bias_correction(beta1, t), corr_noise_m),
      "iid": _tree_scale(1.0 / _safe_bias_correction(beta1, t), iid_noise_m),
  }
  r_difference_error_corr = _normalized_tree_error(
      r_from_difference["corr"], r["corr"]
  )
  r_difference_error_iid = _normalized_tree_error(
      r_from_difference["iid"], r["iid"]
  )

  v_c = _tree_scale(1.0 / _safe_bias_correction(beta2, t), clean_v)
  bias_v = beta2 * state.bias_v + (1.0 - beta2) * jnp.asarray(phi_t)
  Phi_t = bias_v / _safe_bias_correction(beta2, t)
  p_c = jax.tree_util.tree_map(lambda value: 1.0 / (jnp.sqrt(value) + eps), v_c)
  p_phi = jax.tree_util.tree_map(
      lambda value: 1.0 / (jnp.sqrt(value + Phi_t) + eps), v_c
  )
  v_private = {
      "corr": _tree_scale(1.0 / _safe_bias_correction(beta2, t), corr_v),
      "iid": _tree_scale(1.0 / _safe_bias_correction(beta2, t), iid_v),
  }
  p_private = {
      branch: jax.tree_util.tree_map(
          lambda value: 1.0 / (jnp.sqrt(value) + eps), v_private[branch]
      ) for branch in BRANCHES
  }
  N = {
      branch: {
          "P0": r[branch],
          "P1": jax.tree_util.tree_map(lambda p, value: p * value, p_c, r[branch]),
          "P2": jax.tree_util.tree_map(lambda p, value: p * value, p_phi, r[branch]),
          "P3": jax.tree_util.tree_map(
              lambda p, value: p * value, p_private[branch], r[branch]
          ),
      } for branch in BRANCHES
  }
  x = {
      branch: {path: _tree_scale(-learning_rate, N[branch][path]) for path in PATHS}
      for branch in BRANCHES
  }

  mhat_c = m_c
  p_private_corr = p_private["corr"]
  q_p = jax.tree_util.tree_map(
      lambda p, m, noise: p * (m + noise), p_private_corr, mhat_c, r["corr"]
  )
  q_c = jax.tree_util.tree_map(lambda p, m: p * m, p_c, mhat_c)
  delta_p = jax.tree_util.tree_map(lambda a, b: a - b, p_private_corr, p_c)
  A = jax.tree_util.tree_map(lambda p, noise: p * noise, p_c, r["corr"])
  B = jax.tree_util.tree_map(lambda dp, m: dp * m, delta_p, mhat_c)
  I = jax.tree_util.tree_map(lambda dp, noise: dp * noise, delta_p, r["corr"])
  dq = jax.tree_util.tree_map(lambda a, b: a - b, q_p, q_c)
  reconstruction_residual = jax.tree_util.tree_map(
      lambda value, a, b, i: value - a - b - i, dq, A, B, I
  )
  residual_norm = jnp.sqrt(_tree_norm_sq(reconstruction_residual))
  dq_norm = jnp.sqrt(_tree_norm_sq(dq))
  component_scale = (
      1.0 + jnp.sqrt(_tree_norm_sq(A)) + jnp.sqrt(_tree_norm_sq(B))
      + jnp.sqrt(_tree_norm_sq(I))
  )
  # When dq is numerically zero, a relative error has no meaningful scale;
  # treat an absolute roundoff-sized residual as an exact reconstruction.
  reconstruction_error = jnp.where(
      dq_norm <= 1e-6 * component_scale,
      0.0,
      residual_norm / (dq_norm + 1e-30),
  )

  new_state = Exp8ShadowState(
      clean_m=clean_m, clean_v=clean_v, corr_m=corr_m, iid_m=iid_m,
      corr_noise_m=corr_noise_m, iid_noise_m=iid_noise_m,
      corr_v=corr_v, iid_v=iid_v, bias_v=bias_v, step=t,
  )
  return new_state, Exp8DiagnosticStep(
      r=r, x=x, A=A, B=B, I=I, dq=dq,
      xi={"corr": xi_corr, "iid": xi_iid}, z=_zeros_like(g),
      phi_t=jnp.asarray(phi_t), Phi_t=Phi_t,
      reconstruction_error=reconstruction_error,
      r_difference_error_corr=r_difference_error_corr,
      r_difference_error_iid=r_difference_error_iid,
  )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp8TrainState:
  """Real baseline state plus shadow-only diagnostic state."""

  params: PyTree
  optimizer_state: Any
  training_noise_state: BandInvMFNoiseState
  training_rng_key: jax.Array
  diagnostic_noise_state: BandInvMFNoiseState
  diagnostic_rng_key: jax.Array
  step: jax.Array
  shadow: Exp8ShadowState
  last_step: Exp8DiagnosticStep

  def tree_flatten(self):
    return (
        self.params, self.optimizer_state, self.training_noise_state,
        self.training_rng_key, self.diagnostic_noise_state,
        self.diagnostic_rng_key, self.step, self.shadow, self.last_step,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)

  @property
  def noise_state(self) -> BandInvMFNoiseState:
    """Compatibility name for the real training BandInvMF state."""
    return self.training_noise_state

  @property
  def rng_key(self) -> jax.Array:
    """Compatibility name for the real training RNG stream."""
    return self.training_rng_key


def _keys_equal(left: jax.Array, right: jax.Array) -> bool:
  left, right = jnp.asarray(left), jnp.asarray(right)
  if left.shape != right.shape:
    return False
  return bool(jnp.array_equal(left, right))


def init_exp8_train_state(
    params: PyTree,
    strategy: BandInvMFStrategy,
    training_rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
    diagnostic_rng_key: jax.Array | None = None,
) -> Exp8TrainState:
  """Initialise the real trajectory and independent diagnostic stream."""
  if diagnostic_rng_key is None:
    training_rng_key, diagnostic_rng_key = jax.random.split(training_rng_key)
  if _keys_equal(training_rng_key, diagnostic_rng_key):
    raise ValueError("training and diagnostic RNG keys must be independent")
  return Exp8TrainState(
      params=params,
      optimizer_state=optimizer.init(params),
      training_noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      training_rng_key=training_rng_key,
      diagnostic_noise_state=init_diagnostic_noise_state(params, strategy),
      diagnostic_rng_key=diagnostic_rng_key,
      step=jnp.asarray(0, jnp.int32),
      shadow=init_diagnostic_shadow_state(params),
      last_step=_zero_step(params),
  )


def make_exp8_train_step(
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
    Callable[[Exp8TrainState, Any], Exp8TrainState], optax.GradientTransformation
]:
  """Build the one-query real update with paired diagnostic shadows."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  validate_nonamplified_bandinv_privacy_setup(
      strategy, calibration, participation_spec
  )
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate, beta1=beta1, beta2=beta2,
      eps=eps, weight_decay=weight_decay,
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=calibration.clip_norm,
      normalize_by=calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )
  phi = bandinv_marginal_variances(strategy, calibration.iid_noise_std)

  def train_step(state: Exp8TrainState, batch: Any) -> Exp8TrainState:
    # The only model/loss gradient call in the complete Exp8 step.
    g = clipped_query(state.params, batch)

    train_step0 = state.training_noise_state.step
    coef = jnp.asarray(strategy.noising_coef)
    runtime_coef = coef + jnp.asarray(train_step0, coef.dtype) * jnp.zeros_like(coef)
    sigma = jnp.asarray(calibration.iid_noise_std)
    runtime_sigma = sigma + jnp.asarray(train_step0, sigma.dtype) * jnp.zeros_like(sigma)
    training_noise, new_training_noise_state, new_training_key = sample_bandinv_noise(
        state.training_rng_key,
        state.training_noise_state,
        runtime_coef,
        runtime_sigma,
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation, g, training_noise
    )
    updates, new_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    new_params = optax.apply_updates(state.params, updates)

    diagnostic_step0 = state.diagnostic_noise_state.step
    phi_t = phi[diagnostic_step0]
    xi_corr, xi_iid, z, new_diag_noise_state, new_diag_key = sample_paired_diagnostic_noise(
        state.diagnostic_rng_key,
        state.diagnostic_noise_state,
        strategy,
        calibration.iid_noise_std,
        phi_t,
    )
    new_shadow, last = advance_diagnostic_shadow(
        state.shadow, g, xi_corr, xi_iid, phi_t,
        beta1=beta1, beta2=beta2, learning_rate=learning_rate, eps=eps,
    )
    # Keep z in the step payload for paired-construction tests and debugging.
    last = Exp8DiagnosticStep(
        r=last.r, x=last.x, A=last.A, B=last.B, I=last.I, dq=last.dq,
        xi=last.xi, z=z, phi_t=last.phi_t, Phi_t=last.Phi_t,
        reconstruction_error=last.reconstruction_error,
        r_difference_error_corr=last.r_difference_error_corr,
        r_difference_error_iid=last.r_difference_error_iid,
    )
    return Exp8TrainState(
        params=new_params,
        optimizer_state=new_optimizer_state,
        training_noise_state=new_training_noise_state,
        training_rng_key=new_training_key,
        diagnostic_noise_state=new_diag_noise_state,
        diagnostic_rng_key=new_diag_key,
        step=state.step + jnp.asarray(1, state.step.dtype),
        shadow=new_shadow,
        last_step=last,
    )

  return train_step, optimizer


__all__ = [
    "BRANCHES", "PATHS", "Exp8DiagnosticStep", "Exp8ShadowState",
    "Exp8TrainState", "advance_diagnostic_shadow", "bandinv_marginal_variances",
    "init_diagnostic_noise_state", "init_diagnostic_shadow_state",
    "init_exp8_train_state", "make_exp8_train_step",
    "paired_diagnostic_noise_from_innovation", "sample_paired_diagnostic_noise",
]
