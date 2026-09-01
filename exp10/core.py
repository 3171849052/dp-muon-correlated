"""Real paired MF/IID trajectories and second-moment diagnostics for Exp10.

The two branches in this module are deliberately both part of the training
state.  They have separate parameters and Optax AdamW states, while one
standard-normal innovation is sampled per optimizer step and fed to both
noise constructions.  The MF branch uses the repository's causal
``filter_latent_noise`` implementation; the IID branch uses the exact
per-step marginal variance of that filter.
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
BRANCHES = ("mf", "iid")
INSTANTANEOUS_COMPONENTS = ("g2", "g2_cross", "xi2")
EMA_COMPONENTS = ("V_g", "V_g_cross", "V_xi")
COMPONENTS = INSTANTANEOUS_COMPONENTS + EMA_COMPONENTS


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add(left: PyTree, right: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_sub(left: PyTree, right: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda a, b: a - b, left, right)


def _tree_square(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda value: value * value, tree)


def _tree_sum(tree: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(leaf, jnp.float32))
      for leaf in jax.tree_util.tree_leaves(tree)
  )


def _tree_numel(tree: PyTree) -> int:
  return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))


def _tree_max_abs(tree: PyTree) -> jax.Array:
  values = [jnp.max(jnp.abs(jnp.asarray(leaf, jnp.float32)))
            for leaf in jax.tree_util.tree_leaves(tree)]
  if not values:
    return jnp.asarray(0.0, jnp.float32)
  result = values[0]
  for value in values[1:]:
    result = jnp.maximum(result, value)
  return result


def _tree_rms(tree: PyTree, denominator: int | None = None) -> jax.Array:
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    return jnp.asarray(0.0, jnp.float32)
  count = _tree_numel(tree) if denominator is None else int(denominator)
  return jnp.sqrt(_tree_sum(_tree_square(tree)) / max(1, count))


def _safe_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
  return jnp.where(
      jnp.isfinite(denominator) & (jnp.abs(denominator) > 1e-30),
      numerator / denominator,
      0.0,
  )


def _tree_ema(old: PyTree, value: PyTree, beta: float) -> PyTree:
  return jax.tree_util.tree_map(
      lambda previous, current: beta * previous + (1.0 - beta) * current,
      old,
      value,
  )


def _bias_correct(tree: PyTree, step: jax.Array, beta: float) -> PyTree:
  correction = 1.0 - jnp.asarray(beta) ** step
  correction = jnp.where(step == 0, 1.0, correction)
  return jax.tree_util.tree_map(lambda value: value / correction, tree)


def bandinv_marginal_variances(
    strategy: BandInvMFStrategy, iid_noise_std: float | jax.Array
) -> jax.Array:
  """Return ``phi_t = Var(xi_mf_t)`` for the streamed causal filter."""
  coef = jnp.asarray(strategy.noising_coef)
  if coef.ndim != 1 or coef.shape != (strategy.bandwidth,):
    raise ValueError("strategy.noising_coef must match strategy.bandwidth")
  sigma = jnp.asarray(iid_noise_std, dtype=coef.dtype)
  if sigma.ndim != 0:
    raise ValueError("iid_noise_std must be scalar")
  cumulative = jnp.cumsum(jnp.square(coef))
  row_index = jnp.minimum(
      jnp.arange(strategy.horizon), strategy.bandwidth - 1
  )
  return jnp.square(sigma) * cumulative[row_index]


def _standard_normal_tree(key: jax.Array, template: PyTree) -> PyTree:
  leaves, treedef = jax.tree_util.tree_flatten(template)
  if not leaves:
    raise ValueError("template must contain at least one array leaf")
  keys = jax.random.split(key, len(leaves))
  values = [
      jax.random.normal(
          subkey, jnp.asarray(leaf).shape, dtype=jnp.asarray(leaf).dtype
      )
      for subkey, leaf in zip(keys, leaves, strict=True)
  ]
  return jax.tree_util.tree_unflatten(treedef, values)


def sample_shared_latent_noise(
    key: jax.Array, params_template: PyTree
) -> tuple[PyTree, jax.Array]:
  """Sample one standard-normal ``z_t`` tree and return ``(z_t, next_key)``."""
  z_key, next_key = jax.random.split(key)
  return _standard_normal_tree(z_key, params_template), next_key


def paired_noise_from_innovation(
    mf_noise_state: BandInvMFNoiseState,
    z_t: PyTree,
    strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    phi_t: float | jax.Array,
) -> tuple[PyTree, PyTree, BandInvMFNoiseState]:
  """Construct ``(xi_mf_t, xi_iid_t)`` from the same current innovation.

  ``xi_mf_t`` is delegated to the package BandInvMF causal filter.  No random
  draw happens here, which makes the shared-innovation invariant explicit.
  """
  latent = jax.tree_util.tree_map(
      lambda value: jnp.asarray(iid_noise_std) * value, z_t
  )
  # The state-dependent zero keeps the coefficient on the traced execution
  # path when this helper is called through ``jax.jit``.  This is numerically
  # identical to the fitted coefficient and avoids eager-only validation
  # trying to convert a traced finiteness check to a Python bool.
  coefficient = jnp.asarray(strategy.noising_coef)
  coefficient = coefficient + mf_noise_state.step.astype(
      coefficient.dtype
  ) * jnp.zeros_like(coefficient)
  xi_mf, new_state = filter_latent_noise(
      mf_noise_state, latent, coefficient
  )
  xi_iid = jax.tree_util.tree_map(
      lambda value: jnp.sqrt(jnp.asarray(phi_t)) * value, z_t
  )
  return xi_mf, xi_iid, new_state


def second_moment_components(g: PyTree, xi: PyTree) -> dict[str, PyTree]:
  """Return the three coordinatewise terms requested by Exp10."""
  g2 = _tree_square(g)
  g2_cross = jax.tree_util.tree_map(
      lambda gradient, noise: gradient * gradient + 2.0 * gradient * noise,
      g,
      xi,
  )
  xi2 = _tree_square(xi)
  return {"g2": g2, "g2_cross": g2_cross, "xi2": xi2}


def component_identity_residual(g: PyTree, xi: PyTree) -> PyTree:
  """Return ``(g + xi)^2 - (g2_cross + xi2)`` coordinatewise."""
  components = second_moment_components(g, xi)
  private2 = _tree_square(_tree_add(g, xi))
  return _tree_sub(private2, _tree_add(components["g2_cross"], components["xi2"]))


def component_metrics(
    components: dict[str, PyTree], *,
    cross_term: PyTree | None = None,
    num_coordinates: int | None = None,
) -> dict[str, jax.Array]:
  """Compute all requested per-step scalar metrics for one branch."""
  g2 = components["g2"]
  g2_cross = components["g2_cross"]
  xi2 = components["xi2"]
  # The training path supplies the literal ``2 * g * xi`` tree.  The fallback
  # keeps this helper convenient for serialized/test component dictionaries.
  cross_term = _tree_sub(g2_cross, g2) if cross_term is None else cross_term
  count = _tree_numel(g2) if num_coordinates is None else int(num_coordinates)
  denominator = jnp.asarray(max(1, count), jnp.float32)
  sum_g2 = _tree_sum(g2)
  sum_cross = _tree_sum(g2_cross)
  sum_xi2 = _tree_sum(xi2)
  sum_term = _tree_sum(cross_term)
  sum_abs_term = _tree_sum(
      jax.tree_util.tree_map(lambda value: jnp.abs(value), cross_term)
  )
  sum_term_sq = _tree_sum(_tree_square(cross_term))
  negative = _tree_sum(
      jax.tree_util.tree_map(
          lambda value: (value < 0).astype(jnp.float32), g2_cross
      )
  )
  return {
      "mean_g2": sum_g2 / denominator,
      "mean_g2_cross": sum_cross / denominator,
      "mean_xi2": sum_xi2 / denominator,
      "mean_2gxi": sum_term / denominator,
      "mean_abs_2gxi": sum_abs_term / denominator,
      "rms_2gxi": jnp.sqrt(sum_term_sq / denominator),
      "negative_fraction_g2_cross": negative / denominator,
      "R_signed": _safe_ratio(sum_term, sum_g2),
      "R_abs": _safe_ratio(sum_abs_term, sum_g2),
      "R_noise": _safe_ratio(sum_xi2, sum_g2),
      "rho_fb": _safe_ratio(sum_term, sum_xi2),
      # This explicit alias makes the IID negative control easy to find in CSV.
      "mean_g2_cross_minus_g2": sum_term / denominator,
      "num_coordinates": denominator,
  }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdamSecondMomentDiagnostics:
  """Uncorrected EMA storage for the three requested second-moment terms."""

  v_g_ema: PyTree
  v_g_cross_ema: PyTree
  v_xi_ema: PyTree
  step: jax.Array

  def tree_flatten(self):
    return (
        self.v_g_ema, self.v_g_cross_ema, self.v_xi_ema, self.step
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_second_moment_diagnostics(params: PyTree) -> AdamSecondMomentDiagnostics:
  zeros = _zeros_like(params)
  return AdamSecondMomentDiagnostics(
      v_g_ema=zeros,
      v_g_cross_ema=zeros,
      v_xi_ema=zeros,
      step=jnp.asarray(0, jnp.int32),
  )


def advance_second_moment_diagnostics(
    state: AdamSecondMomentDiagnostics,
    components: dict[str, PyTree],
    *,
    beta2: float,
) -> tuple[AdamSecondMomentDiagnostics, dict[str, PyTree]]:
  """Advance beta2-consistent EMAs and return their bias-corrected values."""
  step = state.step + jnp.asarray(1, state.step.dtype)
  new_state = AdamSecondMomentDiagnostics(
      v_g_ema=_tree_ema(state.v_g_ema, components["g2"], beta2),
      v_g_cross_ema=_tree_ema(
          state.v_g_cross_ema, components["g2_cross"], beta2
      ),
      v_xi_ema=_tree_ema(state.v_xi_ema, components["xi2"], beta2),
      step=step,
  )
  return new_state, {
      "V_g": _bias_correct(new_state.v_g_ema, step, beta2),
      "V_g_cross": _bias_correct(new_state.v_g_cross_ema, step, beta2),
      "V_xi": _bias_correct(new_state.v_xi_ema, step, beta2),
  }


def _find_adam_state(value: Any) -> Any:
  """Find Optax's real ScaleByAdamState without depending on tuple layout."""
  if all(hasattr(value, name) for name in ("count", "mu", "nu")):
    return value
  if isinstance(value, (tuple, list)):
    for item in value:
      try:
        return _find_adam_state(item)
      except TypeError:
        continue
  raise TypeError("optimizer_state does not contain an Optax Adam state")


def adam_private_v_hat(optimizer_state: Any, *, beta2: float) -> PyTree:
  """Read the actual Adam second moment from an updated Optax state."""
  adam = _find_adam_state(optimizer_state)
  count = jnp.asarray(adam.count)
  correction = 1.0 - jnp.asarray(beta2) ** count
  correction = jnp.where(count == 0, 1.0, correction)
  return jax.tree_util.tree_map(lambda value: value / correction, adam.nu)


def _decomposition_error(
    private_v_hat: PyTree, ema: dict[str, PyTree]
) -> tuple[jax.Array, jax.Array]:
  expected = _tree_add(ema["V_g_cross"], ema["V_xi"])
  difference = _tree_sub(private_v_hat, expected)
  return _tree_max_abs(difference), _tree_rms(difference)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp10Step:
  """Current-step arrays and scalars exposed to the online collector."""

  instantaneous: dict[str, dict[str, PyTree]]
  ema: dict[str, dict[str, PyTree]]
  metrics: dict[str, dict[str, jax.Array]]
  decomposition_error_max_abs: dict[str, jax.Array]
  decomposition_error_rms: dict[str, jax.Array]
  phi_t: jax.Array

  def tree_flatten(self):
    return (
        self.instantaneous,
        self.ema,
        self.metrics,
        self.decomposition_error_max_abs,
        self.decomposition_error_rms,
        self.phi_t,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def _zero_step(params: PyTree) -> Exp10Step:
  zero = _zeros_like(params)
  instantaneous = {
      branch: {name: zero for name in INSTANTANEOUS_COMPONENTS}
      for branch in BRANCHES
  }
  ema = {
      branch: {name: zero for name in EMA_COMPONENTS} for branch in BRANCHES
  }
  metrics = {
      branch: {
          name: jnp.asarray(0.0, jnp.float32)
          for name in (
              "mean_g2", "mean_g2_cross", "mean_xi2", "mean_2gxi",
              "mean_abs_2gxi", "rms_2gxi",
              "negative_fraction_g2_cross", "R_signed", "R_abs",
              "R_noise", "rho_fb", "mean_g2_cross_minus_g2",
              "num_coordinates",
          )
      } for branch in BRANCHES
  }
  return Exp10Step(
      instantaneous=instantaneous,
      ema=ema,
      metrics=metrics,
      decomposition_error_max_abs={
          branch: jnp.asarray(0.0, jnp.float32) for branch in BRANCHES
      },
      decomposition_error_rms={
          branch: jnp.asarray(0.0, jnp.float32) for branch in BRANCHES
      },
      phi_t=jnp.asarray(0.0, jnp.float32),
  )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp10TrainState:
  """Two real closed-loop DP-AdamW trajectories in one checkpointable state."""

  mf_params: PyTree
  iid_params: PyTree
  mf_optimizer_state: Any
  iid_optimizer_state: Any
  mf_noise_state: BandInvMFNoiseState
  rng_key: jax.Array
  step: jax.Array
  mf_diagnostics: AdamSecondMomentDiagnostics
  iid_diagnostics: AdamSecondMomentDiagnostics
  last_step: Exp10Step
  num_coordinates: int

  def tree_flatten(self):
    return (
        self.mf_params,
        self.iid_params,
        self.mf_optimizer_state,
        self.iid_optimizer_state,
        self.mf_noise_state,
        self.rng_key,
        self.step,
        self.mf_diagnostics,
        self.iid_diagnostics,
        self.last_step,
    ), self.num_coordinates

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, num_coordinates=aux_data)

  @property
  def params(self) -> dict[str, PyTree]:
    """Convenience view of both current parameter pytrees."""
    return {"mf": self.mf_params, "iid": self.iid_params}


def init_exp10_train_state(
    params: PyTree,
    strategy: BandInvMFStrategy,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> Exp10TrainState:
  """Initialize identical parameters and independent AdamW moment trees."""
  if not isinstance(strategy, BandInvMFStrategy):
    raise TypeError("strategy must be a BandInvMFStrategy")
  return Exp10TrainState(
      mf_params=params,
      iid_params=params,
      mf_optimizer_state=optimizer.init(params),
      iid_optimizer_state=optimizer.init(params),
      mf_noise_state=init_bandinv_noise_state(params, strategy.bandwidth),
      rng_key=rng_key,
      step=jnp.asarray(0, jnp.int32),
      num_coordinates=_tree_numel(params),
      mf_diagnostics=init_second_moment_diagnostics(params),
      iid_diagnostics=init_second_moment_diagnostics(params),
      last_step=_zero_step(params),
  )


def make_exp10_train_step(
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
) -> tuple[Callable[[Exp10TrainState, Any], Exp10TrainState], optax.GradientTransformation]:
  """Build the paired real MF/IID clipped-gradient DP-AdamW step."""
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
  phi_schedule = bandinv_marginal_variances(
      strategy, calibration.iid_noise_std
  )

  def train_step(state: Exp10TrainState, batch: Any) -> Exp10TrainState:
    # These are two real queries at the two branches' current parameters.
    g_mf = clipped_query(state.mf_params, batch)
    g_iid = clipped_query(state.iid_params, batch)

    z_t, next_key = sample_shared_latent_noise(state.rng_key, state.mf_params)
    step0 = state.mf_noise_state.step
    phi_t = phi_schedule[step0]
    xi_mf, xi_iid, new_noise_state = paired_noise_from_innovation(
        state.mf_noise_state,
        z_t,
        strategy,
        calibration.iid_noise_std,
        phi_t,
    )

    private_mf = _tree_add(g_mf, xi_mf)
    private_iid = _tree_add(g_iid, xi_iid)
    mf_updates, mf_optimizer_state = optimizer.update(
        private_mf, state.mf_optimizer_state, state.mf_params
    )
    iid_updates, iid_optimizer_state = optimizer.update(
        private_iid, state.iid_optimizer_state, state.iid_params
    )
    mf_params = optax.apply_updates(state.mf_params, mf_updates)
    iid_params = optax.apply_updates(state.iid_params, iid_updates)

    mf_components = second_moment_components(g_mf, xi_mf)
    iid_components = second_moment_components(g_iid, xi_iid)
    mf_diagnostics, mf_ema = advance_second_moment_diagnostics(
        state.mf_diagnostics, mf_components, beta2=beta2
    )
    iid_diagnostics, iid_ema = advance_second_moment_diagnostics(
        state.iid_diagnostics, iid_components, beta2=beta2
    )

    mf_private_v_hat = adam_private_v_hat(
        mf_optimizer_state, beta2=beta2
    )
    iid_private_v_hat = adam_private_v_hat(
        iid_optimizer_state, beta2=beta2
    )
    mf_error = _decomposition_error(mf_private_v_hat, mf_ema)
    iid_error = _decomposition_error(iid_private_v_hat, iid_ema)
    new_step = state.step + jnp.asarray(1, state.step.dtype)
    last_step = Exp10Step(
        instantaneous={"mf": mf_components, "iid": iid_components},
        ema={"mf": mf_ema, "iid": iid_ema},
        metrics={
            "mf": component_metrics(
                mf_components,
                cross_term=jax.tree_util.tree_map(
                    lambda gradient, noise: 2.0 * gradient * noise,
                    g_mf,
                    xi_mf,
                ),
                num_coordinates=state.num_coordinates,
            ),
            "iid": component_metrics(
                iid_components,
                cross_term=jax.tree_util.tree_map(
                    lambda gradient, noise: 2.0 * gradient * noise,
                    g_iid,
                    xi_iid,
                ),
                num_coordinates=state.num_coordinates,
            ),
        },
        decomposition_error_max_abs={"mf": mf_error[0], "iid": iid_error[0]},
        decomposition_error_rms={"mf": mf_error[1], "iid": iid_error[1]},
        phi_t=phi_t,
    )
    return Exp10TrainState(
        mf_params=mf_params,
        iid_params=iid_params,
        mf_optimizer_state=mf_optimizer_state,
        iid_optimizer_state=iid_optimizer_state,
        mf_noise_state=new_noise_state,
        rng_key=next_key,
        step=new_step,
        num_coordinates=state.num_coordinates,
        mf_diagnostics=mf_diagnostics,
        iid_diagnostics=iid_diagnostics,
        last_step=last_step,
    )

  return train_step, optimizer


__all__ = [
    "AdamSecondMomentDiagnostics",
    "BRANCHES",
    "COMPONENTS",
    "EMA_COMPONENTS",
    "Exp10Step",
    "Exp10TrainState",
    "INSTANTANEOUS_COMPONENTS",
    "adam_private_v_hat",
    "advance_second_moment_diagnostics",
    "bandinv_marginal_variances",
    "component_identity_residual",
    "component_metrics",
    "init_exp10_train_state",
    "init_second_moment_diagnostics",
    "make_exp10_train_step",
    "paired_noise_from_innovation",
    "sample_shared_latent_noise",
    "second_moment_components",
]
