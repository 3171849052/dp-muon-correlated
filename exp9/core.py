"""Core mechanics for Experiment 9.

The production path is the existing non-amplified BandInvMF DP-Muon update.
The diagnostic path is shadow-only: it receives the one clipped-but-unnoised
gradient query, applies the same classic-Nesterov linear frontend, and then
evaluates the smooth float32 Muon Q map in four ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

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
from dp_muon.optim import (
    muon_q,
    muon_q_stages,
    vit_muon_parameter_labels,
)
from dp_muon.optim.linear_workload import nesterov_kernel_coef
from dp_muon.privacy import (
    ParticipationSpec,
    PrivacyCalibration,
    make_clipped_gradient_query,
)
from dp_muon.training.nonamplified_dpmuon import make_nonamplified_dpmuon_optimizer
from dp_muon.training.nonamplified_linear import validate_nonamplified_bandinv_privacy_setup


PyTree = Any
BRANCHES = ("corr", "iid")
PATHS = ("P0", "P1", "P2", "P3")
STAGES = ("linear", "bf16", "norm", "ns", "scale")
PRIMARY_STAGES = ("linear", "norm", "ns", "scale")


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add(left: PyTree, right: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_scale(scale: Any, tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda value: scale * value, tree)


def _tree_norm_sq(tree: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(leaf, jnp.float32) ** 2)
      for leaf in jax.tree_util.tree_leaves(tree)
  )


def _tree_norm(tree: PyTree) -> jax.Array:
  return jnp.sqrt(_tree_norm_sq(tree))


def _path_key(path: tuple[Any, ...]) -> str:
  """Stable human-readable key for a flattened JAX parameter path."""
  parts = []
  for entry in path:
    if hasattr(entry, "key"):
      parts.append(str(entry.key))
    elif hasattr(entry, "idx"):
      parts.append(str(entry.idx))
    else:
      parts.append(str(entry))
  return "/".join(parts)


def muon_parameter_paths(params: PyTree) -> tuple[tuple[Any, ...], ...]:
  """Returns exactly the parameter paths labelled ``muon`` by the project."""
  labels = vit_muon_parameter_labels(params)
  return tuple(
      path for path, label in jax.tree_util.tree_leaves_with_path(labels)
      if label == "muon"
  )


def _get_path(tree: PyTree, path: tuple[Any, ...]) -> Any:
  value = tree
  for entry in path:
    if hasattr(entry, "key"):
      value = value[entry.key]
    elif hasattr(entry, "idx"):
      value = value[entry.idx]
    else:
      raise TypeError(f"unsupported JAX path entry {entry!r}")
  return value


def extract_muon_blocks(tree: PyTree, paths: Sequence[tuple[Any, ...]]) -> dict[str, jax.Array]:
  """Extracts the Muon-labelled rank-two leaves using fixed parameter paths."""
  result = {}
  for path in paths:
    value = jnp.asarray(_get_path(tree, path))
    if value.ndim != 2:
      raise ValueError(f"Muon leaf {_path_key(path)!r} must be rank two")
    result[_path_key(path)] = value
  return result


def classic_nesterov_frontend(
    previous_momentum: jax.Array,
    gradient: jax.Array,
    momentum: float,
) -> tuple[jax.Array, jax.Array]:
  """One exact ``classic_nesterov_momentum`` state/update pair.

  The production transform is ``scale(1-beta)`` followed by Optax trace with
  ``nesterov=True``.  Consequently ``m=(1-beta)g+beta*m_prev`` and the output
  is ``(1-beta)g+beta*m``.  No bias correction is present.
  """
  beta = jnp.asarray(momentum, dtype=jnp.asarray(gradient).dtype)
  new_momentum = (1.0 - beta) * gradient + beta * previous_momentum
  update = (1.0 - beta) * gradient + beta * new_momentum
  return new_momentum, update


def linear_frontend(
    gradients: Sequence[jax.Array], momentum: float
) -> tuple[list[jax.Array], list[jax.Array]]:
  """Applies the frontend to a sequence, useful for exact regression tests."""
  if not gradients:
    return [], []
  previous = jnp.zeros_like(gradients[0])
  states, outputs = [], []
  for gradient in gradients:
    previous, output = classic_nesterov_frontend(previous, gradient, momentum)
    states.append(previous)
    outputs.append(output)
  return states, outputs


def bandinv_marginal_variances(
    strategy: BandInvMFStrategy, iid_noise_std: float | jax.Array
) -> jax.Array:
  """Raw per-step marginal variance of a streamed BandInvMF output."""
  coef = jnp.asarray(strategy.noising_coef)
  if coef.ndim != 1 or coef.shape != (strategy.bandwidth,):
    raise ValueError("strategy.noising_coef must match strategy.bandwidth")
  sigma = jnp.asarray(iid_noise_std, dtype=coef.dtype)
  if sigma.ndim != 0:
    raise ValueError("iid_noise_std must be scalar")
  cumulative = jnp.cumsum(jnp.square(coef))
  row_index = jnp.minimum(jnp.arange(strategy.horizon), strategy.bandwidth - 1)
  return jnp.square(sigma) * cumulative[row_index]


def pre_q_marginal_variances(
    strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    momentum: float,
) -> dict[str, jax.Array]:
  """Returns actual pre-Q variances for correlated and matched-IID branches.

  The IID control matches the *raw gradient-noise* marginal at every step.
  The returned ``pre_q_*`` values additionally include the classic-Nesterov
  frontend, so their square roots are the ``rho_t^b`` used by bias probes.
  """
  raw = bandinv_marginal_variances(strategy, iid_noise_std)
  h = jnp.asarray(nesterov_kernel_coef(strategy.horizon, momentum), dtype=raw.dtype)
  coef = jnp.asarray(strategy.noising_coef, dtype=raw.dtype)
  corr_filter = jnp.convolve(h, coef)[: strategy.horizon]
  corr = jnp.square(jnp.asarray(iid_noise_std, raw.dtype)) * jnp.cumsum(
      jnp.square(corr_filter)
  )
  # IID raw steps have variance raw[t], while the same causal Nesterov kernel
  # mixes those independent innovations at the current pre-Q step.
  iid = jnp.asarray([
      jnp.sum(jnp.square(h[: t + 1]) * raw[t::-1])
      for t in range(strategy.horizon)
  ])
  return {"raw_corr": raw, "raw_iid": raw, "pre_q_corr": corr, "pre_q_iid": iid}


def _standard_normal_tree(key: jax.Array, template: PyTree) -> PyTree:
  leaves, treedef = jax.tree_util.tree_flatten(template)
  keys = jax.random.split(key, len(leaves))
  values = [
      jax.random.normal(subkey, jnp.asarray(leaf).shape, dtype=jnp.asarray(leaf).dtype)
      for subkey, leaf in zip(keys, leaves, strict=True)
  ]
  return jax.tree_util.tree_unflatten(treedef, values)


def paired_diagnostic_noise_from_innovation(
    corr_state: BandInvMFNoiseState,
    z: PyTree,
    diagnostic_strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    raw_marginal_variance: float | jax.Array,
    *,
    noising_coef: jax.Array | None = None,
) -> tuple[PyTree, PyTree, BandInvMFNoiseState]:
  """Build correlated and matched-raw-marginal IID noise from the same ``z``."""
  coefficient = (
      jnp.asarray(diagnostic_strategy.noising_coef)
      if noising_coef is None else jnp.asarray(noising_coef)
  )
  latent = jax.tree_util.tree_map(
      lambda value: jnp.asarray(iid_noise_std) * value, z
  )
  corr, new_state = filter_latent_noise(corr_state, latent, coefficient)
  iid_scale = jnp.sqrt(jnp.asarray(raw_marginal_variance))
  iid = jax.tree_util.tree_map(lambda value: iid_scale * value, z)
  return corr, iid, new_state


def sample_paired_diagnostic_noise(
    key: jax.Array,
    corr_state: BandInvMFNoiseState,
    diagnostic_strategy: BandInvMFStrategy,
    iid_noise_std: float | jax.Array,
    raw_marginal_variance: float | jax.Array,
) -> tuple[PyTree, PyTree, PyTree, BandInvMFNoiseState, jax.Array]:
  """Sample one paired diagnostic step; only the correlated branch has state."""
  leaves = jax.tree_util.tree_leaves(corr_state.buffer)
  if not leaves:
    raise ValueError("diagnostic noise state must contain at least one leaf")
  z_key, new_key = jax.random.split(key)
  template = jax.tree_util.tree_map(
      lambda buffer: jnp.zeros(buffer.shape[1:], dtype=buffer.dtype), corr_state.buffer
  )
  z = _standard_normal_tree(z_key, template)
  coef = jnp.asarray(diagnostic_strategy.noising_coef)
  runtime_coef = coef + corr_state.step.astype(coef.dtype) * jnp.zeros_like(coef)
  corr, iid, new_state = paired_diagnostic_noise_from_innovation(
      corr_state, z, diagnostic_strategy, iid_noise_std, raw_marginal_variance,
      noising_coef=runtime_coef,
  )
  return corr, iid, z, new_state, new_key


def _smooth_muon_q(
    matrix: jax.Array,
    *,
    ns_steps: int,
    consistent_rms: float,
) -> jax.Array:
  return smooth_muon_q(matrix, ns_steps=ns_steps, consistent_rms=consistent_rms)


def smooth_muon_q(
    matrix: jax.Array,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
) -> jax.Array:
  """Primary analysis Muon Q: float32 throughout, with no BF16 NS cast."""
  # Primary diagnostics intentionally never inherit production BF16 casts.
  return muon_q(
      jnp.asarray(matrix, jnp.float32), ns_steps=ns_steps,
      consistent_rms=consistent_rms, use_bf16_ns=False,
  ).astype(jnp.float32)


def _probe_normals(
    key: jax.Array, x: jax.Array, *, replicates: int, probes: int
) -> jax.Array:
  """Generate a ``(replicates, probes, *x.shape)`` probe batch."""
  keys = jax.random.split(key, replicates * probes)
  values = jax.vmap(
      lambda subkey: jax.random.normal(subkey, x.shape, dtype=jnp.float32)
  )(keys)
  return values.reshape((replicates, probes, *x.shape))


def _bias_from_probes(
    clean_pre_q: jax.Array,
    rho: float | jax.Array,
    probe_u: jax.Array,
    *,
    f0: jax.Array | None = None,
    ns_steps: int,
    consistent_rms: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Return ``(B_hat, B_A, B_B)`` using one shared probe batch."""
  x = jnp.asarray(clean_pre_q, jnp.float32)
  u = jnp.asarray(probe_u, jnp.float32)
  if u.ndim != x.ndim + 2 or u.shape[0] != 2:
    raise ValueError("probe_u must have shape (2, K, *clean_pre_q.shape)")
  rho = jnp.asarray(rho, jnp.float32)
  f = lambda value: _smooth_muon_q(
      value, ns_steps=ns_steps, consistent_rms=consistent_rms
  )
  if f0 is None:
    f0 = f(x)

  def one_replicate(probes_for_replicate: jax.Array) -> jax.Array:
    plus = jax.vmap(f)(x + rho * probes_for_replicate)
    minus = jax.vmap(f)(x - rho * probes_for_replicate)
    return jnp.mean((plus + minus) * jnp.asarray(.5, jnp.float32), axis=0) - f0

  replicate_bias = jax.vmap(one_replicate)(u)
  bias_a, bias_b = replicate_bias[0], replicate_bias[1]
  return (bias_a + bias_b) * jnp.asarray(.5, jnp.float32), bias_a, bias_b


def estimate_output_bias_replicates(
    clean_pre_q: jax.Array,
    rho: float | jax.Array,
    key: jax.Array,
    *,
    probes: int = 8,
    replicates: int = 2,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Estimate output bias with independent A/B replicates in one vmap."""
  if isinstance(probes, bool) or not isinstance(probes, int) or probes < 1:
    raise ValueError("probes must be a positive integer")
  if replicates != 2:
    raise ValueError("Experiment 9 requires exactly two bias probe replicates")
  x = jnp.asarray(clean_pre_q, jnp.float32)
  f0 = _smooth_muon_q(x, ns_steps=ns_steps, consistent_rms=consistent_rms)
  probe_u = _probe_normals(key, x, replicates=replicates, probes=probes)
  return _bias_from_probes(
      x, rho, probe_u, f0=f0, ns_steps=ns_steps,
      consistent_rms=consistent_rms,
  )


def estimate_output_bias(
    clean_pre_q: jax.Array,
    rho: float | jax.Array,
    key: jax.Array,
    *,
    probes: int = 8,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
) -> jax.Array:
  """Return only ``B_hat``; A/B details are available from the replicate API."""
  return estimate_output_bias_replicates(
      clean_pre_q, rho, key, probes=probes, replicates=2,
      ns_steps=ns_steps, consistent_rms=consistent_rms,
  )[0]


def nonlinear_response_decomposition(
    clean_pre_q: jax.Array,
    noise_pre_q: jax.Array,
    *,
    bias_key: jax.Array | None = None,
    rho: float | jax.Array = 0.0,
    probes: int = 8,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    probe_u: jax.Array | None = None,
    f0: jax.Array | None = None,
    plus: jax.Array | None = None,
    minus: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Return P0--P3, raw Y, even response, and independently probed bias."""
  x = jnp.asarray(clean_pre_q, jnp.float32)
  r = jnp.asarray(noise_pre_q, jnp.float32)
  f = lambda value: _smooth_muon_q(
      value, ns_steps=ns_steps, consistent_rms=consistent_rms
  )
  f0 = f(x) if f0 is None else f0
  plus = f(x + r) if plus is None else plus
  minus = f(x - r) if minus is None else minus
  _, p1 = jax.jvp(f, (x,), (r,))
  p2 = (plus - minus) * jnp.asarray(0.5, jnp.float32)
  raw = plus - f0
  even = (plus + minus) * jnp.asarray(0.5, jnp.float32) - f0
  if bias_key is None and probe_u is None:
    bias = jnp.zeros_like(x)
    bias_a = jnp.zeros_like(x)
    bias_b = jnp.zeros_like(x)
  else:
    if probe_u is None:
      probe_u = _probe_normals(bias_key, x, replicates=2, probes=probes)
    bias, bias_a, bias_b = _bias_from_probes(
        x, rho, probe_u, f0=f0, ns_steps=ns_steps,
        consistent_rms=consistent_rms,
    )
  p0 = _consistent_rms_scale(x, consistent_rms) * r
  return {"P0": p0, "P1": p1, "P2": p2, "P3": raw - bias,
          "Y": raw, "even": even, "bias": bias, "bias_A": bias_a,
          "bias_B": bias_b, "probe_disagreement": bias_a - bias_b,
          "odd_reconstruction_error": raw - p2 - even}


def _stage_odd_response(
    x: jax.Array,
    r: jax.Array,
    *,
    ns_steps: int,
    consistent_rms: float,
    use_bf16_ns: bool,
) -> dict[str, jax.Array]:
  plus = muon_q_stages(
      x + r, ns_steps=ns_steps, consistent_rms=consistent_rms,
      use_bf16_ns=use_bf16_ns,
  )
  minus = muon_q_stages(
      x - r, ns_steps=ns_steps, consistent_rms=consistent_rms,
      use_bf16_ns=use_bf16_ns,
  )
  return {stage: (plus[stage] - minus[stage]) * 0.5 for stage in STAGES}


def _consistent_rms_scale(matrix: jax.Array, consistent_rms: float) -> jax.Array:
  """The fixed parameter-axis scale used by production Muon."""
  return jnp.asarray(consistent_rms, jnp.float32) * jnp.sqrt(
      jnp.asarray(max(matrix.shape), jnp.float32)
  )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp9DiagnosticStep:
  """One shadow step; all maps in ``x`` contain Muon leaves only."""

  x: dict[str, dict[str, dict[str, jax.Array]]]
  clean_pre_q: dict[str, jax.Array]
  noise_pre_q: dict[str, dict[str, jax.Array]]
  raw_response: dict[str, dict[str, jax.Array]]
  bias: dict[str, dict[str, jax.Array]]
  bias_A: dict[str, dict[str, jax.Array]]
  bias_B: dict[str, dict[str, jax.Array]]
  probe_disagreement: dict[str, dict[str, jax.Array]]
  probe_disagreement_norm: dict[str, jax.Array]
  probe_disagreement_relative_to_bias: dict[str, jax.Array]
  even_response: dict[str, dict[str, jax.Array]]
  stage_odd: dict[str, dict[str, dict[str, jax.Array]]]
  secondary_stage_odd: dict[str, dict[str, dict[str, jax.Array]]]
  block_ratio_mean: dict[str, jax.Array]
  block_ratio_max: dict[str, jax.Array]
  global_noise_signal_ratio: dict[str, jax.Array]
  clean_pre_q_norm: dict[str, jax.Array]
  clean_pre_q_norm_min: jax.Array
  phi_pre_q: dict[str, jax.Array]
  rho_pre_q: dict[str, jax.Array]
  clipped_clean_gradient: PyTree
  odd_reconstruction_error: dict[str, jax.Array]

  def tree_flatten(self):
    return (
        self.x, self.clean_pre_q, self.noise_pre_q, self.raw_response, self.bias,
        self.bias_A, self.bias_B, self.probe_disagreement,
        self.probe_disagreement_norm, self.probe_disagreement_relative_to_bias,
        self.even_response,
        self.stage_odd, self.secondary_stage_odd, self.block_ratio_mean,
        self.block_ratio_max, self.global_noise_signal_ratio,
        self.clean_pre_q_norm, self.clean_pre_q_norm_min,
        self.phi_pre_q, self.rho_pre_q,
        self.clipped_clean_gradient, self.odd_reconstruction_error,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)

  @property
  def Y(self) -> dict[str, dict[str, jax.Array]]:
    """Raw private-clean response, retained as a named formula alias."""
    return self.raw_response

  @property
  def B_hat(self) -> dict[str, dict[str, jax.Array]]:
    """Independent output-bias estimate, retained as a named formula alias."""
    return self.bias


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp9ShadowState:
  clean_momentum: dict[str, jax.Array]
  corr_noise_momentum: dict[str, jax.Array]
  iid_noise_momentum: dict[str, jax.Array]
  step: jax.Array

  def tree_flatten(self):
    return (self.clean_momentum, self.corr_noise_momentum,
            self.iid_noise_momentum, self.step), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_exp9_shadow_state(muon_blocks: Mapping[str, jax.Array]) -> Exp9ShadowState:
  zeros = {key: jnp.zeros_like(value) for key, value in muon_blocks.items()}
  return Exp9ShadowState(
      clean_momentum=zeros,
      corr_noise_momentum={key: value for key, value in zeros.items()},
      iid_noise_momentum={key: value for key, value in zeros.items()},
      step=jnp.asarray(0, jnp.int32),
  )


def _zero_step(params: PyTree, muon_blocks: Mapping[str, jax.Array]) -> Exp9DiagnosticStep:
  zero_blocks = {key: jnp.zeros_like(value) for key, value in muon_blocks.items()}
  zero_x = {branch: {path: dict(zero_blocks) for path in PATHS} for branch in BRANCHES}
  zero_branch = {branch: dict(zero_blocks) for branch in BRANCHES}
  zero_stages = {
      branch: {stage: dict(zero_blocks) for stage in STAGES} for branch in BRANCHES
  }
  return Exp9DiagnosticStep(
      x=zero_x, clean_pre_q=dict(zero_blocks), noise_pre_q=zero_branch,
      raw_response=zero_branch, bias=zero_branch, bias_A=zero_branch,
      bias_B=zero_branch, probe_disagreement=zero_branch, even_response=zero_branch,
      probe_disagreement_norm={branch: jnp.asarray(0.0) for branch in BRANCHES},
      probe_disagreement_relative_to_bias={
          branch: jnp.asarray(0.0) for branch in BRANCHES
      },
      stage_odd=zero_stages, secondary_stage_odd=zero_stages,
      block_ratio_mean={branch: jnp.asarray(0.0) for branch in BRANCHES},
      block_ratio_max={branch: jnp.asarray(0.0) for branch in BRANCHES},
      global_noise_signal_ratio={branch: jnp.asarray(0.0) for branch in BRANCHES},
      clean_pre_q_norm={key: jnp.asarray(0.0) for key in zero_blocks},
      clean_pre_q_norm_min=jnp.asarray(0.0),
      phi_pre_q={branch: jnp.asarray(0.0) for branch in BRANCHES},
      rho_pre_q={branch: jnp.asarray(0.0) for branch in BRANCHES},
      clipped_clean_gradient=_zeros_like(params),
      odd_reconstruction_error={branch: jnp.asarray(0.0) for branch in BRANCHES},
  )


def advance_exp9_diagnostic(
    state: Exp9ShadowState,
    clean_gradient: Mapping[str, jax.Array],
    xi_corr: Mapping[str, jax.Array],
    xi_iid: Mapping[str, jax.Array],
    phi_pre_q: Mapping[str, float | jax.Array],
    bias_key: jax.Array,
    *,
    momentum: float,
    learning_rate: float,
    ns_steps: int = 5,
    consistent_rms: float = 0.2,
    probes: int = 8,
    secondary_use_bf16_ns: bool = True,
    clipped_clean_gradient: PyTree | None = None,
) -> tuple[Exp9ShadowState, Exp9DiagnosticStep]:
  """Advance the Muon-only shadow and evaluate all primary/secondary paths."""
  if isinstance(probes, bool) or not isinstance(probes, int) or probes < 1:
    raise ValueError("probes must be a positive integer")
  keys = tuple(clean_gradient)
  if tuple(xi_corr) != keys or tuple(xi_iid) != keys:
    raise ValueError("clean and diagnostic Muon block keys must match")
  t = state.step + jnp.asarray(1, state.step.dtype)
  clean_m, clean_x = {}, {}
  corr_m, corr_r = {}, {}
  iid_m, iid_r = {}, {}
  for key in keys:
    clean_m[key], clean_x[key] = classic_nesterov_frontend(
        state.clean_momentum[key], clean_gradient[key], momentum
    )
    corr_m[key], corr_r[key] = classic_nesterov_frontend(
        state.corr_noise_momentum[key], xi_corr[key], momentum
    )
    iid_m[key], iid_r[key] = classic_nesterov_frontend(
        state.iid_noise_momentum[key], xi_iid[key], momentum
    )

  branch_r = {"corr": corr_r, "iid": iid_r}
  branch_phi = {branch: jnp.asarray(phi_pre_q[branch], jnp.float32)
                for branch in BRANCHES}
  branch_x = {branch: {path: {} for path in PATHS} for branch in BRANCHES}
  branch_raw, branch_bias, branch_bias_a, branch_bias_b = {}, {}, {}, {}
  branch_probe_disagreement, branch_even = {}, {}
  branch_stage, branch_secondary = {}, {}
  branch_block_ratios, branch_global_ratios = {}, {}
  reconstruction = {}
  clean_norm = {key: jnp.linalg.norm(clean_x[key]) for key in keys}
  probe_keys = jax.random.split(bias_key, len(keys))
  probe_u_by_key = {
      key: _probe_normals(block_key, clean_x[key], replicates=2, probes=probes)
      for key, block_key in zip(keys, probe_keys, strict=True)
  }
  primary_clean_stages = {
      key: muon_q_stages(
          clean_x[key], ns_steps=ns_steps, consistent_rms=consistent_rms,
          use_bf16_ns=False,
      ) for key in keys
  }
  for branch_index, branch in enumerate(BRANCHES):
    del branch_index
    branch_raw[branch], branch_bias[branch] = {}, {}
    branch_bias_a[branch], branch_bias_b[branch] = {}, {}
    branch_probe_disagreement[branch], branch_even[branch] = {}, {}
    branch_stage[branch] = {stage: {} for stage in STAGES}
    branch_secondary[branch] = {stage: {} for stage in STAGES}
    reconstruction_terms = []
    for key in keys:
      rho = jnp.sqrt(jnp.maximum(branch_phi[branch], 0.0))
      primary_plus = muon_q_stages(
          clean_x[key] + branch_r[branch][key], ns_steps=ns_steps,
          consistent_rms=consistent_rms, use_bf16_ns=False,
      )
      primary_minus = muon_q_stages(
          clean_x[key] - branch_r[branch][key], ns_steps=ns_steps,
          consistent_rms=consistent_rms, use_bf16_ns=False,
      )
      decomp = nonlinear_response_decomposition(
          clean_x[key], branch_r[branch][key], probe_u=probe_u_by_key[key], rho=rho,
          f0=primary_clean_stages[key]["scale"],
          plus=primary_plus["scale"], minus=primary_minus["scale"],
          probes=probes, ns_steps=ns_steps, consistent_rms=consistent_rms,
      )
      # P0 is the same linear response with Muon's fixed parameter-axis
      # consistent-RMS block scale.  It is not a new temporal workload.
      decomp["P0"] = _consistent_rms_scale(clean_x[key], consistent_rms) * branch_r[branch][key]
      for path in PATHS:
        branch_x[branch][path][key] = -jnp.asarray(learning_rate, jnp.float32) * decomp[path]
      branch_raw[branch][key] = decomp["Y"]
      branch_bias[branch][key] = decomp["bias"]
      branch_bias_a[branch][key] = decomp["bias_A"]
      branch_bias_b[branch][key] = decomp["bias_B"]
      branch_probe_disagreement[branch][key] = decomp["probe_disagreement"]
      branch_even[branch][key] = decomp["even"]
      reconstruction_terms.append(_tree_norm({"value": decomp["odd_reconstruction_error"]}))
      primary_stages = {
          stage: (primary_plus[stage] - primary_minus[stage]) * 0.5
          for stage in STAGES
      }
      if secondary_use_bf16_ns:
        secondary_plus = muon_q_stages(
            clean_x[key] + branch_r[branch][key], ns_steps=ns_steps,
            consistent_rms=consistent_rms, use_bf16_ns=True,
        )
        secondary_minus = muon_q_stages(
            clean_x[key] - branch_r[branch][key], ns_steps=ns_steps,
            consistent_rms=consistent_rms, use_bf16_ns=True,
        )
        secondary_stages = {
            stage: (secondary_plus[stage] - secondary_minus[stage]) * 0.5
            for stage in STAGES
        }
      else:
        secondary_stages = primary_stages
      for stage in STAGES:
        branch_stage[branch][stage][key] = primary_stages[stage]
        branch_secondary[branch][stage][key] = secondary_stages[stage]
    noise_sq = sum(jnp.sum(branch_r[branch][key] ** 2) for key in keys)
    signal_sq = sum(jnp.sum(clean_x[key] ** 2) for key in keys)
    branch_global_ratios[branch] = jnp.sqrt(noise_sq) / (
        jnp.sqrt(signal_sq) + jnp.asarray(1e-12, jnp.float32)
    )
    ratios = {
        key: jnp.linalg.norm(branch_r[branch][key]) / (
            clean_norm[key] + jnp.asarray(1e-12, jnp.float32)
        ) for key in keys
    }
    branch_block_ratios[branch] = ratios
    reconstruction[branch] = jnp.sqrt(sum(value * value for value in reconstruction_terms))
  probe_norm = {
      branch: _tree_norm(branch_probe_disagreement[branch]) for branch in BRANCHES
  }
  bias_norm = {branch: _tree_norm(branch_bias[branch]) for branch in BRANCHES}
  probe_relative = {
      branch: probe_norm[branch] / (bias_norm[branch] + jnp.asarray(1e-12, jnp.float32))
      for branch in BRANCHES
  }
  block_ratio_mean = {
      branch: (jnp.mean(jnp.stack(tuple(ratios.values()))) if ratios else jnp.asarray(0.0))
      for branch, ratios in branch_block_ratios.items()
  }
  block_ratio_max = {
      branch: (jnp.max(jnp.stack(tuple(ratios.values()))) if ratios else jnp.asarray(0.0))
      for branch, ratios in branch_block_ratios.items()
  }
  new_state = Exp9ShadowState(
      clean_momentum=clean_m, corr_noise_momentum=corr_m,
      iid_noise_momentum=iid_m, step=t,
  )
  return new_state, Exp9DiagnosticStep(
      x=branch_x, clean_pre_q=clean_x, noise_pre_q=branch_r,
      raw_response=branch_raw, bias=branch_bias, bias_A=branch_bias_a,
      bias_B=branch_bias_b, probe_disagreement=branch_probe_disagreement,
      probe_disagreement_norm=probe_norm,
      probe_disagreement_relative_to_bias=probe_relative,
      even_response=branch_even,
      stage_odd=branch_stage, secondary_stage_odd=branch_secondary,
      block_ratio_mean=block_ratio_mean, block_ratio_max=block_ratio_max,
      global_noise_signal_ratio=branch_global_ratios,
      clean_pre_q_norm=clean_norm,
      clean_pre_q_norm_min=(jnp.min(jnp.stack(tuple(clean_norm.values())))
                            if clean_norm else jnp.asarray(0.0)),
      phi_pre_q=branch_phi,
      rho_pre_q={branch: jnp.sqrt(jnp.maximum(branch_phi[branch], 0.0))
                 for branch in BRANCHES},
      clipped_clean_gradient=(_zeros_like(clean_gradient)
                              if clipped_clean_gradient is None else clipped_clean_gradient),
      odd_reconstruction_error=reconstruction,
  )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp9TrainState:
  """Real correlated DP-Muon state plus independent shadow streams."""

  params: PyTree
  optimizer_state: Any
  training_noise_state: BandInvMFNoiseState
  training_rng_key: jax.Array
  diagnostic_noise_state: BandInvMFNoiseState
  diagnostic_rng_key: jax.Array
  bias_probe_rng_key: jax.Array
  step: jax.Array
  shadow: Exp9ShadowState
  last_step: Exp9DiagnosticStep

  def tree_flatten(self):
    return (self.params, self.optimizer_state, self.training_noise_state,
            self.training_rng_key, self.diagnostic_noise_state,
            self.diagnostic_rng_key, self.bias_probe_rng_key, self.step,
            self.shadow, self.last_step), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)

  @property
  def noise_state(self) -> BandInvMFNoiseState:
    return self.training_noise_state

  @property
  def rng_key(self) -> jax.Array:
    return self.training_rng_key


def init_exp9_train_state(
    params: PyTree,
    training_strategy: BandInvMFStrategy,
    training_rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
    diagnostic_rng_key: jax.Array,
    *,
    bias_probe_rng_key: jax.Array | None = None,
    diagnostic_strategy: BandInvMFStrategy,
) -> Exp9TrainState:
  """Initialise independent production, diagnostic, and bias-probe streams."""
  if bias_probe_rng_key is None:
    bias_probe_rng_key = jax.random.fold_in(diagnostic_rng_key, 91_991)
  if jnp.array_equal(training_rng_key, diagnostic_rng_key) or jnp.array_equal(
      training_rng_key, bias_probe_rng_key
  ) or jnp.array_equal(diagnostic_rng_key, bias_probe_rng_key):
    raise ValueError("training, diagnostic, and bias-probe RNG keys must be independent")
  paths = muon_parameter_paths(params)
  blocks = extract_muon_blocks(params, paths)
  return Exp9TrainState(
      params=params, optimizer_state=optimizer.init(params),
      training_noise_state=init_bandinv_noise_state(params, training_strategy.bandwidth),
      training_rng_key=training_rng_key,
      diagnostic_noise_state=init_bandinv_noise_state(params, diagnostic_strategy.bandwidth),
      diagnostic_rng_key=diagnostic_rng_key, bias_probe_rng_key=bias_probe_rng_key,
      step=jnp.asarray(0, jnp.int32), shadow=init_exp9_shadow_state(blocks),
      last_step=_zero_step(params, blocks),
  )


def make_exp9_train_step(
    loss_fn: Callable[..., Any],
    training_strategy: BandInvMFStrategy,
    training_calibration: PrivacyCalibration,
    participation_spec: ParticipationSpec,
    *,
    diagnostic_strategy: BandInvMFStrategy,
    diagnostic_calibration: PrivacyCalibration,
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
    probes: int = 8,
    secondary_use_bf16_ns: bool = True,
) -> tuple[Callable[[Exp9TrainState, Any], Exp9TrainState], optax.GradientTransformation]:
  """Build the one-query production Muon step and shadow diagnostics."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  validate_nonamplified_bandinv_privacy_setup(
      training_strategy, training_calibration, participation_spec
  )
  validate_nonamplified_bandinv_privacy_setup(
      diagnostic_strategy, diagnostic_calibration, participation_spec
  )
  optimizer = make_nonamplified_dpmuon_optimizer(
      muon_learning_rate=muon_learning_rate, muon_weight_decay=muon_weight_decay,
      momentum=momentum, ns_steps=ns_steps, consistent_rms=consistent_rms,
      adamw_learning_rate=adamw_learning_rate, adamw_beta1=adamw_beta1,
      adamw_beta2=adamw_beta2, adamw_eps=adamw_eps,
      adamw_weight_decay=adamw_weight_decay,
      use_bf16_ns=secondary_use_bf16_ns,
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn, clip_norm=training_calibration.clip_norm,
      normalize_by=training_calibration.normalize_by, batch_argnums=1,
      keep_batch_dim=True, microbatch_size=microbatch_size,
  )
  variance = pre_q_marginal_variances(
      diagnostic_strategy, diagnostic_calibration.iid_noise_std, momentum
  )
  paths_holder: list[tuple[Any, ...]] | None = None

  def train_step(state: Exp9TrainState, batch: Any) -> Exp9TrainState:
    nonlocal paths_holder
    # The sole clipped gradient query in the complete step.
    clipped = clipped_query(state.params, batch)
    paths = muon_parameter_paths(state.params) if paths_holder is None else tuple(paths_holder)
    if paths_holder is None:
      paths_holder = list(paths)
    clean_blocks = extract_muon_blocks(clipped, paths)

    train_step0 = state.training_noise_state.step
    train_coef = jnp.asarray(training_strategy.noising_coef)
    train_coef = train_coef + train_step0.astype(train_coef.dtype) * jnp.zeros_like(train_coef)
    train_sigma = jnp.asarray(training_calibration.iid_noise_std)
    train_sigma = train_sigma + train_step0.astype(train_sigma.dtype) * jnp.zeros_like(train_sigma)
    training_noise, next_training_noise, next_training_key = sample_bandinv_noise(
        state.training_rng_key, state.training_noise_state, train_coef, train_sigma
    )
    private_grad = jax.tree_util.tree_map(lambda g, n: g + n, clipped, training_noise)
    updates, next_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    next_params = optax.apply_updates(state.params, updates)

    diagnostic_step0 = state.diagnostic_noise_state.step
    raw_phi = variance["raw_corr"][diagnostic_step0]
    diagnostic_sigma = jnp.asarray(diagnostic_calibration.iid_noise_std)
    diagnostic_sigma = diagnostic_sigma + diagnostic_step0.astype(diagnostic_sigma.dtype) * jnp.zeros_like(diagnostic_sigma)
    diagnostic_noise, iid_noise, _, next_diagnostic_noise, next_diagnostic_key = sample_paired_diagnostic_noise(
        state.diagnostic_rng_key, state.diagnostic_noise_state, diagnostic_strategy,
        diagnostic_sigma, raw_phi
    )
    corr_blocks = extract_muon_blocks(diagnostic_noise, paths)
    iid_blocks = extract_muon_blocks(iid_noise, paths)
    phi_pre_q = {branch: variance[f"pre_q_{branch}"][diagnostic_step0]
                 for branch in BRANCHES}
    next_bias_key, current_bias_key = jax.random.split(state.bias_probe_rng_key)
    next_shadow, last = advance_exp9_diagnostic(
        state.shadow, clean_blocks, corr_blocks, iid_blocks, phi_pre_q,
        current_bias_key, momentum=momentum, learning_rate=muon_learning_rate,
        ns_steps=ns_steps, consistent_rms=consistent_rms, probes=probes,
        secondary_use_bf16_ns=secondary_use_bf16_ns,
        clipped_clean_gradient=clipped,
    )
    return Exp9TrainState(
        params=next_params, optimizer_state=next_optimizer_state,
        training_noise_state=next_training_noise,
        training_rng_key=next_training_key,
        diagnostic_noise_state=next_diagnostic_noise,
        diagnostic_rng_key=next_diagnostic_key,
        bias_probe_rng_key=next_bias_key,
        step=state.step + jnp.asarray(1, state.step.dtype),
        shadow=next_shadow, last_step=last,
    )

  return train_step, optimizer


__all__ = [
    "BRANCHES", "PATHS", "STAGES", "Exp9DiagnosticStep", "Exp9ShadowState",
    "Exp9TrainState", "advance_exp9_diagnostic", "bandinv_marginal_variances",
    "classic_nesterov_frontend", "estimate_output_bias", "extract_muon_blocks",
    "init_exp9_shadow_state", "init_exp9_train_state", "linear_frontend",
    "make_exp9_train_step", "muon_parameter_paths", "nonlinear_response_decomposition",
    "paired_diagnostic_noise_from_innovation", "pre_q_marginal_variances",
    "sample_paired_diagnostic_noise", "smooth_muon_q",
]
