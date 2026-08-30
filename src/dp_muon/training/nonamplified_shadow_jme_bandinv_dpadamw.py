"""DP-AdamW with correlated warmup and shadow-second-moment JME segments.

The module intentionally has no dependency on the historical frozen-p or
segmented trainers.  Warmup runs ordinary Optax AdamW.  After warmup,
``FrozenPAdamW`` consumes only the first JME channel while an independent
second channel updates ``v_shadow``.  A segment boundary is a host-side
strategy boundary; :func:`begin_shadow_jme_segment` can be called by an
orchestrator after fitting the next pair of strategies from the DP shadow
output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.bandinvmf import (
    BandInvMFNoiseState,
    BandInvMFStrategy,
    fit_bandinv_strategy,
    init_bandinv_noise_state,
    sample_bandinv_noise,
)
from dp_muon.optim import (
    FrozenPAdamW,
    FrozenPAdamWState,
    adam_first_moment_workload_matrix,
)
from dp_muon.privacy import (
    ShadowJMEPrivacyCalibration,
    jme_gamma_and_joint_sensitivity,
    make_clipped_gradient_query,
)

from .bandinvmf_strategy_manager import (
    ShadowJMEFirstBandInvMFFitRequest,
    ShadowJMESecondBandInvMFFitRequest,
    fit_shadow_jme_first_strategy,
    fit_shadow_jme_second_strategy,
)
from .nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer


PyTree = Any
_JME_STREAM_TAG = 0x534A4D45
_JME_SENSITIVITY_REL_TOL = 1e-6
_JME_SURROGATE_SENSITIVITY_BOUND_FACTOR = 1.25


def _block_lengths(post_horizon: int, segment_length: int) -> tuple[int, ...]:
  if (
      not isinstance(post_horizon, Integral)
      or isinstance(post_horizon, bool)
      or post_horizon < 1
      or not isinstance(segment_length, Integral)
      or isinstance(segment_length, bool)
      or segment_length < 1
  ):
    raise ValueError("post_horizon and segment_length must be positive integers")
  full, remainder = divmod(int(post_horizon), int(segment_length))
  return (int(segment_length),) * full + ((remainder,) if remainder else ())


@dataclass(frozen=True)
class ShadowJMEPlan:
  """Static plan plus the initial host-fitted strategy pair per segment.

  The strategy tuples are optional at construction time for callers that want
  to fit them only after warmup.  The production CIFAR path supplies a
  public surrogate pair for compilation and replaces each pair at its host
  boundary using the current DP ``P``.
  """

  condition: str
  warmup_steps: int
  segment_lengths: tuple[int, ...]
  warmup_strategy: BandInvMFStrategy
  first_strategies: tuple[BandInvMFStrategy, ...]
  second_strategies: tuple[BandInvMFStrategy, ...]
  calibration: ShadowJMEPrivacyCalibration
  runtime_bandwidth: int
  beta1: float = 0.9
  beta2: float = 0.999
  learning_rate: float = 1.0
  eps: float = 1e-8
  weight_decay: float = 0.0
  v_floor: float = 0.0
  max_optimizer_steps: int = 1000
  reduction: str = "mean"
  min_sep: int | None = None
  max_participations: int | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.warmup_steps, Integral) or self.warmup_steps < 1:
      raise ValueError("warmup_steps must be a positive integer")
    if not self.segment_lengths or any(
        not isinstance(length, Integral) or length < 1 for length in self.segment_lengths
    ):
      raise ValueError("segment_lengths must contain positive integers")
    if not isinstance(self.warmup_strategy, BandInvMFStrategy):
      raise TypeError("warmup_strategy must be a BandInvMFStrategy")
    if self.first_strategies and len(self.first_strategies) != len(self.segment_lengths):
      raise ValueError("first_strategies must be empty or match segment_lengths")
    if self.second_strategies and len(self.second_strategies) != len(self.segment_lengths):
      raise ValueError("second_strategies must be empty or match segment_lengths")
    if bool(self.first_strategies) != bool(self.second_strategies):
      raise ValueError("first_strategies and second_strategies must be supplied together")
    for length, first, second in zip(
        self.segment_lengths, self.first_strategies, self.second_strategies, strict=False
    ):
      if first.horizon != length or second.horizon != length:
        raise ValueError("JME strategy horizons must match segment_lengths")
    if self.runtime_bandwidth < 1:
      raise ValueError("runtime_bandwidth must be positive")
    if not isinstance(self.calibration, ShadowJMEPrivacyCalibration):
      raise TypeError("calibration must be a ShadowJMEPrivacyCalibration")
    if len(self.calibration.segment_sensitivity_squared) != len(self.segment_lengths):
      raise ValueError("calibration must match the number of segments")

  @property
  def post_horizon(self) -> int:
    return sum(self.segment_lengths)

  @property
  def horizon(self) -> int:
    return self.warmup_steps + self.post_horizon


def _empty_frozen_state(params: PyTree) -> FrozenPAdamWState:
  zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
  ones = jax.tree_util.tree_map(jnp.ones_like, params)
  return FrozenPAdamWState(
      count=jnp.array(0, dtype=jnp.int32), mu=zeros, frozen_nu=zeros, p_star=ones
  )


def _pad_coef(strategy: BandInvMFStrategy, bandwidth: int) -> jax.Array:
  coef = jnp.asarray(strategy.noising_coef)
  if coef.ndim != 1 or coef.shape[0] > bandwidth:
    raise ValueError("strategy noising coefficient is incompatible with runtime bandwidth")
  return jnp.pad(coef, (0, bandwidth - coef.shape[0]))


def _segment_start(plan: ShadowJMEPlan, segment_index: int) -> int:
  if not 0 <= segment_index < len(plan.segment_lengths):
    raise ValueError("segment_index is outside the JME plan")
  return plan.warmup_steps + sum(plan.segment_lengths[:segment_index])


def _segment_keys(root_key: jax.Array, segment_index: jax.Array) -> tuple[jax.Array, jax.Array]:
  tagged = jax.random.fold_in(root_key, _JME_STREAM_TAG)
  base = jnp.asarray(segment_index, dtype=jnp.int32) * jnp.array(2, dtype=jnp.int32)
  return jax.random.fold_in(tagged, base), jax.random.fold_in(tagged, base + 1)


def _strategy_schedule(
    plan: ShadowJMEPlan,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  bandwidth = plan.runtime_bandwidth
  if not plan.first_strategies:
    zeros = jnp.zeros((len(plan.segment_lengths), bandwidth))
    return zeros, zeros, jnp.ones((len(plan.segment_lengths),), dtype=zeros.dtype)
  first = jnp.stack([_pad_coef(strategy, bandwidth) for strategy in plan.first_strategies])
  second = jnp.stack([_pad_coef(strategy, bandwidth) for strategy in plan.second_strategies])
  gammas = jnp.asarray([
      jme_gamma_and_joint_sensitivity(
          first_strategy, second_strategy,
          clip_norm=plan.calibration.clip_norm,
          normalize_by=plan.calibration.normalize_by,
          adjacency=plan.calibration.adjacency,
      )[0]
      for first_strategy, second_strategy in zip(
          plan.first_strategies, plan.second_strategies, strict=True
      )
  ])
  return first, second, gammas


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ShadowJMEBandInvDPAdamWState:
  """Checkpointable state for both JME channels and the AdamW phases."""

  params: PyTree
  optimizer_state: Any
  frozen_state: FrozenPAdamWState
  v_shadow: PyTree
  v_shadow_count: jax.Array
  noise_state_m: BandInvMFNoiseState
  noise_state_v: BandInvMFNoiseState
  rng_root_key: jax.Array
  rng_key_m: jax.Array
  rng_key_v: jax.Array
  step: jax.Array
  phase: jax.Array  # 0 = dynamic warmup, 1 = frozen-p/JME
  segment_index: jax.Array
  segment_start: jax.Array
  first_noising_coef: jax.Array
  second_noising_coef: jax.Array
  gamma: jax.Array

  def tree_flatten(self):
    return (
        self.params, self.optimizer_state, self.frozen_state, self.v_shadow,
        self.v_shadow_count, self.noise_state_m, self.noise_state_v,
        self.rng_root_key, self.rng_key_m, self.rng_key_v, self.step, self.phase,
        self.segment_index, self.segment_start, self.first_noising_coef,
        self.second_noising_coef, self.gamma,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_nonamplified_shadow_jme_bandinv_dpadamw_state(
    params: PyTree,
    plan: ShadowJMEPlan,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> ShadowJMEBandInvDPAdamWState:
  """Initializes dynamic AdamW and empty independent JME FIR states."""
  if not isinstance(plan, ShadowJMEPlan):
    raise TypeError("plan must be a ShadowJMEPlan")
  if not isinstance(optimizer, optax.GradientTransformation):
    raise TypeError("optimizer must be an Optax GradientTransformation")
  first_schedule, second_schedule, gamma_schedule = _strategy_schedule(plan)
  if plan.first_strategies:
    first_coef, second_coef, gamma = (
        first_schedule[0], second_schedule[0], gamma_schedule[0]
    )
  else:
    first_coef = _pad_coef(plan.warmup_strategy, plan.runtime_bandwidth)
    second_coef = jnp.zeros_like(first_coef)
    gamma = jnp.array(1.0, dtype=first_coef.dtype)
  noise_m = init_bandinv_noise_state(params, plan.runtime_bandwidth)
  noise_v = init_bandinv_noise_state(params, plan.runtime_bandwidth)
  return ShadowJMEBandInvDPAdamWState(
      params=params,
      optimizer_state=optimizer.init(params),
      frozen_state=_empty_frozen_state(params),
      v_shadow=jax.tree_util.tree_map(jnp.zeros_like, params),
      v_shadow_count=jnp.array(0, dtype=jnp.int32),
      noise_state_m=noise_m,
      noise_state_v=noise_v,
      rng_root_key=rng_key,
      rng_key_m=rng_key,
      rng_key_v=jax.random.fold_in(rng_key, _JME_STREAM_TAG + 1),
      step=jnp.array(0, dtype=jnp.int32),
      phase=jnp.array(0, dtype=jnp.int32),
      segment_index=jnp.array(0, dtype=jnp.int32),
      segment_start=jnp.array(0, dtype=jnp.int32),
      first_noising_coef=first_coef,
      second_noising_coef=second_coef,
      gamma=gamma,
  )


def _process_shadow_v(
    v_shadow: PyTree, *, count: jax.Array, beta2: float, eps: float, v_floor: float
) -> tuple[PyTree, PyTree]:
  correction = 1.0 - jnp.asarray(beta2) ** count
  processed = jax.tree_util.tree_map(
      lambda value: jnp.maximum(value / correction, v_floor), v_shadow
  )
  p_star = jax.tree_util.tree_map(
      lambda value: 1.0 / (jnp.sqrt(value) + eps), processed
  )
  return processed, p_star


def _validated_state(state: ShadowJMEBandInvDPAdamWState, plan: ShadowJMEPlan) -> None:
  if not isinstance(state, ShadowJMEBandInvDPAdamWState):
    raise TypeError("state must be a ShadowJMEBandInvDPAdamWState")
  for noise_state in (state.noise_state_m, state.noise_state_v):
    if not isinstance(noise_state, BandInvMFNoiseState):
      raise TypeError("JME noise states must be BandInvMFNoiseState values")
    if noise_state.bandwidth != plan.runtime_bandwidth:
      raise ValueError("JME noise state has the wrong runtime bandwidth")
  if not isinstance(state.step, jax.core.Tracer):
    step = int(state.step)
    if not 0 <= step <= plan.horizon:
      raise ValueError("state.step is outside the JME horizon")


def _guard_shadow_jme_segment_sensitivity(
    plan: ShadowJMEPlan,
    segment_index: int,
    first_strategy: BandInvMFStrategy,
    second_strategy: BandInvMFStrategy,
) -> float:
  """Checks a host-fitted pair against its pre-calibrated bound.

  A dynamic refit may change the actual BandInvMF operator norms.  The
  calibrated Gaussian standard deviation is intentionally never reduced or
  silently adjusted here: an over-bound pair is rejected before it can be
  installed in the mechanism.
  """
  gamma, sensitivity = jme_gamma_and_joint_sensitivity(
      first_strategy,
      second_strategy,
      clip_norm=plan.calibration.clip_norm,
      normalize_by=plan.calibration.normalize_by,
      adjacency=plan.calibration.adjacency,
  )
  actual_squared = float(sensitivity * sensitivity)
  calibrated_squared = float(
      plan.calibration.calibrated_segment_sensitivity_squared[segment_index]
  )
  if not math.isfinite(gamma) or gamma <= 0 or not math.isfinite(actual_squared):
    raise RuntimeError(
        f"shadow-JME segment {segment_index} has a non-finite sensitivity or gamma"
    )
  if not math.isfinite(calibrated_squared) or calibrated_squared <= 0:
    raise RuntimeError(
        f"shadow-JME segment {segment_index} has no finite positive calibrated bound"
    )
  if actual_squared > calibrated_squared * (1.0 + _JME_SENSITIVITY_REL_TOL):
    raise RuntimeError(
        "shadow-JME refit sensitivity exceeds its calibrated conservative "
        f"bound at segment {segment_index}: actual_squared={actual_squared:.9g}, "
        f"calibrated_squared={calibrated_squared:.9g}, "
        f"relative_tolerance={_JME_SENSITIVITY_REL_TOL:.3g}"
    )
  return actual_squared


def make_nonamplified_shadow_jme_bandinv_dpadamw_train_step(
    loss_fn: Callable[..., Any],
    plan: ShadowJMEPlan,
    *,
    learning_rate: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    eps: float | None = None,
    weight_decay: float | None = None,
    microbatch_size: int | None = None,
) -> tuple[
    Callable[[ShadowJMEBandInvDPAdamWState, Any], ShadowJMEBandInvDPAdamWState],
    optax.GradientTransformation,
]:
  """Builds the dynamic-warmup plus frozen-p/shadow-JME train step."""
  if not callable(loss_fn):
    raise TypeError("loss_fn must be callable")
  if not isinstance(plan, ShadowJMEPlan):
    raise TypeError("plan must be a ShadowJMEPlan")
  learning_rate = plan.learning_rate if learning_rate is None else learning_rate
  beta1 = plan.beta1 if beta1 is None else beta1
  beta2 = plan.beta2 if beta2 is None else beta2
  eps = plan.eps if eps is None else eps
  weight_decay = plan.weight_decay if weight_decay is None else weight_decay
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=learning_rate, beta1=beta1, beta2=beta2,
      eps=eps, weight_decay=weight_decay,
  )
  frozen_optimizer = FrozenPAdamW(
      learning_rate=learning_rate, beta1=beta1, weight_decay=weight_decay
  )
  clipped_query = make_clipped_gradient_query(
      loss_fn,
      clip_norm=plan.calibration.clip_norm,
      normalize_by=plan.calibration.normalize_by,
      batch_argnums=1,
      keep_batch_dim=True,
      microbatch_size=microbatch_size,
  )
  first_schedule, second_schedule, gamma_schedule = _strategy_schedule(plan)
  segment_lengths = jnp.asarray(plan.segment_lengths, dtype=jnp.int32)

  def reset_noise(state: ShadowJMEBandInvDPAdamWState, segment_index: jax.Array):
    key_m, key_v = _segment_keys(state.rng_root_key, segment_index)
    return (
        init_bandinv_noise_state(state.params, plan.runtime_bandwidth),
        init_bandinv_noise_state(state.params, plan.runtime_bandwidth),
        key_m,
        key_v,
    )

  def warmup_step(
      state: ShadowJMEBandInvDPAdamWState, clipped_grad: PyTree
  ) -> ShadowJMEBandInvDPAdamWState:
    warmup_coef = _pad_coef(plan.warmup_strategy, plan.runtime_bandwidth)
    runtime_coef = warmup_coef + (
        jnp.asarray(state.step, dtype=warmup_coef.dtype) * jnp.zeros_like(warmup_coef)
    )
    noise, new_noise, new_key = sample_bandinv_noise(
        state.rng_key_m, state.noise_state_m, runtime_coef,
        jnp.asarray(plan.calibration.iid_noise_std),
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation, clipped_grad, noise
    )
    updates, new_optimizer_state = optimizer.update(
        private_grad, state.optimizer_state, state.params
    )
    new_step = state.step + jnp.array(1, dtype=state.step.dtype)
    params = optax.apply_updates(state.params, updates)
    is_switch = new_step == plan.warmup_steps

    def switch(_: None) -> ShadowJMEBandInvDPAdamWState:
      del _
      from dp_muon.optim import freeze_optax_adamw

      frozen = freeze_optax_adamw(new_optimizer_state, beta2=beta2, eps=eps)
      reset_m, reset_v, key_m, key_v = reset_noise(state, jnp.array(0, dtype=jnp.int32))
      if plan.first_strategies:
        first_coef, second_coef, gamma = (
            first_schedule[0], second_schedule[0], gamma_schedule[0]
        )
      else:
        first_coef, second_coef, gamma = runtime_coef, jnp.zeros_like(runtime_coef), jnp.array(1.0)
      return replace(
          state,
          params=params,
          optimizer_state=new_optimizer_state,
          frozen_state=frozen,
          v_shadow=jax.tree_util.tree_map(lambda value: value, _adam_nu(new_optimizer_state)),
          v_shadow_count=new_step,
          noise_state_m=reset_m,
          noise_state_v=reset_v,
          rng_key_m=key_m,
          rng_key_v=key_v,
          step=new_step,
          phase=jnp.array(1, dtype=state.phase.dtype),
          segment_index=jnp.array(0, dtype=state.segment_index.dtype),
          segment_start=new_step,
          first_noising_coef=first_coef,
          second_noising_coef=second_coef,
          gamma=gamma,
      )

    def continue_warmup(_: None) -> ShadowJMEBandInvDPAdamWState:
      del _
      return replace(
          state,
          params=params,
          optimizer_state=new_optimizer_state,
          noise_state_m=new_noise,
          rng_key_m=new_key,
          step=new_step,
      )

    return jax.lax.cond(is_switch, switch, continue_warmup, operand=None)

  def post_step(
      state: ShadowJMEBandInvDPAdamWState, clipped_grad: PyTree
  ) -> ShadowJMEBandInvDPAdamWState:
    coef_m = state.first_noising_coef + (
        jnp.asarray(state.step, dtype=state.first_noising_coef.dtype)
        * jnp.zeros_like(state.first_noising_coef)
    )
    coef_v = state.second_noising_coef + (
        jnp.asarray(state.step, dtype=state.second_noising_coef.dtype)
        * jnp.zeros_like(state.second_noising_coef)
    )
    noise_m, new_noise_m, new_key_m = sample_bandinv_noise(
        state.rng_key_m, state.noise_state_m, coef_m,
        jnp.asarray(plan.calibration.iid_noise_std),
    )
    noise_v, new_noise_v, new_key_v = sample_bandinv_noise(
        state.rng_key_v, state.noise_state_v, coef_v,
        jnp.asarray(plan.calibration.iid_noise_std),
    )
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation, clipped_grad, noise_m
    )
    q_private = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient * gradient
        + state.gamma.astype(jnp.asarray(gradient).dtype) ** (-0.5) * perturbation,
        clipped_grad,
        noise_v,
    )
    # Crucial JME invariant: the second channel is built from x*x, never from
    # the already-noised first channel.
    new_shadow = jax.tree_util.tree_map(
        lambda old, query: beta2 * old + (1.0 - beta2) * query,
        state.v_shadow,
        q_private,
    )
    updates, new_frozen = frozen_optimizer.update(
        private_grad, state.frozen_state, state.params
    )
    params = optax.apply_updates(state.params, updates)
    new_step = state.step + jnp.array(1, dtype=state.step.dtype)
    new_shadow_count = state.v_shadow_count + jnp.array(1, dtype=state.v_shadow_count.dtype)
    local_next = new_step - state.segment_start
    segment_end = local_next >= segment_lengths[state.segment_index]

    def finish_segment(_: None) -> ShadowJMEBandInvDPAdamWState:
      del _
      processed, p_star = _process_shadow_v(
          new_shadow, count=new_shadow_count, beta2=beta2, eps=eps,
          v_floor=plan.v_floor,
      )
      next_index = state.segment_index + jnp.array(1, dtype=state.segment_index.dtype)
      reset_m, reset_v, key_m, key_v = reset_noise(state, next_index)
      has_next = next_index < len(plan.segment_lengths)

      def load_static(_: None):
        return first_schedule[next_index], second_schedule[next_index], gamma_schedule[next_index]

      def retain(_: None):
        return state.first_noising_coef, state.second_noising_coef, state.gamma

      first_coef, second_coef, gamma = jax.lax.cond(has_next, load_static, retain, operand=None)
      return replace(
          state,
          params=params,
          frozen_state=replace(new_frozen, frozen_nu=processed, p_star=p_star),
          v_shadow=new_shadow,
          v_shadow_count=new_shadow_count,
          noise_state_m=reset_m,
          noise_state_v=reset_v,
          rng_key_m=key_m,
          rng_key_v=key_v,
          step=new_step,
          segment_index=next_index,
          segment_start=new_step,
          first_noising_coef=first_coef,
          second_noising_coef=second_coef,
          gamma=gamma,
      )

    def continue_segment(_: None) -> ShadowJMEBandInvDPAdamWState:
      del _
      return replace(
          state,
          params=params,
          frozen_state=new_frozen,
          v_shadow=new_shadow,
          v_shadow_count=new_shadow_count,
          noise_state_m=new_noise_m,
          noise_state_v=new_noise_v,
          rng_key_m=new_key_m,
          rng_key_v=new_key_v,
          step=new_step,
      )

    return jax.lax.cond(segment_end, finish_segment, continue_segment, operand=None)

  def train_step(
      state: ShadowJMEBandInvDPAdamWState, batch: Any
  ) -> ShadowJMEBandInvDPAdamWState:
    _validated_state(state, plan)
    clipped_grad = clipped_query(state.params, batch)
    return jax.lax.cond(
        state.phase == 0,
        lambda value: warmup_step(value[0], value[1]),
        lambda value: post_step(value[0], value[1]),
        (state, clipped_grad),
    )

  return train_step, optimizer


def _adam_nu(optimizer_state: Any) -> PyTree:
  """Extracts Optax's second moment without depending on tuple positions."""
  if all(hasattr(optimizer_state, name) for name in ("count", "mu", "nu")):
    return optimizer_state.nu
  if isinstance(optimizer_state, (tuple, list)):
    for value in optimizer_state:
      try:
        return _adam_nu(value)
      except TypeError:
        pass
  raise TypeError("optimizer_state does not contain an Adam second moment")


def begin_shadow_jme_segment(
    state: ShadowJMEBandInvDPAdamWState,
    plan: ShadowJMEPlan,
    *,
    segment_index: int,
    first_strategy: BandInvMFStrategy,
    second_strategy: BandInvMFStrategy,
) -> ShadowJMEBandInvDPAdamWState:
  """Installs a freshly fitted strategy pair at a host-side boundary.

  The function is deliberately eager/orchestration-facing.  It never runs
  inside a jitted train step, and it only consumes the DP ``v_shadow`` output.
  """
  _validated_state(state, plan)
  if not isinstance(segment_index, Integral) or isinstance(segment_index, bool):
    raise ValueError("segment_index must be an integer")
  index = int(segment_index)
  if not 0 <= index < len(plan.segment_lengths):
    raise ValueError("segment_index is outside the JME plan")
  length = plan.segment_lengths[index]
  if first_strategy.horizon != length or second_strategy.horizon != length:
    raise ValueError("strategy horizons must match the target segment")
  _guard_shadow_jme_segment_sensitivity(
      plan, index, first_strategy, second_strategy
  )
  expected_start = _segment_start(plan, index)
  if not isinstance(state.step, jax.core.Tracer) and int(state.step) != expected_start:
    raise ValueError("segment can only begin at its planned global boundary")
  frozen = state.frozen_state
  if index > 0:
    processed, p_star = _process_shadow_v(
        state.v_shadow,
        count=state.v_shadow_count,
        beta2=plan.beta2,
        eps=plan.eps,
        v_floor=plan.v_floor,
    )
    frozen = replace(frozen, frozen_nu=processed, p_star=p_star)
  gamma, _ = jme_gamma_and_joint_sensitivity(
      first_strategy,
      second_strategy,
      clip_norm=plan.calibration.clip_norm,
      normalize_by=plan.calibration.normalize_by,
      adjacency=plan.calibration.adjacency,
  )
  noise_m = init_bandinv_noise_state(state.params, plan.runtime_bandwidth)
  noise_v = init_bandinv_noise_state(state.params, plan.runtime_bandwidth)
  key_m, key_v = _segment_keys(
      state.rng_root_key, jnp.asarray(index, dtype=jnp.int32)
  )
  return replace(
      state,
      frozen_state=frozen,
      noise_state_m=noise_m,
      noise_state_v=noise_v,
      rng_key_m=key_m,
      rng_key_v=key_v,
      phase=jnp.array(1, dtype=state.phase.dtype),
      segment_index=jnp.array(index, dtype=state.segment_index.dtype),
      segment_start=jnp.array(expected_start, dtype=state.segment_start.dtype),
      first_noising_coef=_pad_coef(first_strategy, plan.runtime_bandwidth),
      second_noising_coef=_pad_coef(second_strategy, plan.runtime_bandwidth),
      gamma=jnp.asarray(gamma),
  )


def _p_scale_from_state(state: ShadowJMEBandInvDPAdamWState) -> float:
  values = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state.frozen_state.p_star)]
  if not values:
    raise ValueError("frozen p must contain at least one leaf")
  scale = max(float(np.max(np.abs(value))) for value in values)
  if not math_is_finite_positive(scale):
    raise ValueError("frozen p scale must be finite and positive")
  return scale


def math_is_finite_positive(value: float) -> bool:
  return bool(np.isfinite(value) and value > 0)


def fit_shadow_jme_segment_strategies(
    state: ShadowJMEBandInvDPAdamWState,
    plan: ShadowJMEPlan,
    *,
    segment_index: int,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> tuple[BandInvMFStrategy, BandInvMFStrategy]:
  """Fits ``C_m,k`` and ``C_v,k`` on the host from the current DP ``P_k``."""
  if not isinstance(segment_index, Integral) or isinstance(segment_index, bool):
    raise ValueError("segment_index must be an integer")
  index = int(segment_index)
  if not 0 <= index < len(plan.segment_lengths):
    raise ValueError("segment_index is outside the JME plan")
  start = _segment_start(plan, index)
  p_scale = _p_scale_from_state(state)
  first = fit_shadow_jme_first_strategy(
      ShadowJMEFirstBandInvMFFitRequest(
          segment_length=plan.segment_lengths[index],
          segment_start_step=start,
          min_sep=plan.min_sep or plan.segment_lengths[index],
          max_participations=plan.max_participations or 1,
          bandwidth=plan.runtime_bandwidth,
          beta1=plan.beta1,
          learning_rate=plan.learning_rate,
          weight_decay=plan.weight_decay,
          frozen_preconditioner=p_scale,
          reduction=plan.reduction,  # type: ignore[arg-type]
          max_optimizer_steps=plan.max_optimizer_steps,
      ),
      fit_strategy=fit_strategy,
  )
  second = fit_shadow_jme_second_strategy(
      ShadowJMESecondBandInvMFFitRequest(
          segment_length=plan.segment_lengths[index],
          min_sep=plan.min_sep or plan.segment_lengths[index],
          max_participations=plan.max_participations or 1,
          bandwidth=plan.runtime_bandwidth,
          beta2=plan.beta2,
          reduction=plan.reduction,  # type: ignore[arg-type]
          max_optimizer_steps=plan.max_optimizer_steps,
      ),
      fit_strategy=fit_strategy,
  )
  _guard_shadow_jme_segment_sensitivity(plan, index, first, second)
  return first, second


def fit_shadow_jme_plan(
    *,
    horizon: int,
    warmup_steps: int,
    segment_length: int,
    min_sep: int,
    max_participations: int,
    bandwidth: int,
    reduction: str,
    max_optimizer_steps: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    epsilon: float,
    delta: float,
    clip_norm: float,
    normalize_by: float,
    adjacency: str,
    v_floor: float = 0.0,
    fit_strategy: Callable[..., BandInvMFStrategy] = fit_bandinv_strategy,
) -> ShadowJMEPlan:
  """Fits a complete initial plan using a unit frozen-``P`` surrogate."""
  if warmup_steps > min_sep or segment_length > min_sep:
    raise ValueError("warmup_steps and segment_length must be <= min_sep")
  if horizon <= warmup_steps:
    raise ValueError("horizon must exceed warmup_steps")
  lengths = _block_lengths(horizon - warmup_steps, segment_length)
  warmup_workload = jnp.asarray(np.abs(np.asarray(
      adam_first_moment_workload_matrix(
          warmup_steps, beta1, learning_rate, weight_decay
      )
  )))
  warmup = fit_strategy(
      warmup_steps, min(bandwidth, warmup_steps), min(min_sep, warmup_steps),
      max_participations=max_participations,
      workload_matrix=warmup_workload,
      max_optimizer_steps=max_optimizer_steps,
      reduction=reduction,
  )
  if not isinstance(warmup, BandInvMFStrategy):
    raise TypeError("fit_strategy must return a BandInvMFStrategy")
  first_strategies = []
  second_strategies = []
  for index, length in enumerate(lengths):
    first, second = fit_shadow_jme_first_strategy(
        ShadowJMEFirstBandInvMFFitRequest(
            segment_length=length,
            segment_start_step=warmup_steps + sum(lengths[:index]),
            min_sep=min_sep,
            max_participations=max_participations,
            bandwidth=min(bandwidth, length),
            beta1=beta1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            frozen_preconditioner=1.0,
            reduction=reduction,  # type: ignore[arg-type]
            max_optimizer_steps=max_optimizer_steps,
        ),
        fit_strategy=fit_strategy,
    ), fit_shadow_jme_second_strategy(
        ShadowJMESecondBandInvMFFitRequest(
            segment_length=length,
            min_sep=min_sep,
            max_participations=max_participations,
            bandwidth=min(bandwidth, length),
            beta2=beta2,
            reduction=reduction,  # type: ignore[arg-type]
            max_optimizer_steps=max_optimizer_steps,
        ),
        fit_strategy=fit_strategy,
    )
    first_strategies.append(first)
    second_strategies.append(second)
  from dp_muon.privacy import calibrate_shadow_jme

  # Unit-P fitting is a surrogate for the later DP-P refits.  Reserve an
  # explicit per-segment sensitivity envelope and still check every accepted
  # refit against it at the host boundary.  This keeps calibration tied to a
  # finite upper bound instead of assuming numerical scale invariance alone.
  surrogate_calibration = calibrate_shadow_jme(
      epsilon=epsilon,
      delta=delta,
      clip_norm=clip_norm,
      normalize_by=normalize_by,
      adjacency=adjacency,  # type: ignore[arg-type]
      warmup_strategy=warmup,
      first_strategies=first_strategies,
      second_strategies=second_strategies,
  )
  safe_segment_bounds = tuple(
      _JME_SURROGATE_SENSITIVITY_BOUND_FACTOR * value
      for value in surrogate_calibration.segment_sensitivity_squared
  )
  calibration = calibrate_shadow_jme(
      epsilon=epsilon,
      delta=delta,
      clip_norm=clip_norm,
      normalize_by=normalize_by,
      adjacency=adjacency,  # type: ignore[arg-type]
      warmup_strategy=warmup,
      first_strategies=first_strategies,
      second_strategies=second_strategies,
      segment_sensitivity_upper_bounds=safe_segment_bounds,
  )
  runtime_bandwidth = max(
      bandwidth,
      warmup.bandwidth,
      *(strategy.bandwidth for strategy in first_strategies),
      *(strategy.bandwidth for strategy in second_strategies),
  )
  return ShadowJMEPlan(
      condition=f"shadow-jme-t{warmup_steps}-seg{segment_length}",
      warmup_steps=warmup_steps,
      segment_lengths=tuple(lengths),
      warmup_strategy=warmup,
      first_strategies=tuple(first_strategies),
      second_strategies=tuple(second_strategies),
      calibration=calibration,
      runtime_bandwidth=runtime_bandwidth,
      beta1=beta1,
      beta2=beta2,
      learning_rate=learning_rate,
      eps=eps,
      weight_decay=weight_decay,
      v_floor=v_floor,
      max_optimizer_steps=max_optimizer_steps,
      reduction=reduction,
      min_sep=min_sep,
      max_participations=max_participations,
  )


# Naming aliases mirror the existing non-amplified segmented/frozen modules.
NonAmplifiedShadowJMEBandInvDPAdamWState = ShadowJMEBandInvDPAdamWState
init_shadow_jme_bandinv_dpadamw_state = init_nonamplified_shadow_jme_bandinv_dpadamw_state
make_shadow_jme_bandinv_dpadamw_train_step = make_nonamplified_shadow_jme_bandinv_dpadamw_train_step


__all__ = [
    "ShadowJMEPlan",
    "ShadowJMEBandInvDPAdamWState",
    "NonAmplifiedShadowJMEBandInvDPAdamWState",
    "begin_shadow_jme_segment",
    "fit_shadow_jme_plan",
    "fit_shadow_jme_segment_strategies",
    "init_nonamplified_shadow_jme_bandinv_dpadamw_state",
    "make_nonamplified_shadow_jme_bandinv_dpadamw_train_step",
    "init_shadow_jme_bandinv_dpadamw_state",
    "make_shadow_jme_bandinv_dpadamw_train_step",
]
