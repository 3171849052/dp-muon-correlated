"""Online four-path diagnostics layered on Experiment 3's shadow state.

The real update is still built by :func:`exp3.online_shadow.make_online_shadow_train_step`.
This module only consumes the moments that that step already maintains.  Its
JAX accumulator keeps vector-valued window sums on the device and exposes a
row only when a window closes, avoiding a second model pass and avoiding a
per-step device-to-host copy of the parameter tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from exp3.online_shadow import OnlineShadowState

from .diagnostics import window_row


PyTree = Any


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(lambda value: jnp.zeros_like(value), tree)


def _sqnorm(tree: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(value, dtype=jnp.float32) ** 2)
      for value in jax.tree_util.tree_leaves(tree)
  )


def _debiased(tree: PyTree, beta: float, step: jax.Array) -> PyTree:
  correction = 1.0 - beta ** step
  return jax.tree_util.tree_map(lambda value: value / correction, tree)


def _clean_p(clean_v: PyTree, beta2: float, step: jax.Array, eps: float) -> PyTree:
  v_hat = _debiased(clean_v, beta2, step)
  return jax.tree_util.tree_map(
      lambda value: 1.0 / (jnp.sqrt(value) + eps), v_hat
  )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WindowAccumulatorState:
  """Device-resident state for one currently open non-overlapping window."""

  previous_p: PyTree
  frozen_p: PyTree
  weighted_momentum: PyTree
  weighted_frozen_p: PyTree
  weighted_dynamic_clean_p: PyTree
  weighted_real_adamw: PyTree
  denominator_momentum: jax.Array
  denominator_frozen_p: jax.Array
  denominator_dynamic_clean_p: jax.Array
  denominator_real_adamw: jax.Array
  p_change_sum: jax.Array
  count: jax.Array
  window_index: jax.Array
  has_previous_p: jax.Array
  emitted: jax.Array
  emitted_start_step: jax.Array
  emitted_end_step: jax.Array
  emitted_mean_p_relative_change: jax.Array
  emitted_C_momentum: jax.Array
  emitted_C_frozen_p: jax.Array
  emitted_C_dynamic_clean_p: jax.Array
  emitted_C_real_adamw: jax.Array

  def tree_flatten(self):
    children = (
        self.previous_p,
        self.frozen_p,
        self.weighted_momentum,
        self.weighted_frozen_p,
        self.weighted_dynamic_clean_p,
        self.weighted_real_adamw,
        self.denominator_momentum,
        self.denominator_frozen_p,
        self.denominator_dynamic_clean_p,
        self.denominator_real_adamw,
        self.p_change_sum,
        self.count,
        self.window_index,
        self.has_previous_p,
        self.emitted,
        self.emitted_start_step,
        self.emitted_end_step,
        self.emitted_mean_p_relative_change,
        self.emitted_C_momentum,
        self.emitted_C_frozen_p,
        self.emitted_C_dynamic_clean_p,
        self.emitted_C_real_adamw,
    )
    return children, None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_window_accumulator(params: PyTree) -> WindowAccumulatorState:
  """Initialize only diagnostic state; no training state is touched."""
  zeros = _zeros_like(params)
  scalar = lambda value=0.0: jnp.asarray(value, dtype=jnp.float32)
  return WindowAccumulatorState(
      previous_p=zeros,
      frozen_p=zeros,
      weighted_momentum=zeros,
      weighted_frozen_p=zeros,
      weighted_dynamic_clean_p=zeros,
      weighted_real_adamw=zeros,
      denominator_momentum=scalar(),
      denominator_frozen_p=scalar(),
      denominator_dynamic_clean_p=scalar(),
      denominator_real_adamw=scalar(),
      p_change_sum=scalar(),
      count=jnp.asarray(0, dtype=jnp.int32),
      window_index=jnp.asarray(0, dtype=jnp.int32),
      has_previous_p=jnp.asarray(False),
      emitted=jnp.asarray(False),
      emitted_start_step=jnp.asarray(0, dtype=jnp.int32),
      emitted_end_step=jnp.asarray(0, dtype=jnp.int32),
      emitted_mean_p_relative_change=scalar(),
      emitted_C_momentum=scalar(),
      emitted_C_frozen_p=scalar(),
      emitted_C_dynamic_clean_p=scalar(),
      emitted_C_real_adamw=scalar(),
  )


def _update_window_accumulator(
    accumulator: WindowAccumulatorState,
    clean_m: PyTree,
    clean_v: PyTree,
    dp_m: PyTree,
    dp_v: PyTree,
    noise_m: PyTree,
    step: jax.Array,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    weight_decay: float,
    eps: float,
    window_size: int,
    eps_num: float,
) -> WindowAccumulatorState:
  """Add one already-computed Experiment 3 shadow step to the local window."""
  p = _clean_p(clean_v, beta2, step, eps)
  clean_m_hat = _debiased(clean_m, beta1, step)
  dp_m_hat = _debiased(dp_m, beta1, step)
  dp_v_hat = _debiased(dp_v, beta2, step)
  q_clean = jax.tree_util.tree_map(lambda m, p_value: m * p_value, clean_m_hat, p)
  q_dp = jax.tree_util.tree_map(
      lambda m, v: m / (jnp.sqrt(v) + eps), dp_m_hat, dp_v_hat
  )
  delta_q = jax.tree_util.tree_map(lambda dp, clean: dp - clean, q_dp, q_clean)
  r_linear = _debiased(noise_m, beta1, step)

  at_window_start = accumulator.count == 0
  frozen_p = jax.tree_util.tree_map(
      lambda old, current: jnp.where(at_window_start, current, old),
      accumulator.frozen_p,
      p,
  )
  x_momentum = jax.tree_util.tree_map(lambda value: -learning_rate * value, r_linear)
  x_frozen = jax.tree_util.tree_map(
      lambda p_value, value: -learning_rate * p_value * value, frozen_p, r_linear
  )
  x_dynamic = jax.tree_util.tree_map(
      lambda p_value, value: -learning_rate * p_value * value, p, r_linear
  )
  x_real = jax.tree_util.tree_map(lambda value: -learning_rate * value, delta_q)

  decay = jnp.asarray(1.0 - learning_rate * weight_decay, dtype=jnp.float32)
  decay_squared = decay * decay

  def update_weighted(old, value):
    return decay * old + value

  weighted_momentum = jax.tree_util.tree_map(
      update_weighted, accumulator.weighted_momentum, x_momentum
  )
  weighted_frozen = jax.tree_util.tree_map(
      update_weighted, accumulator.weighted_frozen_p, x_frozen
  )
  weighted_dynamic = jax.tree_util.tree_map(
      update_weighted, accumulator.weighted_dynamic_clean_p, x_dynamic
  )
  weighted_real = jax.tree_util.tree_map(
      update_weighted, accumulator.weighted_real_adamw, x_real
  )

  denominator_momentum = decay_squared * accumulator.denominator_momentum + _sqnorm(x_momentum)
  denominator_frozen = decay_squared * accumulator.denominator_frozen_p + _sqnorm(x_frozen)
  denominator_dynamic = decay_squared * accumulator.denominator_dynamic_clean_p + _sqnorm(x_dynamic)
  denominator_real = decay_squared * accumulator.denominator_real_adamw + _sqnorm(x_real)

  previous_norm = jnp.sqrt(_sqnorm(accumulator.previous_p))
  p_change = jnp.where(
      accumulator.has_previous_p,
      jnp.sqrt(_sqnorm(jax.tree_util.tree_map(
          lambda current, previous: current - previous, p, accumulator.previous_p
      ))) / (previous_norm + eps_num),
      jnp.asarray(0.0, dtype=jnp.float32),
  )
  count = accumulator.count + jnp.asarray(1, dtype=accumulator.count.dtype)
  complete = count == window_size
  p_change_total = accumulator.p_change_sum + p_change

  def score(weighted, denominator):
    return _sqnorm(weighted) / (denominator + eps_num)

  c_momentum = score(weighted_momentum, denominator_momentum)
  c_frozen = score(weighted_frozen, denominator_frozen)
  c_dynamic = score(weighted_dynamic, denominator_dynamic)
  c_real = score(weighted_real, denominator_real)
  mean_p_change = p_change_total / count

  return WindowAccumulatorState(
      previous_p=p,
      frozen_p=jax.tree_util.tree_map(
          lambda value: jnp.where(complete, 0.0, value), frozen_p
      ),
      weighted_momentum=jax.tree_util.tree_map(
          lambda value: jnp.where(complete, 0.0, value), weighted_momentum
      ),
      weighted_frozen_p=jax.tree_util.tree_map(
          lambda value: jnp.where(complete, 0.0, value), weighted_frozen
      ),
      weighted_dynamic_clean_p=jax.tree_util.tree_map(
          lambda value: jnp.where(complete, 0.0, value), weighted_dynamic
      ),
      weighted_real_adamw=jax.tree_util.tree_map(
          lambda value: jnp.where(complete, 0.0, value), weighted_real
      ),
      denominator_momentum=jnp.where(complete, 0.0, denominator_momentum),
      denominator_frozen_p=jnp.where(complete, 0.0, denominator_frozen),
      denominator_dynamic_clean_p=jnp.where(complete, 0.0, denominator_dynamic),
      denominator_real_adamw=jnp.where(complete, 0.0, denominator_real),
      p_change_sum=jnp.where(complete, 0.0, p_change_total),
      count=jnp.where(complete, 0, count),
      window_index=accumulator.window_index + complete.astype(jnp.int32),
      has_previous_p=jnp.asarray(True),
      emitted=complete,
      emitted_start_step=accumulator.window_index * window_size + 1,
      emitted_end_step=step,
      emitted_mean_p_relative_change=jnp.where(complete, mean_p_change, 0.0),
      emitted_C_momentum=jnp.where(complete, c_momentum, 0.0),
      emitted_C_frozen_p=jnp.where(complete, c_frozen, 0.0),
      emitted_C_dynamic_clean_p=jnp.where(complete, c_dynamic, 0.0),
      emitted_C_real_adamw=jnp.where(complete, c_real, 0.0),
  )


class WindowDiagnosticsCollector:
  """Host-side row collector around a JIT-compiled device accumulator."""

  def __init__(
      self,
      params: PyTree,
      *,
      seed: int,
      beta1: float,
      beta2: float,
      learning_rate: float,
      weight_decay: float,
      eps: float,
      window_size: int = 16,
      eps_num: float = 1e-30,
  ) -> None:
    if window_size < 1:
      raise ValueError("window_size must be positive")
    if eps_num <= 0:
      raise ValueError("eps_num must be positive")
    self.seed = int(seed)
    self.window_size = int(window_size)
    self.eps_num = float(eps_num)
    self._state = init_window_accumulator(params)
    self._rows: list[dict[str, float | int]] = []
    self._update = jax.jit(
        lambda accumulator, clean_m, clean_v, dp_m, dp_v, noise_m, step:
        _update_window_accumulator(
            accumulator, clean_m, clean_v, dp_m, dp_v, noise_m, step,
            beta1=beta1, beta2=beta2, learning_rate=learning_rate,
            weight_decay=weight_decay, eps=eps, window_size=self.window_size,
            eps_num=self.eps_num,
        )
    )

  @property
  def state(self) -> WindowAccumulatorState:
    return self._state

  @property
  def rows(self) -> list[dict[str, float | int]]:
    return list(self._rows)

  def _append_emitted_row(self) -> None:
    state = self._state
    if not bool(np.asarray(state.emitted)):
      return
    start_step = int(np.asarray(state.emitted_start_step))
    self._rows.append(window_row(
        seed=self.seed,
        window_index=(start_step - 1) // self.window_size,
        start_step=start_step,
        end_step=int(np.asarray(state.emitted_end_step)),
        mean_p_relative_change=float(np.asarray(state.emitted_mean_p_relative_change)),
        C_momentum=float(np.asarray(state.emitted_C_momentum)),
        C_frozen_p=float(np.asarray(state.emitted_C_frozen_p)),
        C_dynamic_clean_p=float(np.asarray(state.emitted_C_dynamic_clean_p)),
        C_real_adamw=float(np.asarray(state.emitted_C_real_adamw)),
    ))

  def after_step(self, state: OnlineShadowState, step: int) -> None:
    """Callback compatible with ``run_training(after_step=...)``."""
    self._state = self._update(
        self._state, state.clean_m, state.clean_v, state.dp_m, state.dp_v,
        state.noise_m, jnp.asarray(step, dtype=jnp.int32),
    )
    self._append_emitted_row()

  def finalize(self) -> list[dict[str, float | int]]:
    """Flush the final short window, if any, and return all rows."""
    state = self._state
    if int(np.asarray(state.count)):
      count = int(np.asarray(state.count))
      start = int(np.asarray(state.window_index)) * self.window_size + 1
      end = start + count - 1

      def score(weighted, denominator):
        return float(np.asarray(_sqnorm(weighted) / (denominator + self.eps_num)))

      self._rows.append(window_row(
          seed=self.seed,
          window_index=int(np.asarray(state.window_index)),
          start_step=start,
          end_step=end,
          mean_p_relative_change=float(np.asarray(state.p_change_sum)) / count,
          C_momentum=score(state.weighted_momentum, state.denominator_momentum),
          C_frozen_p=score(state.weighted_frozen_p, state.denominator_frozen_p),
          C_dynamic_clean_p=score(
              state.weighted_dynamic_clean_p, state.denominator_dynamic_clean_p
          ),
          C_real_adamw=score(state.weighted_real_adamw, state.denominator_real_adamw),
      ))
      # Make a second finalize call idempotent and leave the local diagnostic
      # state clear.  This does not touch the Experiment 3 training state.
      self._state = WindowAccumulatorState(
          previous_p=state.previous_p,
          frozen_p=state.frozen_p,
          weighted_momentum=_zeros_like(state.weighted_momentum),
          weighted_frozen_p=_zeros_like(state.weighted_frozen_p),
          weighted_dynamic_clean_p=_zeros_like(state.weighted_dynamic_clean_p),
          weighted_real_adamw=_zeros_like(state.weighted_real_adamw),
          denominator_momentum=jnp.asarray(0.0, jnp.float32),
          denominator_frozen_p=jnp.asarray(0.0, jnp.float32),
          denominator_dynamic_clean_p=jnp.asarray(0.0, jnp.float32),
          denominator_real_adamw=jnp.asarray(0.0, jnp.float32),
          p_change_sum=jnp.asarray(0.0, jnp.float32),
          count=jnp.asarray(0, jnp.int32),
          window_index=state.window_index + 1,
          has_previous_p=state.has_previous_p,
          emitted=jnp.asarray(False),
          emitted_start_step=jnp.asarray(0, jnp.int32),
          emitted_end_step=jnp.asarray(0, jnp.int32),
          emitted_mean_p_relative_change=jnp.asarray(0.0, jnp.float32),
          emitted_C_momentum=jnp.asarray(0.0, jnp.float32),
          emitted_C_frozen_p=jnp.asarray(0.0, jnp.float32),
          emitted_C_dynamic_clean_p=jnp.asarray(0.0, jnp.float32),
          emitted_C_real_adamw=jnp.asarray(0.0, jnp.float32),
      )
    return self.rows


__all__ = [
    "WindowAccumulatorState",
    "WindowDiagnosticsCollector",
    "init_window_accumulator",
]
