"""Device-resident 16-step shadow accumulator for Experiment 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from exp7.core import DEFAULT_V_FLOOR, Exp7TrainState
from exp7.diagnostics import PATHS, window_row


PyTree = Any


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _sqnorm(tree: PyTree) -> jax.Array:
  return sum(jnp.sum(jnp.asarray(x, jnp.float32) ** 2) for x in jax.tree_util.tree_leaves(tree))


def _nonpositive_fraction(tree: PyTree) -> jax.Array:
  leaves = jax.tree_util.tree_leaves(tree)
  count = sum(leaf.size for leaf in leaves)
  return sum(jnp.sum(jnp.asarray(leaf) <= 0) for leaf in leaves) / jnp.asarray(count, jnp.float32)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp7WindowState:
  previous_p: PyTree
  weighted: dict[str, PyTree]
  denominator: dict[str, jax.Array]
  negative_sum: dict[str, jax.Array]
  p_change_sum: jax.Array
  count: jax.Array
  window_index: jax.Array
  has_previous_p: jax.Array
  emitted: jax.Array
  emitted_start_step: jax.Array
  emitted_end_step: jax.Array
  emitted_p_change: jax.Array
  emitted_scores: dict[str, jax.Array]
  emitted_negative: dict[str, jax.Array]

  def tree_flatten(self):
    return (
        self.previous_p, self.weighted, self.denominator, self.negative_sum,
        self.p_change_sum, self.count, self.window_index, self.has_previous_p,
        self.emitted, self.emitted_start_step, self.emitted_end_step,
        self.emitted_p_change, self.emitted_scores, self.emitted_negative,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_window_state(params: PyTree) -> Exp7WindowState:
  zero_tree = _zeros_like(params)
  scalar = lambda value=0.0: jnp.asarray(value, jnp.float32)
  return Exp7WindowState(
      previous_p=zero_tree,
      weighted={path: zero_tree for path in PATHS},
      denominator={path: scalar() for path in PATHS},
      negative_sum={path: scalar() for path in PATHS},
      p_change_sum=scalar(), count=jnp.asarray(0, jnp.int32),
      window_index=jnp.asarray(0, jnp.int32), has_previous_p=jnp.asarray(False),
      emitted=jnp.asarray(False), emitted_start_step=jnp.asarray(0, jnp.int32),
      emitted_end_step=jnp.asarray(0, jnp.int32), emitted_p_change=scalar(),
      emitted_scores={path: scalar() for path in PATHS},
      emitted_negative={path: scalar() for path in PATHS},
  )


def _update(
    accumulator: Exp7WindowState, train: Exp7TrainState, *, beta1: float,
    beta2: float, learning_rate: float, weight_decay: float, eps: float,
    v_floor: float, window_size: int, eps_num: float,
) -> Exp7WindowState:
  t = train.step
  m_c = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), train.clean_m)
  m_dp = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), train.dp_m)
  m_noise = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), train.noise_m)
  moments = {"00": train.v00, "10": train.v10, "01": train.v01, "11": train.v11}
  vhats = {
      path: jax.tree_util.tree_map(lambda x: x / (1.0 - beta2 ** t), value)
      for path, value in moments.items()
  }
  phi_hat = train.bias_v / (1.0 - beta2 ** t)
  vhats["BC"] = jax.tree_util.tree_map(lambda x: x - phi_hat, vhats["11"])
  p = {
      path: jax.tree_util.tree_map(
          lambda x: 1.0 / (jnp.sqrt(jnp.maximum(x, v_floor if path == "BC" else 0.0)) + eps),
          value,
      ) for path, value in vhats.items()
  }
  p00 = p["00"]
  q = {
      # This algebraic form of q_00 exactly reuses Exp6's dynamic-clean-p path.
      "00": jax.tree_util.tree_map(lambda m, scale: m * scale, m_noise, p00),
  }
  for path in ("10", "01", "11", "BC"):
    q[path] = jax.tree_util.tree_map(
        lambda md, scale, mc, clean_scale: md * scale - mc * clean_scale,
        m_dp, p[path], m_c, p00,
    )
  x = {
      path: jax.tree_util.tree_map(lambda value: -learning_rate * value, q[path])
      for path in PATHS
  }
  decay = jnp.asarray(1.0 - learning_rate * weight_decay, jnp.float32)
  weighted = {
      path: jax.tree_util.tree_map(lambda old, value: decay * old + value,
                                   accumulator.weighted[path], x[path])
      for path in PATHS
  }
  denominator = {
      path: decay * decay * accumulator.denominator[path] + _sqnorm(x[path])
      for path in PATHS
  }
  negative = {path: _nonpositive_fraction(vhats[path]) for path in PATHS}
  negative_sum = {path: accumulator.negative_sum[path] + negative[path] for path in PATHS}
  previous_norm = jnp.sqrt(_sqnorm(accumulator.previous_p))
  p_change = jnp.where(
      accumulator.has_previous_p,
      jnp.sqrt(_sqnorm(jax.tree_util.tree_map(lambda a, b: a - b, p00, accumulator.previous_p)))
      / (previous_norm + eps_num),
      0.0,
  )
  count = accumulator.count + 1
  complete = count == window_size
  p_change_sum = accumulator.p_change_sum + p_change
  scores = {path: _sqnorm(weighted[path]) / (denominator[path] + eps_num) for path in PATHS}
  return Exp7WindowState(
      previous_p=p00,
      weighted={path: jax.tree_util.tree_map(lambda value: jnp.where(complete, 0.0, value), weighted[path]) for path in PATHS},
      denominator={path: jnp.where(complete, 0.0, denominator[path]) for path in PATHS},
      negative_sum={path: jnp.where(complete, 0.0, negative_sum[path]) for path in PATHS},
      p_change_sum=jnp.where(complete, 0.0, p_change_sum),
      count=jnp.where(complete, 0, count),
      window_index=accumulator.window_index + complete.astype(jnp.int32),
      has_previous_p=jnp.asarray(True), emitted=complete,
      emitted_start_step=accumulator.window_index * window_size + 1,
      emitted_end_step=t,
      emitted_p_change=jnp.where(complete, p_change_sum / count, 0.0),
      emitted_scores={path: jnp.where(complete, scores[path], 0.0) for path in PATHS},
      emitted_negative={path: jnp.where(complete, negative_sum[path] / count, 0.0) for path in PATHS},
  )


class Exp7WindowCollector:
  def __init__(
      self, params: PyTree, *, seed: int, algorithm: str, beta1: float,
      beta2: float, learning_rate: float, weight_decay: float, eps: float,
      v_floor: float = DEFAULT_V_FLOOR, window_size: int = 16,
      eps_num: float = 1e-30,
  ) -> None:
    self.seed, self.algorithm = int(seed), algorithm
    self.window_size, self.eps_num = int(window_size), float(eps_num)
    self._state = init_window_state(params)
    self._rows: list[dict[str, object]] = []
    self._compiled = jax.jit(lambda acc, train: _update(
        acc, train, beta1=beta1, beta2=beta2, learning_rate=learning_rate,
        weight_decay=weight_decay, eps=eps, v_floor=v_floor,
        window_size=window_size, eps_num=eps_num,
    ))

  @property
  def rows(self):
    return list(self._rows)

  def _append(self, *, start: int, end: int, p_change: float,
              scores: dict[str, float], negative: dict[str, float]) -> None:
    self._rows.append(window_row(
        seed=self.seed, algorithm=self.algorithm,
        window_index=(start - 1) // self.window_size,
        start_step=start, end_step=end, mean_p_relative_change=p_change,
        scores=scores, negative_fractions=negative,
    ))

  def after_step(self, state: Exp7TrainState, step: int) -> None:
    if step != int(np.asarray(state.step)):
      raise ValueError("callback step must equal train state step")
    self._state = self._compiled(self._state, state)
    if bool(np.asarray(self._state.emitted)):
      self._append(
          start=int(np.asarray(self._state.emitted_start_step)),
          end=int(np.asarray(self._state.emitted_end_step)),
          p_change=float(np.asarray(self._state.emitted_p_change)),
          scores={p: float(np.asarray(self._state.emitted_scores[p])) for p in PATHS},
          negative={p: float(np.asarray(self._state.emitted_negative[p])) for p in PATHS},
      )

  def finalize(self):
    state = self._state
    count = int(np.asarray(state.count))
    if count:
      start = int(np.asarray(state.window_index)) * self.window_size + 1
      scores = {
          path: float(np.asarray(_sqnorm(state.weighted[path]) / (state.denominator[path] + self.eps_num)))
          for path in PATHS
      }
      negative = {
          path: float(np.asarray(state.negative_sum[path])) / count for path in PATHS
      }
      self._append(
          start=start, end=start + count - 1,
          p_change=float(np.asarray(state.p_change_sum)) / count,
          scores=scores, negative=negative,
      )
      self._state = init_window_state(state.previous_p)
    return self.rows


__all__ = ["Exp7WindowCollector", "Exp7WindowState", "init_window_state"]
