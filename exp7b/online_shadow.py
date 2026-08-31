"""Device-resident cancellation and stability diagnostics for Experiment 7b."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from exp7.core import Exp7TrainState
from exp7.diagnostics import PATHS
from exp7b.core import paper_bc_preconditioner
from exp7b.diagnostics import window_row


PyTree = Any
NORM_NAMES = (
    "raw_optimizer_update_l2", "applied_parameter_update_l2", "parameter_l2"
)
DEFAULT_HISTOGRAM_BINS = 4096


def _zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _sqnorm(tree: PyTree) -> jax.Array:
  return sum(jnp.sum(jnp.asarray(x, jnp.float32) ** 2)
             for x in jax.tree_util.tree_leaves(tree))


def _fraction(tree: PyTree, predicate) -> jax.Array:
  leaves = jax.tree_util.tree_leaves(tree)
  count = sum(leaf.size for leaf in leaves)
  selected = sum(jnp.sum(predicate(jnp.asarray(leaf))) for leaf in leaves)
  return selected / jnp.asarray(count, jnp.float32)


def _p_histogram(tree: PyTree, p_max: jax.Array, bins: int) -> jax.Array:
  histogram = jnp.zeros((bins,), jnp.int32)
  for leaf in jax.tree_util.tree_leaves(tree):
    values = jnp.asarray(leaf, jnp.float32).reshape(-1)
    indices = jnp.minimum(
        jnp.floor(values / p_max * bins).astype(jnp.int32), bins - 1
    )
    histogram = histogram + jnp.bincount(indices, length=bins)
  return histogram


def _histogram_quantile(histogram: jax.Array, q: float, p_max: jax.Array) -> jax.Array:
  total = jnp.sum(histogram)
  target = jnp.ceil(jnp.asarray(q, jnp.float32) * total)
  index = jnp.argmax(jnp.cumsum(histogram) >= target)
  # The upper bin edge is conservative and the last bin is exactly p_max.
  return p_max * (index.astype(jnp.float32) + 1.0) / histogram.size


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp7bWindowState:
  previous_p: PyTree
  previous_params: PyTree
  weighted: dict[str, PyTree]
  denominator: dict[str, jax.Array]
  negative_sum: dict[str, jax.Array]
  corrected_nonpositive_sum: jax.Array
  floor_sum: jax.Array
  p_histogram: jax.Array
  p_maximum: jax.Array
  norm_sum: dict[str, jax.Array]
  norm_square_sum: dict[str, jax.Array]
  norm_minimum: dict[str, jax.Array]
  norm_maximum: dict[str, jax.Array]
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
  emitted_stability: dict[str, jax.Array]
  latest_negative: dict[str, jax.Array]
  latest_corrected_nonpositive: jax.Array
  latest_floor: jax.Array
  latest_p_histogram: jax.Array
  latest_p_maximum: jax.Array
  latest_norms: dict[str, jax.Array]

  def tree_flatten(self):
    return (
        self.previous_p, self.previous_params, self.weighted, self.denominator,
        self.negative_sum, self.corrected_nonpositive_sum, self.floor_sum,
        self.p_histogram, self.p_maximum, self.norm_sum, self.norm_square_sum,
        self.norm_minimum, self.norm_maximum, self.p_change_sum, self.count,
        self.window_index, self.has_previous_p, self.emitted,
        self.emitted_start_step, self.emitted_end_step, self.emitted_p_change,
        self.emitted_scores, self.emitted_negative, self.emitted_stability,
        self.latest_negative, self.latest_corrected_nonpositive,
        self.latest_floor, self.latest_p_histogram, self.latest_p_maximum,
        self.latest_norms,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def _scalar(value=0.0):
  return jnp.asarray(value, jnp.float32)


def init_window_state(params: PyTree, histogram_bins: int = DEFAULT_HISTOGRAM_BINS) -> Exp7bWindowState:
  if histogram_bins < 2:
    raise ValueError("histogram_bins must be at least 2")
  zero_tree = _zeros_like(params)
  return Exp7bWindowState(
      previous_p=zero_tree, previous_params=params,
      weighted={path: zero_tree for path in PATHS},
      denominator={path: _scalar() for path in PATHS},
      negative_sum={path: _scalar() for path in PATHS},
      corrected_nonpositive_sum=_scalar(), floor_sum=_scalar(),
      p_histogram=jnp.zeros((histogram_bins,), jnp.int32), p_maximum=_scalar(),
      norm_sum={name: _scalar() for name in NORM_NAMES},
      norm_square_sum={name: _scalar() for name in NORM_NAMES},
      norm_minimum={name: _scalar(jnp.inf) for name in NORM_NAMES},
      norm_maximum={name: _scalar() for name in NORM_NAMES},
      p_change_sum=_scalar(), count=jnp.asarray(0, jnp.int32),
      window_index=jnp.asarray(0, jnp.int32), has_previous_p=jnp.asarray(False),
      emitted=jnp.asarray(False), emitted_start_step=jnp.asarray(0, jnp.int32),
      emitted_end_step=jnp.asarray(0, jnp.int32), emitted_p_change=_scalar(),
      emitted_scores={path: _scalar() for path in PATHS},
      emitted_negative={path: _scalar() for path in PATHS},
      emitted_stability={},
      latest_negative={path: _scalar() for path in PATHS},
      latest_corrected_nonpositive=_scalar(), latest_floor=_scalar(),
      latest_p_histogram=jnp.zeros((histogram_bins,), jnp.int32),
      latest_p_maximum=_scalar(),
      latest_norms={name: _scalar() for name in NORM_NAMES},
  )


def _stability_values(
    *, count: jax.Array, corrected_sum: jax.Array, floor_sum: jax.Array,
    histogram: jax.Array, p_maximum: jax.Array, implied_p_max: jax.Array,
    norm_sum: dict[str, jax.Array], norm_square_sum: dict[str, jax.Array],
    norm_minimum: dict[str, jax.Array], norm_maximum: dict[str, jax.Array],
) -> dict[str, jax.Array]:
  count_f = count.astype(jnp.float32)
  values = {
      "corrected_v_nonpositive_fraction": corrected_sum / count_f,
      "floor_activation_fraction": floor_sum / count_f,
      "p_bc_median": _histogram_quantile(histogram, .5, implied_p_max),
      "p_bc_q99": _histogram_quantile(histogram, .99, implied_p_max),
      "p_bc_q99_9": _histogram_quantile(histogram, .999, implied_p_max),
      "p_bc_max": p_maximum,
  }
  for name in NORM_NAMES:
    mean = norm_sum[name] / count_f
    variance = jnp.maximum(norm_square_sum[name] / count_f - mean * mean, 0.0)
    values[f"{name}_mean"] = mean
    values[f"{name}_std"] = jnp.sqrt(variance)
    values[f"{name}_min"] = norm_minimum[name]
    values[f"{name}_max"] = norm_maximum[name]
  return values


def _update(
    accumulator: Exp7bWindowState, train: Exp7TrainState, *, algorithm: str,
    beta1: float, beta2: float, learning_rate: float, weight_decay: float,
    eps: float, gamma_prime: float, window_size: int, eps_num: float,
) -> Exp7bWindowState:
  t = train.step
  m_c = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), train.clean_m)
  m_dp = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), train.dp_m)
  m_noise = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), train.noise_m)
  moments = {"00": train.v00, "10": train.v10, "01": train.v01, "11": train.v11}
  vhats = {path: jax.tree_util.tree_map(lambda x: x / (1.0 - beta2 ** t), value)
           for path, value in moments.items()}
  phi_hat = train.bias_v / (1.0 - beta2 ** t)
  corrected = jax.tree_util.tree_map(lambda x: x - phi_hat, vhats["11"])
  vhats["BC"] = corrected
  p = {
      path: jax.tree_util.tree_map(
          lambda x: 1.0 / (jnp.sqrt(jnp.maximum(x, 0.0)) + eps), value
      ) for path, value in vhats.items() if path != "BC"
  }
  p["BC"] = jax.tree_util.tree_map(
      lambda x: paper_bc_preconditioner(x, gamma_prime), corrected
  )
  p00 = p["00"]
  q = {"00": jax.tree_util.tree_map(lambda m, scale: m * scale, m_noise, p00)}
  for path in ("10", "01", "11", "BC"):
    q[path] = jax.tree_util.tree_map(
        lambda md, scale, mc, clean_scale: md * scale - mc * clean_scale,
        m_dp, p[path], m_c, p00,
    )
  x = {path: jax.tree_util.tree_map(lambda value: -learning_rate * value, q[path])
       for path in PATHS}
  decay = jnp.asarray(1.0 - learning_rate * weight_decay, jnp.float32)
  weighted = {path: jax.tree_util.tree_map(
      lambda old, value: decay * old + value, accumulator.weighted[path], x[path]
  ) for path in PATHS}
  denominator = {path: decay * decay * accumulator.denominator[path] + _sqnorm(x[path])
                 for path in PATHS}
  negative = {path: _fraction(vhats[path], lambda value: value <= 0) for path in PATHS}
  negative_sum = {path: accumulator.negative_sum[path] + negative[path] for path in PATHS}

  corrected_nonpositive = _fraction(corrected, lambda value: value <= 0)
  floor = _fraction(corrected, lambda value: value <= gamma_prime)
  implied_p_max = jax.lax.rsqrt(jnp.asarray(gamma_prime, jnp.float32))
  latest_histogram = _p_histogram(
      p["BC"], implied_p_max, accumulator.p_histogram.size
  )
  histogram = accumulator.p_histogram + latest_histogram
  current_p_max = jnp.max(jnp.stack([
      jnp.max(leaf) for leaf in jax.tree_util.tree_leaves(p["BC"])
  ]))
  p_maximum = jnp.maximum(accumulator.p_maximum, current_p_max)

  actual_p = p["11"] if algorithm == "baseline" else p["BC"]
  raw_update = jax.tree_util.tree_map(lambda m, scale: m * scale, m_dp, actual_p)
  applied_update = jax.tree_util.tree_map(
      lambda new, old: new - old, train.params, accumulator.previous_params
  )
  norms = {
      "raw_optimizer_update_l2": jnp.sqrt(_sqnorm(raw_update)),
      "applied_parameter_update_l2": jnp.sqrt(_sqnorm(applied_update)),
      "parameter_l2": jnp.sqrt(_sqnorm(train.params)),
  }
  norm_sum = {name: accumulator.norm_sum[name] + norms[name] for name in NORM_NAMES}
  norm_square_sum = {
      name: accumulator.norm_square_sum[name] + norms[name] ** 2 for name in NORM_NAMES
  }
  norm_minimum = {name: jnp.minimum(accumulator.norm_minimum[name], norms[name])
                  for name in NORM_NAMES}
  norm_maximum = {name: jnp.maximum(accumulator.norm_maximum[name], norms[name])
                  for name in NORM_NAMES}

  previous_norm = jnp.sqrt(_sqnorm(accumulator.previous_p))
  p_change = jnp.where(
      accumulator.has_previous_p,
      jnp.sqrt(_sqnorm(jax.tree_util.tree_map(
          lambda a, b: a - b, p00, accumulator.previous_p
      ))) / (previous_norm + eps_num), 0.0,
  )
  count = accumulator.count + 1
  complete = count == window_size
  p_change_sum = accumulator.p_change_sum + p_change
  scores = {path: _sqnorm(weighted[path]) / (denominator[path] + eps_num) for path in PATHS}
  stability = _stability_values(
      count=count,
      corrected_sum=accumulator.corrected_nonpositive_sum + corrected_nonpositive,
      floor_sum=accumulator.floor_sum + floor,
      histogram=histogram, p_maximum=p_maximum, implied_p_max=implied_p_max,
      norm_sum=norm_sum, norm_square_sum=norm_square_sum,
      norm_minimum=norm_minimum, norm_maximum=norm_maximum,
  )
  reset_min = {name: jnp.where(complete, jnp.inf, norm_minimum[name]) for name in NORM_NAMES}
  return Exp7bWindowState(
      previous_p=p00, previous_params=train.params,
      weighted={path: jax.tree_util.tree_map(
          lambda value: jnp.where(complete, 0.0, value), weighted[path]
      ) for path in PATHS},
      denominator={path: jnp.where(complete, 0.0, denominator[path]) for path in PATHS},
      negative_sum={path: jnp.where(complete, 0.0, negative_sum[path]) for path in PATHS},
      corrected_nonpositive_sum=jnp.where(
          complete, 0.0, accumulator.corrected_nonpositive_sum + corrected_nonpositive
      ),
      floor_sum=jnp.where(complete, 0.0, accumulator.floor_sum + floor),
      p_histogram=jnp.where(complete, jnp.zeros_like(histogram), histogram),
      p_maximum=jnp.where(complete, 0.0, p_maximum),
      norm_sum={name: jnp.where(complete, 0.0, norm_sum[name]) for name in NORM_NAMES},
      norm_square_sum={name: jnp.where(complete, 0.0, norm_square_sum[name])
                       for name in NORM_NAMES},
      norm_minimum=reset_min,
      norm_maximum={name: jnp.where(complete, 0.0, norm_maximum[name]) for name in NORM_NAMES},
      p_change_sum=jnp.where(complete, 0.0, p_change_sum),
      count=jnp.where(complete, 0, count),
      window_index=accumulator.window_index + complete.astype(jnp.int32),
      has_previous_p=jnp.asarray(True), emitted=complete,
      emitted_start_step=accumulator.window_index * window_size + 1,
      emitted_end_step=t,
      emitted_p_change=jnp.where(complete, p_change_sum / count, 0.0),
      emitted_scores={path: jnp.where(complete, scores[path], 0.0) for path in PATHS},
      emitted_negative={path: jnp.where(complete, negative_sum[path] / count, 0.0)
                        for path in PATHS},
      emitted_stability={name: jnp.where(complete, value, 0.0)
                         for name, value in stability.items()},
      latest_negative=negative,
      latest_corrected_nonpositive=corrected_nonpositive,
      latest_floor=floor,
      latest_p_histogram=latest_histogram,
      latest_p_maximum=current_p_max,
      latest_norms=norms,
  )


class Exp7bWindowCollector:
  def __init__(
      self, params: PyTree, *, seed: int, algorithm: str, beta1: float,
      beta2: float, learning_rate: float, weight_decay: float, eps: float,
      gamma_prime: float, window_size: int = 16, eps_num: float = 1e-30,
      histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
  ) -> None:
    if algorithm not in {"baseline", "bc"}:
      raise ValueError("algorithm must be baseline or bc")
    self.seed, self.algorithm = int(seed), algorithm
    self.gamma_prime = float(gamma_prime)
    self.window_size, self.eps_num = int(window_size), float(eps_num)
    self._state = init_window_state(params, histogram_bins)
    self._rows: list[dict[str, object]] = []
    self._step_diagnostics: list[dict[str, object]] = []
    self._compiled = jax.jit(lambda acc, train: _update(
        acc, train, algorithm=algorithm, beta1=beta1, beta2=beta2,
        learning_rate=learning_rate, weight_decay=weight_decay, eps=eps,
        gamma_prime=gamma_prime, window_size=window_size, eps_num=eps_num,
    ))

  @property
  def rows(self):
    return list(self._rows)

  @property
  def step_diagnostics(self):
    return list(self._step_diagnostics)

  def _append(self, *, start: int, end: int, p_change: float,
              scores: dict[str, float], negative: dict[str, float],
              stability: dict[str, float]) -> None:
    self._rows.append(window_row(
        seed=self.seed, algorithm=self.algorithm,
        window_index=(start - 1) // self.window_size,
        start_step=start, end_step=end, mean_p_relative_change=p_change,
        scores=scores, negative_fractions=negative, stability=stability,
    ))

  def after_step(self, state: Exp7TrainState, step: int) -> None:
    if step != int(np.asarray(state.step)):
      raise ValueError("callback step must equal train state step")
    self._state = self._compiled(self._state, state)
    self._step_diagnostics.append({
        "seed": self.seed,
        "algorithm": self.algorithm,
        "step": step,
        **{
            f"nonpositive_v_fraction_{path}": float(np.asarray(
                self._state.latest_negative[path]
            )) for path in PATHS
        },
        "corrected_v_nonpositive_fraction": float(np.asarray(
            self._state.latest_corrected_nonpositive
        )),
        "floor_activation_fraction": float(np.asarray(self._state.latest_floor)),
        "p_bc_histogram": np.asarray(self._state.latest_p_histogram, np.int64),
        "p_bc_max": float(np.asarray(self._state.latest_p_maximum)),
        **{
            name: float(np.asarray(self._state.latest_norms[name]))
            for name in NORM_NAMES
        },
    })
    if bool(np.asarray(self._state.emitted)):
      self._append(
          start=int(np.asarray(self._state.emitted_start_step)),
          end=int(np.asarray(self._state.emitted_end_step)),
          p_change=float(np.asarray(self._state.emitted_p_change)),
          scores={path: float(np.asarray(self._state.emitted_scores[path])) for path in PATHS},
          negative={path: float(np.asarray(self._state.emitted_negative[path])) for path in PATHS},
          stability={name: float(np.asarray(value))
                     for name, value in self._state.emitted_stability.items()},
      )

  def finalize(self):
    state = self._state
    count = int(np.asarray(state.count))
    if count:
      start = int(np.asarray(state.window_index)) * self.window_size + 1
      scores = {path: float(np.asarray(
          _sqnorm(state.weighted[path]) / (state.denominator[path] + self.eps_num)
      )) for path in PATHS}
      negative = {path: float(np.asarray(state.negative_sum[path])) / count for path in PATHS}
      stability_arrays = _stability_values(
          count=state.count, corrected_sum=state.corrected_nonpositive_sum,
          floor_sum=state.floor_sum, histogram=state.p_histogram,
          p_maximum=state.p_maximum,
          implied_p_max=jax.lax.rsqrt(jnp.asarray(self.gamma_prime, jnp.float32)),
          norm_sum=state.norm_sum, norm_square_sum=state.norm_square_sum,
          norm_minimum=state.norm_minimum, norm_maximum=state.norm_maximum,
      )
      self._append(
          start=start, end=start + count - 1,
          p_change=float(np.asarray(state.p_change_sum)) / count,
          scores=scores, negative=negative,
          stability={name: float(np.asarray(value))
                     for name, value in stability_arrays.items()},
      )
      self._state = init_window_state(state.previous_params, state.p_histogram.size)
    return self.rows


__all__ = [
    "DEFAULT_HISTOGRAM_BINS", "Exp7bWindowCollector", "Exp7bWindowState",
    "NORM_NAMES", "init_window_state",
]
