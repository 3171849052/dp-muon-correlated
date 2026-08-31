"""Online exact-window and exact-stage accumulation for Experiment 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from exp8.core import BRANCHES, PATHS, Exp8DiagnosticStep, Exp8TrainState
from exp8.diagnostics import (
    DECOMP_FIELDS,
    METRIC_FIELDS,
    cancellation_metrics_from_jd,
    make_window_row,
)


PyTree = Any
WINDOW_SIZE = 16
_NUMERIC_EPS = 1e-30


def _scalar(value: float = 0.0) -> jax.Array:
  return jnp.asarray(value, jnp.float32)


def _tree_zeros_like(tree: PyTree) -> PyTree:
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _nested_tree(template: PyTree, value: float = 0.0) -> dict[str, dict[str, PyTree]]:
  return {branch: {path: _tree_zeros_like(template) for path in PATHS} for branch in BRANCHES}


def _nested_scalars(value: float = 0.0) -> dict[str, dict[str, jax.Array]]:
  return {branch: {path: _scalar(value) for path in PATHS} for branch in BRANCHES}


def _decomp_scalars(value: float = 0.0) -> dict[str, jax.Array]:
  return {field: _scalar(value) for field in DECOMP_FIELDS}


def _metric_scalars(value: float = 0.0) -> dict[str, dict[str, dict[str, jax.Array]]]:
  return {
      branch: {
          path: {field: _scalar(value) for field in METRIC_FIELDS}
          for path in PATHS
      } for branch in BRANCHES
  }


def _sqnorm(tree: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(leaf, jnp.float32) ** 2)
      for leaf in jax.tree_util.tree_leaves(tree)
  )


def _dot(left: PyTree, right: PyTree) -> jax.Array:
  return sum(
      jnp.sum(jnp.asarray(a, jnp.float32) * jnp.asarray(b, jnp.float32))
      for a, b in zip(
          jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
      )
  )


def _safe_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
  return jnp.where(
      jnp.isfinite(denominator) & (jnp.abs(denominator) > _NUMERIC_EPS),
      numerator / denominator,
      0.0,
  )


def _path_metrics(
    d: dict[str, dict[str, PyTree]],
    denominator: dict[str, dict[str, jax.Array]],
) -> dict[str, dict[str, dict[str, jax.Array]]]:
  result = _metric_scalars()
  for branch in BRANCHES:
    for path in PATHS:
      j = _sqnorm(d[branch][path])
      den = denominator[branch][path]
      result[branch][path] = {
          "J": j,
          "D": den,
          "C": _safe_ratio(j, den),
      }
  return result


def _reset_nested_tree(
    values: dict[str, dict[str, PyTree]], condition: jax.Array
) -> dict[str, dict[str, PyTree]]:
  return {
      branch: {
          path: jax.tree_util.tree_map(
              lambda value: jnp.where(condition, 0.0, value), values[branch][path]
          ) for path in PATHS
      } for branch in BRANCHES
  }


def _reset_nested_scalars(
    values: dict[str, dict[str, jax.Array]], condition: jax.Array
) -> dict[str, dict[str, jax.Array]]:
  return {
      branch: {
          path: jnp.where(condition, 0.0, values[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }


def _add_decomp(
    old: dict[str, jax.Array], step: Exp8DiagnosticStep
) -> dict[str, jax.Array]:
  values = {
      "A_energy": _sqnorm(step.A),
      "B_energy": _sqnorm(step.B),
      "I_energy": _sqnorm(step.I),
      "AB_dot": _dot(step.A, step.B),
      "AI_dot": _dot(step.A, step.I),
      "BI_dot": _dot(step.B, step.I),
      "reconstruction_error": step.reconstruction_error,
  }
  return {
      field: (
          jnp.maximum(old[field], values[field])
          if field == "reconstruction_error"
          else old[field] + values[field]
      ) for field in DECOMP_FIELDS
  }


def _reset_decomp(values: dict[str, jax.Array], condition: jax.Array) -> dict[str, jax.Array]:
  return {field: jnp.where(condition, 0.0, value) for field, value in values.items()}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Exp8AccumulatorState:
  window_d: dict[str, dict[str, PyTree]]
  window_D: dict[str, dict[str, jax.Array]]
  window_decomp: dict[str, jax.Array]
  window_count: jax.Array
  window_index: jax.Array
  full_d: dict[str, dict[str, PyTree]]
  full_D: dict[str, dict[str, jax.Array]]
  full_decomp: dict[str, jax.Array]
  stage_d: dict[str, dict[str, PyTree]]
  stage_D: dict[str, dict[str, jax.Array]]
  stage_decomp: dict[str, jax.Array]
  early_J: dict[str, dict[str, jax.Array]]
  early_D: dict[str, dict[str, jax.Array]]
  late_J: dict[str, dict[str, jax.Array]]
  late_D: dict[str, dict[str, jax.Array]]
  early_decomp: dict[str, jax.Array]
  late_decomp: dict[str, jax.Array]
  emitted: jax.Array
  emitted_start_step: jax.Array
  emitted_end_step: jax.Array
  emitted_metrics: dict[str, dict[str, dict[str, jax.Array]]]
  emitted_decomp: dict[str, jax.Array]

  def tree_flatten(self):
    return (
        self.window_d, self.window_D, self.window_decomp, self.window_count,
        self.window_index, self.full_d, self.full_D, self.full_decomp,
        self.stage_d, self.stage_D, self.stage_decomp, self.early_J,
        self.early_D, self.late_J, self.late_D, self.early_decomp,
        self.late_decomp, self.emitted, self.emitted_start_step,
        self.emitted_end_step, self.emitted_metrics, self.emitted_decomp,
    ), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def init_accumulator(params: PyTree) -> Exp8AccumulatorState:
  return Exp8AccumulatorState(
      window_d=_nested_tree(params), window_D=_nested_scalars(),
      window_decomp=_decomp_scalars(), window_count=jnp.asarray(0, jnp.int32),
      window_index=jnp.asarray(0, jnp.int32),
      full_d=_nested_tree(params), full_D=_nested_scalars(),
      full_decomp=_decomp_scalars(), stage_d=_nested_tree(params),
      stage_D=_nested_scalars(), stage_decomp=_decomp_scalars(),
      early_J=_nested_scalars(), early_D=_nested_scalars(),
      late_J=_nested_scalars(), late_D=_nested_scalars(),
      early_decomp=_decomp_scalars(), late_decomp=_decomp_scalars(),
      emitted=jnp.asarray(False), emitted_start_step=jnp.asarray(0, jnp.int32),
      emitted_end_step=jnp.asarray(0, jnp.int32),
      emitted_metrics=_metric_scalars(), emitted_decomp=_decomp_scalars(),
  )


def _update_accumulator(
    accumulator: Exp8AccumulatorState,
    step: Exp8DiagnosticStep,
    step_number: jax.Array,
    *,
    learning_rate: float,
    weight_decay: float,
    window_size: int,
    early_end: int,
    horizon: int,
) -> Exp8AccumulatorState:
  a = jnp.asarray(1.0 - learning_rate * weight_decay, jnp.float32)
  new_window_d = {
      branch: {
          path: jax.tree_util.tree_map(
              lambda old, x: a * old + x,
              accumulator.window_d[branch][path], step.x[branch][path],
          ) for path in PATHS
      } for branch in BRANCHES
  }
  new_full_d = {
      branch: {
          path: jax.tree_util.tree_map(
              lambda old, x: a * old + x,
              accumulator.full_d[branch][path], step.x[branch][path],
          ) for path in PATHS
      } for branch in BRANCHES
  }
  new_stage_d = {
      branch: {
          path: jax.tree_util.tree_map(
              lambda old, x: a * old + x,
              accumulator.stage_d[branch][path], step.x[branch][path],
          ) for path in PATHS
      } for branch in BRANCHES
  }
  new_window_D = {
      branch: {
          path: a * a * accumulator.window_D[branch][path] + _sqnorm(step.x[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  new_full_D = {
      branch: {
          path: a * a * accumulator.full_D[branch][path] + _sqnorm(step.x[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  new_stage_D = {
      branch: {
          path: a * a * accumulator.stage_D[branch][path] + _sqnorm(step.x[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  window_decomp = _add_decomp(accumulator.window_decomp, step)
  full_decomp = _add_decomp(accumulator.full_decomp, step)
  stage_decomp = _add_decomp(accumulator.stage_decomp, step)
  count = accumulator.window_count + 1
  complete = count == window_size
  emitted_metrics = _path_metrics(new_window_d, new_window_D)
  emitted_metrics = {
      branch: {
          path: {
              field: jnp.where(complete, value, 0.0)
              for field, value in emitted_metrics[branch][path].items()
          } for path in PATHS
      } for branch in BRANCHES
  }
  emitted_decomp = {
      field: jnp.where(complete, value, 0.0)
      for field, value in window_decomp.items()
  }
  early_boundary = step_number == early_end
  late_boundary = (step_number == horizon) & (horizon > early_end)
  stage_j = {
      branch: {path: _sqnorm(new_stage_d[branch][path]) for path in PATHS}
      for branch in BRANCHES
  }
  early_J = {
      branch: {
          path: jnp.where(early_boundary, stage_j[branch][path], accumulator.early_J[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  early_D = {
      branch: {
          path: jnp.where(early_boundary, new_stage_D[branch][path], accumulator.early_D[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  late_J = {
      branch: {
          path: jnp.where(late_boundary, stage_j[branch][path], accumulator.late_J[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  late_D = {
      branch: {
          path: jnp.where(late_boundary, new_stage_D[branch][path], accumulator.late_D[branch][path])
          for path in PATHS
      } for branch in BRANCHES
  }
  early_decomp = {
      field: jnp.where(early_boundary, value, accumulator.early_decomp[field])
      for field, value in stage_decomp.items()
  }
  late_decomp = {
      field: jnp.where(late_boundary, value, accumulator.late_decomp[field])
      for field, value in stage_decomp.items()
  }
  stage_boundary = early_boundary | late_boundary
  return Exp8AccumulatorState(
      window_d=_reset_nested_tree(new_window_d, complete),
      window_D=_reset_nested_scalars(new_window_D, complete),
      window_decomp=_reset_decomp(window_decomp, complete),
      window_count=jnp.where(complete, 0, count),
      window_index=accumulator.window_index + complete.astype(jnp.int32),
      full_d=new_full_d, full_D=new_full_D, full_decomp=full_decomp,
      stage_d=_reset_nested_tree(new_stage_d, stage_boundary),
      stage_D=_reset_nested_scalars(new_stage_D, stage_boundary),
      stage_decomp=_reset_decomp(stage_decomp, stage_boundary),
      early_J=early_J, early_D=early_D, late_J=late_J, late_D=late_D,
      early_decomp=early_decomp, late_decomp=late_decomp,
      emitted=complete,
      emitted_start_step=accumulator.window_index * window_size + 1,
      emitted_end_step=step_number,
      emitted_metrics=emitted_metrics,
      emitted_decomp=emitted_decomp,
  )


def _host_metrics(metrics: Mapping[str, Mapping[str, Mapping[str, jax.Array]]]):
  return {
      branch: {
          path: {field: float(np.asarray(value)) for field, value in values.items()}
          for path, values in paths.items()
      } for branch, paths in metrics.items()
  }


def _host_decomp(values: Mapping[str, jax.Array]) -> dict[str, float]:
  return {field: float(np.asarray(value)) for field, value in values.items()}


class Exp8WindowCollector:
  """Collect exact 16-step rows and exact early/late/full accumulations."""

  def __init__(
      self,
      params: PyTree,
      *,
      seed: int,
      learning_rate: float,
      weight_decay: float,
      horizon: int,
      window_size: int = WINDOW_SIZE,
  ) -> None:
    if horizon < 1 or window_size < 1:
      raise ValueError("horizon and window_size must be positive")
    self.seed = int(seed)
    self.horizon = int(horizon)
    self.window_size = int(window_size)
    self._rows: list[dict[str, object]] = []
    self._state = init_accumulator(params)
    early_end = min(97, self.horizon)
    self._compiled = jax.jit(
        lambda accumulator, step, step_number: _update_accumulator(
            accumulator, step, step_number,
            learning_rate=learning_rate, weight_decay=weight_decay,
            window_size=self.window_size, early_end=early_end,
            horizon=self.horizon,
        )
    )

  @property
  def rows(self) -> list[dict[str, object]]:
    return list(self._rows)

  def _append_row(
      self,
      *,
      start_step: int,
      end_step: int,
      metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
      decomp: Mapping[str, float],
      window_index: int,
  ) -> None:
    self._rows.append(make_window_row(
        seed=self.seed, window_index=window_index,
        start_step=start_step, end_step=end_step,
        metrics=metrics, decomp=decomp,
    ))

  def after_step(self, state: Exp8TrainState, step: int) -> None:
    if step != int(np.asarray(state.step)):
      raise ValueError("callback step must equal Exp8 train state step")
    self._state = self._compiled(self._state, state.last_step, state.step)
    if bool(np.asarray(self._state.emitted)):
      self._append_row(
          start_step=int(np.asarray(self._state.emitted_start_step)),
          end_step=int(np.asarray(self._state.emitted_end_step)),
          metrics=_host_metrics(self._state.emitted_metrics),
          decomp=_host_decomp(self._state.emitted_decomp),
          window_index=(int(np.asarray(self._state.emitted_start_step)) - 1) // self.window_size,
      )

  def finalize(self) -> list[dict[str, object]]:
    count = int(np.asarray(self._state.window_count))
    if count:
      start = int(np.asarray(self._state.window_index)) * self.window_size + 1
      self._append_row(
          start_step=start, end_step=start + count - 1,
          metrics=_host_metrics(_path_metrics(self._state.window_d, self._state.window_D)),
          decomp=_host_decomp(self._state.window_decomp),
          window_index=(start - 1) // self.window_size,
      )
    return self.rows

  def _stage_jd(self, stage: str):
    if stage == "early":
      return self._state.early_J, self._state.early_D, self._state.early_decomp
    if stage == "late":
      return self._state.late_J, self._state.late_D, self._state.late_decomp
    if stage == "full":
      return (
          {
              branch: {path: _sqnorm(self._state.full_d[branch][path]) for path in PATHS}
              for branch in BRANCHES
          },
          self._state.full_D,
          self._state.full_decomp,
      )
    raise ValueError(f"unknown stage {stage!r}")

  def stage_summaries(self) -> dict[str, dict[str, object]]:
    result = {}
    for stage in ("early", "late", "full"):
      start, end = {
          "early": (1, min(97, self.horizon)),
          "late": (98, self.horizon),
          "full": (1, self.horizon),
      }[stage]
      j, d, decomp = self._stage_jd(stage)
      metrics = {
          branch: {
              path: cancellation_metrics_from_jd(j[branch][path], d[branch][path])
              for path in PATHS
          } for branch in BRANCHES
      }
      result[stage] = {
          "start_step": start, "end_step": end,
          "num_steps": max(0, end - start + 1),
          "metrics": _host_metrics(metrics),
          "decomposition": _host_decomp(decomp),
      }
    return result


__all__ = ["Exp8AccumulatorState", "Exp8WindowCollector", "WINDOW_SIZE", "init_accumulator"]
