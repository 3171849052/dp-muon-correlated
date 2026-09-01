"""Exp10 scalar metrics, paired stages, and compact grouped histograms."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from exp10.core import (
    BRANCHES,
    EMA_COMPONENTS,
    Exp10Step,
    Exp10TrainState,
    INSTANTANEOUS_COMPONENTS,
)


WINDOW_SIZE = 16
HISTOGRAM_GROUPS = (
    "instantaneous_signal_cross",
    "instantaneous_noise",
    "ema_signal_cross",
    "ema_noise",
)
HISTOGRAM_GROUP_COMPONENTS = {
    "instantaneous_signal_cross": ("g2", "g2_cross"),
    "instantaneous_noise": ("xi2",),
    "ema_signal_cross": ("V_g", "V_g_cross"),
    "ema_noise": ("V_xi",),
}
HISTOGRAM_COMPONENT_SLOTS = 2

METRIC_FIELDS = (
    "mean_g2",
    "mean_g2_cross",
    "mean_xi2",
    "mean_2gxi",
    "mean_abs_2gxi",
    "rms_2gxi",
    "negative_fraction_g2_cross",
    "R_signed",
    "R_abs",
    "R_noise",
    "rho_fb",
    "mean_g2_cross_minus_g2",
    "mean_private_second_moment",
)
ERROR_FIELDS = ("private_v_decomposition_max_abs", "private_v_decomposition_rms")
STEP_FIELDS = (
    "seed", "step", "branch", "phi_t", "num_coordinates",
) + METRIC_FIELDS + ERROR_FIELDS
STAGE_FIELDS = (
    "seed", "stage", "branch", "start_step", "end_step", "num_steps",
    "num_coordinate_observations", "mean_phi_t",
) + METRIC_FIELDS + ERROR_FIELDS
PAIRED_FIELDS = (
    "delta_traj", "delta_feedback", "delta_noise", "delta_total",
    "delta_decomposition_residual",
)


def histogram_checkpoint_steps(horizon: int, window_size: int = WINDOW_SIZE) -> list[int]:
  """Return aligned checkpoints plus the final non-aligned step if needed."""
  if horizon < 1 or window_size < 1:
    raise ValueError("horizon and window_size must be positive")
  result = list(range(window_size, horizon + 1, window_size))
  if not result or result[-1] != horizon:
    result.append(horizon)
  return result


def stage_bounds(horizon: int) -> dict[str, tuple[int, int]]:
  """Return only semantically valid stages for this horizon."""
  if horizon < 1:
    raise ValueError("horizon must be positive")
  result = {"early": (1, min(97, horizon))}
  if horizon >= 98:
    result["late"] = (98, horizon)
  result["full"] = (1, horizon)
  return result


def _host_scalar(value: Any) -> float:
  result = float(np.asarray(value))
  if not np.isfinite(result):
    raise FloatingPointError(f"non-finite Exp10 diagnostic scalar: {result}")
  return result


def _tree_leaves(tree: Any) -> list[Any]:
  import jax
  return list(jax.tree_util.tree_leaves(tree))


def _flatten_tree(tree: Any) -> np.ndarray:
  leaves = [np.asarray(leaf).reshape(-1) for leaf in _tree_leaves(tree)]
  if not leaves:
    raise ValueError("histogram input tree must contain at least one leaf")
  values = np.concatenate(leaves).astype(np.float64, copy=False)
  if not np.all(np.isfinite(values)):
    raise FloatingPointError("histogram input contains non-finite values")
  return values


def _group_values(last_step: Exp10Step, group: str) -> dict[tuple[str, str], np.ndarray]:
  values: dict[tuple[str, str], np.ndarray] = {}
  for branch in BRANCHES:
    for component in HISTOGRAM_GROUP_COMPONENTS[group]:
      tree = (
          last_step.instantaneous[branch][component]
          if component in INSTANTANEOUS_COMPONENTS
          else last_step.ema[branch][component]
      )
      values[(branch, component)] = _flatten_tree(tree)
  return values


def histogram_extrema(last_step: Exp10Step) -> dict[str, tuple[float, float]]:
  """Collect only group min/max values for the first histogram pass."""
  result = {}
  for group in HISTOGRAM_GROUPS:
    values = np.concatenate(list(_group_values(last_step, group).values()))
    result[group] = (float(np.min(values)), float(np.max(values)))
  return result


def _edges_for_range(low: float, high: float, bins: int) -> np.ndarray:
  if bins < 1:
    raise ValueError("bins must be positive")
  if not np.isfinite(low) or not np.isfinite(high):
    raise FloatingPointError("histogram range must be finite")
  if low == high:
    padding = max(0.5, abs(low) * 0.01)
    low, high = low - padding, high + padding
  return np.linspace(low, high, bins + 1, dtype=np.float64)


def _histogram_record(
    *,
    seed: int,
    step: int,
    last_step: Exp10Step,
    bins: int,
    group_edges: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
  """Build one record; each group has its own shared linear edge vector."""
  values_by_group = {
      group: _group_values(last_step, group) for group in HISTOGRAM_GROUPS
  }
  edges = np.zeros((len(HISTOGRAM_GROUPS), bins + 1), dtype=np.float64)
  counts = np.zeros(
      (len(BRANCHES), len(HISTOGRAM_GROUPS), HISTOGRAM_COMPONENT_SLOTS, bins),
      dtype=np.int64,
  )
  names = np.full(
      (len(HISTOGRAM_GROUPS), HISTOGRAM_COMPONENT_SLOTS), "", dtype="U32"
  )
  for group_index, group in enumerate(HISTOGRAM_GROUPS):
    components = HISTOGRAM_GROUP_COMPONENTS[group]
    names[group_index, :len(components)] = components
    if group_edges is None:
      combined = np.concatenate(list(values_by_group[group].values()))
      edge = _edges_for_range(float(np.min(combined)), float(np.max(combined)), bins)
    else:
      edge = np.asarray(group_edges[group], dtype=np.float64)
      if edge.shape != (bins + 1,):
        raise ValueError(f"group {group!r} edges have the wrong shape")
    edges[group_index] = edge
    for branch_index, branch in enumerate(BRANCHES):
      for component_index, component in enumerate(components):
        counts[branch_index, group_index, component_index], _ = np.histogram(
            values_by_group[group][(branch, component)], bins=edge
        )
  totals = counts.sum(axis=-1, keepdims=True)
  relative = np.divide(
      counts.astype(np.float32),
      totals,
      out=np.zeros_like(counts, dtype=np.float32),
      where=totals != 0,
  )
  return {
      "seed": int(seed),
      "step": int(step),
      "group_bin_edges": edges,
      "counts": counts,
      "relative_frequency": relative,
      "group_component_names": names,
  }


class PooledHistogramBuilder:
  """Two-pass pooled histogram builder that never retains raw coordinates."""

  def __init__(self, *, horizon: int, bins: int = 64, window_size: int = WINDOW_SIZE):
    if horizon < 1 or bins < 1:
      raise ValueError("horizon and bins must be positive")
    self.horizon = int(horizon)
    self.bins = int(bins)
    self.steps = histogram_checkpoint_steps(horizon, window_size)
    self._ranges: dict[int, dict[str, list[float]]] = {}
    self._edges: dict[int, dict[str, np.ndarray]] = {}
    self._counts: dict[int, np.ndarray] = {}
    self._source_seeds: dict[int, set[int]] = {}

  def observe_extrema(self, state: Exp10TrainState, step: int) -> None:
    if int(step) not in self.steps:
      return
    extrema = histogram_extrema(state.last_step)
    ranges = self._ranges.setdefault(int(step), {})
    for group, (low, high) in extrema.items():
      if group not in ranges:
        ranges[group] = [low, high]
      else:
        ranges[group][0] = min(ranges[group][0], low)
        ranges[group][1] = max(ranges[group][1], high)

  def finalize_edges(self) -> None:
    missing = [step for step in self.steps if step not in self._ranges]
    if missing:
      raise ValueError(f"missing histogram extrema for steps {missing}")
    self._edges = {
        step: {
            group: _edges_for_range(low, high, self.bins)
            for group, (low, high) in ranges.items()
        }
        for step, ranges in self._ranges.items()
    }
    self._counts = {
        step: np.zeros(
            (len(BRANCHES), len(HISTOGRAM_GROUPS), HISTOGRAM_COMPONENT_SLOTS, self.bins),
            dtype=np.int64,
        )
        for step in self.steps
    }

  def add_state(self, seed: int, state: Exp10TrainState, step: int) -> dict[str, Any] | None:
    if int(step) not in self.steps:
      return None
    if int(step) not in self._edges:
      raise RuntimeError("finalize_edges() must run before add_state()")
    record = _histogram_record(
        seed=seed,
        step=step,
        last_step=state.last_step,
        bins=self.bins,
        group_edges=self._edges[int(step)],
    )
    self._counts[int(step)] += record["counts"]
    self._source_seeds.setdefault(int(step), set()).add(int(seed))
    return record

  def pooled_records(self) -> list[dict[str, Any]]:
    if not self._edges:
      raise RuntimeError("finalize_edges() must run before pooled_records()")
    names = np.asarray([
        [HISTOGRAM_GROUP_COMPONENTS[group][slot]
         if slot < len(HISTOGRAM_GROUP_COMPONENTS[group]) else ""
         for slot in range(HISTOGRAM_COMPONENT_SLOTS)]
        for group in HISTOGRAM_GROUPS
    ], dtype="U32")
    result = []
    for step in self.steps:
      counts = self._counts[step]
      totals = counts.sum(axis=-1, keepdims=True)
      relative = np.divide(
          counts.astype(np.float32), totals,
          out=np.zeros_like(counts, dtype=np.float32), where=totals != 0,
      )
      result.append({
          "seed": -1,
          "step": step,
          "group_bin_edges": np.stack([
              self._edges[step][group] for group in HISTOGRAM_GROUPS
          ]),
          "counts": counts.copy(),
          "relative_frequency": relative,
          "group_component_names": names.copy(),
          "source_seeds": np.asarray(sorted(self._source_seeds.get(step, set())), dtype=np.int32),
      })
    return result


def _weighted_mean(rows: list[Mapping[str, object]], field: str) -> float:
  if not rows:
    return 0.0
  weights = np.asarray([float(row["num_coordinates"]) for row in rows])
  values = np.asarray([float(row[field]) for row in rows])
  total = float(np.sum(weights))
  return float(np.sum(values * weights) / total) if total > 0 else 0.0


def _safe_ratio_host(numerator: float, denominator: float) -> float:
  if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-30:
    return 0.0
  return float(numerator / denominator)


def _stage_row(
    rows: list[Mapping[str, object]], *, seed: int, branch: str,
    stage: str, start: int, end: int
) -> dict[str, object]:
  result: dict[str, object] = {
      "seed": int(seed), "stage": stage, "branch": branch,
      "start_step": int(start), "end_step": int(end),
      "num_steps": max(0, end - start + 1),
      "num_coordinate_observations": int(sum(
          int(row["num_coordinates"]) for row in rows
      )),
      "mean_phi_t": _weighted_mean(rows, "phi_t"),
  }
  for field in (
      "mean_g2", "mean_g2_cross", "mean_xi2", "mean_2gxi",
      "mean_abs_2gxi", "negative_fraction_g2_cross",
      "mean_g2_cross_minus_g2", "mean_private_second_moment",
  ):
    result[field] = _weighted_mean(rows, field)
  if rows:
    weights = np.asarray([float(row["num_coordinates"]) for row in rows])
    total = float(np.sum(weights))
    rms_sq = np.asarray([float(row["rms_2gxi"]) ** 2 for row in rows])
    result["rms_2gxi"] = float(np.sqrt(np.sum(weights * rms_sq) / total))
    sum_g2 = sum(float(row["mean_g2"]) * float(row["num_coordinates"]) for row in rows)
    sum_term = sum(float(row["mean_2gxi"]) * float(row["num_coordinates"]) for row in rows)
    sum_abs = sum(float(row["mean_abs_2gxi"]) * float(row["num_coordinates"]) for row in rows)
    sum_xi2 = sum(float(row["mean_xi2"]) * float(row["num_coordinates"]) for row in rows)
    result["R_signed"] = _safe_ratio_host(sum_term, sum_g2)
    result["R_abs"] = _safe_ratio_host(sum_abs, sum_g2)
    result["R_noise"] = _safe_ratio_host(sum_xi2, sum_g2)
    result["rho_fb"] = _safe_ratio_host(sum_term, sum_xi2)
    result["private_v_decomposition_max_abs"] = max(
        float(row["private_v_decomposition_max_abs"]) for row in rows
    )
    error_sq = np.asarray([
        float(row["private_v_decomposition_rms"]) ** 2 for row in rows
    ])
    result["private_v_decomposition_rms"] = float(
        np.sqrt(np.sum(weights * error_sq) / total)
    )
  else:
    for field in METRIC_FIELDS + ERROR_FIELDS:
      result[field] = 0.0
  return result


def stage_metrics_from_step_rows(
    rows: Iterable[Mapping[str, object]], horizon: int
) -> list[dict[str, object]]:
  """Aggregate raw step scalars into exact valid stage rows."""
  rows = list(rows)
  result = []
  seed = int(rows[0]["seed"]) if rows else 0
  for stage, (start, end) in stage_bounds(horizon).items():
    for branch in BRANCHES:
      selected = [
          row for row in rows
          if row.get("branch") == branch and start <= int(row["step"]) <= end
      ]
      result.append(_stage_row(
          selected, seed=seed, branch=branch, stage=stage,
          start=start, end=end,
      ))
  return result


def paired_stage_metrics_from_stage_rows(
    rows: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
  """Compute MF-vs-IID deltas from paired rows of the same seed and stage."""
  grouped: dict[tuple[int, str, str], Mapping[str, object]] = {}
  for row in rows:
    grouped[(int(row["seed"]), str(row["stage"]), str(row["branch"]))] = row
  stage_order = {"early": 0, "late": 1, "full": 2}
  keys = sorted(
      {(seed, stage) for seed, stage, _ in grouped},
      key=lambda item: (item[0], stage_order.get(item[1], 99), item[1]),
  )
  result = []
  for seed, stage in keys:
    mf = grouped.get((seed, stage, "mf"))
    iid = grouped.get((seed, stage, "iid"))
    if mf is None or iid is None:
      continue
    mean_g2_mf, mean_g2_iid = float(mf["mean_g2"]), float(iid["mean_g2"])
    mean_feedback_mf = float(mf.get(
        "mean_2gxi", float(mf["mean_g2_cross"]) - mean_g2_mf
    ))
    mean_feedback_iid = float(iid.get(
        "mean_2gxi", float(iid["mean_g2_cross"]) - mean_g2_iid
    ))
    mean_xi2_mf, mean_xi2_iid = float(mf["mean_xi2"]), float(iid["mean_xi2"])
    total_mf = float(mf["mean_g2_cross"]) + mean_xi2_mf
    total_iid = float(iid["mean_g2_cross"]) + mean_xi2_iid
    delta_traj = mean_g2_mf - mean_g2_iid
    delta_feedback = mean_feedback_mf - mean_feedback_iid
    delta_noise = mean_xi2_mf - mean_xi2_iid
    delta_total = total_mf - total_iid
    result.append({
        "seed": seed,
        "stage": stage,
        "start_step": int(mf["start_step"]),
        "end_step": int(mf["end_step"]),
        "num_steps": int(mf["num_steps"]),
        "mean_g2_mf": mean_g2_mf,
        "mean_g2_iid": mean_g2_iid,
        "mean_2gxi_mf": mean_feedback_mf,
        "mean_2gxi_iid": mean_feedback_iid,
        "mean_xi2_mf": mean_xi2_mf,
        "mean_xi2_iid": mean_xi2_iid,
        "mean_private_second_moment_mf": total_mf,
        "mean_private_second_moment_iid": total_iid,
        "delta_traj": delta_traj,
        "delta_feedback": delta_feedback,
        "delta_noise": delta_noise,
        "delta_total": delta_total,
        "delta_decomposition_residual": (
            delta_total - delta_traj - delta_feedback - delta_noise
        ),
    })
  return result


def _confidence_stats(values: Iterable[float]) -> dict[str, float | int]:
  values = np.asarray(list(values), dtype=np.float64)
  n = int(values.size)
  if n == 0:
    return {
        "n": 0, "mean": 0.0, "std": 0.0, "se": 0.0,
        "ci95_low": 0.0, "ci95_high": 0.0,
    }
  mean = float(np.mean(values))
  std = float(np.std(values, ddof=1)) if n > 1 else 0.0
  se = std / np.sqrt(n)
  critical = 1.96
  if n >= 2:
    try:
      from scipy.stats import t as student_t
      critical = float(student_t.ppf(.975, n - 1))
    except (ImportError, ValueError):
      critical = 1.96
  margin = critical * se
  return {
      "n": n, "mean": mean, "std": std, "se": float(se),
      "ci95_low": mean - margin, "ci95_high": mean + margin,
  }


def aggregate_paired_stage_rows(
    rows: Iterable[Mapping[str, object]]
) -> dict[str, dict[str, object]]:
  """Cross-seed mean/std/SE/95% CI for every paired stage delta."""
  grouped: dict[str, list[Mapping[str, object]]] = {}
  for row in rows:
    grouped.setdefault(str(row["stage"]), []).append(row)
  stage_order = {"early": 0, "late": 1, "full": 2}
  output = {}
  for stage in sorted(grouped, key=lambda value: stage_order.get(value, 99)):
    stage_rows = grouped[stage]
    output[stage] = {
        "num_seeds": len(stage_rows),
        **{
            field: _confidence_stats(float(row[field]) for row in stage_rows)
            for field in PAIRED_FIELDS
        },
    }
  return output


def aggregate_stage_field(
    rows: Iterable[Mapping[str, object]], *, stage: str, branch: str, field: str
) -> dict[str, float | int]:
  """Cross-seed confidence summary for one branch/stage scalar."""
  selected = [
      float(row[field]) for row in rows
      if str(row["stage"]) == stage and str(row["branch"]) == branch
  ]
  return _confidence_stats(selected)


def write_step_metrics(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  _write_csv(path, STEP_FIELDS, rows)


def write_stage_metrics(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  _write_csv(path, STAGE_FIELDS, rows)


def write_paired_stage_metrics(
    path: str | Path, rows: Iterable[Mapping[str, object]]
) -> None:
  fields = (
      "seed", "stage", "start_step", "end_step", "num_steps",
      "mean_g2_mf", "mean_g2_iid", "mean_2gxi_mf", "mean_2gxi_iid",
      "mean_xi2_mf", "mean_xi2_iid",
      "mean_private_second_moment_mf", "mean_private_second_moment_iid",
  ) + PAIRED_FIELDS
  _write_csv(path, fields, rows)


def _write_csv(
    path: str | Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]
) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)


def _stack_histogram_records(
    records: Iterable[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  records = list(records)
  if not records:
    raise ValueError("at least one histogram record is required")
  records = sorted(records, key=lambda record: (int(record["seed"]), int(record["step"])))
  seeds = np.asarray([int(record["seed"]) for record in records], dtype=np.int32)
  steps = np.asarray([int(record["step"]) for record in records], dtype=np.int32)
  edges = np.stack([np.asarray(record["group_bin_edges"], dtype=np.float64) for record in records])
  counts = np.stack([np.asarray(record["counts"], dtype=np.int64) for record in records])
  relative = np.stack([
      np.asarray(record["relative_frequency"], dtype=np.float32)
      for record in records
  ])
  return seeds, steps, edges, counts, relative


def _save_histogram_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    bins: int,
    pooled: bool,
    source_seeds: Iterable[int] | None = None,
) -> None:
  records = list(records)
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if records:
    seeds, steps, edges, counts, relative = _stack_histogram_records(records)
    names = np.asarray(records[0]["group_component_names"])
  else:
    seeds = np.empty((0,), dtype=np.int32)
    steps = np.empty((0,), dtype=np.int32)
    edges = np.empty((0, len(HISTOGRAM_GROUPS), bins + 1), dtype=np.float64)
    counts = np.empty(
        (0, len(BRANCHES), len(HISTOGRAM_GROUPS), HISTOGRAM_COMPONENT_SLOTS, bins),
        dtype=np.int64,
    )
    relative = np.empty_like(counts, dtype=np.float32)
    names = np.asarray([
        [HISTOGRAM_GROUP_COMPONENTS[group][slot]
         if slot < len(HISTOGRAM_GROUP_COMPONENTS[group]) else ""
         for slot in range(HISTOGRAM_COMPONENT_SLOTS)]
        for group in HISTOGRAM_GROUPS
    ], dtype="U32")
  kwargs = {
      "steps": steps,
      "group_bin_edges": edges,
      "counts": counts,
      "relative_frequency": relative,
      "branch_names": np.asarray(BRANCHES),
      "group_names": np.asarray(HISTOGRAM_GROUPS),
      "group_component_names": names,
      "format_version": np.asarray("exp10-histograms-v2"),
      "format_kind": np.asarray("pooled" if pooled else "per_seed"),
  }
  if pooled:
    kwargs["source_seeds"] = np.asarray(
        sorted(set(int(seed) for seed in (source_seeds or []))), dtype=np.int32
    )
  else:
    kwargs["seeds"] = seeds
  np.savez_compressed(path, **kwargs)


def save_histograms(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    histogram_bins: int = 64,
) -> None:
  """Save optional per-seed grouped histograms without raw coordinate arrays."""
  _save_histogram_records(
      path, records, bins=histogram_bins, pooled=False
  )


def save_pooled_histograms(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    histogram_bins: int = 64,
    source_seeds: Iterable[int] | None = None,
) -> None:
  """Save cross-seed pooled counts (sum first, normalize second)."""
  _save_histogram_records(
      path, records, bins=histogram_bins, pooled=True, source_seeds=source_seeds
  )


class Exp10Collector:
  """Collect scalar rows and, optionally, per-seed grouped histograms."""

  def __init__(
      self,
      params: Any,
      *,
      seed: int,
      horizon: int,
      histogram_bins: int = 64,
      window_size: int = WINDOW_SIZE,
      collect_histograms: bool = True,
      histogram_edges: Mapping[int, Mapping[str, np.ndarray]] | None = None,
  ) -> None:
    if horizon < 1 or histogram_bins < 1 or window_size < 1:
      raise ValueError("horizon, histogram_bins, and window_size must be positive")
    self.seed = int(seed)
    self.horizon = int(horizon)
    self.histogram_bins = int(histogram_bins)
    self.window_size = int(window_size)
    self._histogram_steps = set(histogram_checkpoint_steps(horizon, window_size))
    self._collect_histograms = bool(collect_histograms)
    self._histogram_edges = histogram_edges
    self._rows: list[dict[str, object]] = []
    self._histograms: list[dict[str, Any]] = []
    del params

  @property
  def rows(self) -> list[dict[str, object]]:
    return list(self._rows)

  @property
  def histogram_records(self) -> list[dict[str, Any]]:
    return list(self._histograms)

  def after_step(self, state: Exp10TrainState, step: int) -> None:
    if step != int(np.asarray(state.step)):
      raise ValueError("callback step must equal Exp10 train state step")
    last_step = state.last_step
    for branch in BRANCHES:
      metrics = last_step.metrics[branch]
      row: dict[str, object] = {
          "seed": self.seed,
          "step": int(step),
          "branch": branch,
          "phi_t": _host_scalar(last_step.phi_t),
          "num_coordinates": int(round(_host_scalar(metrics["num_coordinates"]))),
      }
      for field in METRIC_FIELDS:
        if field == "mean_private_second_moment":
          row[field] = (
              _host_scalar(metrics["mean_g2_cross"])
              + _host_scalar(metrics["mean_xi2"])
          )
        else:
          row[field] = _host_scalar(metrics[field])
      row["private_v_decomposition_max_abs"] = _host_scalar(
          last_step.decomposition_error_max_abs[branch]
      )
      row["private_v_decomposition_rms"] = _host_scalar(
          last_step.decomposition_error_rms[branch]
      )
      self._rows.append(row)
    if self._collect_histograms and int(step) in self._histogram_steps:
      edges = None if self._histogram_edges is None else self._histogram_edges.get(int(step))
      self._histograms.append(_histogram_record(
          seed=self.seed,
          step=int(step),
          last_step=last_step,
          bins=self.histogram_bins,
          group_edges=edges,
      ))

  def stage_rows(self) -> list[dict[str, object]]:
    return stage_metrics_from_step_rows(self._rows, self.horizon)


def aggregate_stage_rows(
    rows: Iterable[Mapping[str, object]]
) -> dict[str, dict[str, dict[str, float | int]]]:
  """Return cross-seed mean/std fields for the stage summary JSON."""
  grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
  for row in rows:
    grouped.setdefault((str(row["stage"]), str(row["branch"])), []).append(row)
  output: dict[str, dict[str, dict[str, float | int]]] = {}
  stage_order = {"early": 0, "late": 1, "full": 2}
  numeric_fields = (
      "num_steps", "num_coordinate_observations", "mean_phi_t",
  ) + METRIC_FIELDS + ERROR_FIELDS
  for (stage, branch), group in sorted(
      grouped.items(), key=lambda item: (stage_order.get(item[0][0], 99), item[0][1])
  ):
    stage_output = output.setdefault(stage, {})
    values: dict[str, float | int] = {
        "num_seeds": len(group),
        "start_step": int(group[0]["start_step"]),
        "end_step": int(group[0]["end_step"]),
    }
    for field in numeric_fields:
      array = np.asarray([float(row[field]) for row in group], dtype=np.float64)
      values[f"{field}_mean"] = float(np.mean(array)) if len(array) else 0.0
      values[f"{field}_std"] = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    stage_output[branch] = values
  return output


__all__ = [
    "ERROR_FIELDS",
    "Exp10Collector",
    "HISTOGRAM_GROUPS",
    "HISTOGRAM_GROUP_COMPONENTS",
    "METRIC_FIELDS",
    "PAIRED_FIELDS",
    "PooledHistogramBuilder",
    "STAGE_FIELDS",
    "STEP_FIELDS",
    "WINDOW_SIZE",
    "aggregate_paired_stage_rows",
    "aggregate_stage_field",
    "aggregate_stage_rows",
    "histogram_checkpoint_steps",
    "histogram_extrema",
    "paired_stage_metrics_from_stage_rows",
    "save_histograms",
    "save_pooled_histograms",
    "stage_bounds",
    "stage_metrics_from_step_rows",
    "write_paired_stage_metrics",
    "write_stage_metrics",
    "write_step_metrics",
]
