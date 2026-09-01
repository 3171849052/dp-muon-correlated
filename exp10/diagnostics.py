"""Online scalar metrics, exact stages, and compact histograms for Exp10."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from exp10.core import (
    BRANCHES,
    COMPONENTS,
    EMA_COMPONENTS,
    Exp10Step,
    Exp10TrainState,
    INSTANTANEOUS_COMPONENTS,
)


WINDOW_SIZE = 16
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
)
ERROR_FIELDS = ("private_v_decomposition_max_abs", "private_v_decomposition_rms")
STEP_FIELDS = (
    "seed", "step", "branch", "phi_t", "num_coordinates",
) + METRIC_FIELDS + ERROR_FIELDS
STAGE_FIELDS = (
    "seed", "stage", "branch", "start_step", "end_step", "num_steps",
    "num_coordinate_observations", "mean_phi_t",
) + METRIC_FIELDS + ERROR_FIELDS
HISTOGRAM_COMPONENTS = COMPONENTS


def histogram_checkpoint_steps(horizon: int, window_size: int = WINDOW_SIZE) -> list[int]:
  """Return 16-step checkpoints plus the final non-aligned step if needed."""
  if horizon < 1 or window_size < 1:
    raise ValueError("horizon and window_size must be positive")
  result = list(range(window_size, horizon + 1, window_size))
  if not result or result[-1] != horizon:
    result.append(horizon)
  return result


def stage_bounds(horizon: int) -> dict[str, tuple[int, int]]:
  if horizon < 1:
    raise ValueError("horizon must be positive")
  return {
      "early": (1, min(97, horizon)),
      "late": (98, horizon),
      "full": (1, horizon),
  }


def _host_scalar(value: Any) -> float:
  result = float(np.asarray(value))
  if not np.isfinite(result):
    raise FloatingPointError(f"non-finite Exp10 diagnostic scalar: {result}")
  return result


def _flatten_tree(tree: Any) -> np.ndarray:
  leaves = [np.asarray(leaf).reshape(-1) for leaf in _tree_leaves(tree)]
  if not leaves:
    raise ValueError("histogram input tree must contain at least one leaf")
  values = np.concatenate(leaves).astype(np.float64, copy=False)
  if not np.all(np.isfinite(values)):
    raise FloatingPointError("histogram input contains non-finite values")
  return values


def _tree_leaves(tree: Any) -> list[Any]:
  # Importing JAX at module import time is unnecessary for CSV-only callers.
  import jax
  return list(jax.tree_util.tree_leaves(tree))


def _histogram_record(
    *, seed: int, step: int, last_step: Exp10Step, bins: int
) -> dict[str, Any]:
  if bins < 1:
    raise ValueError("bins must be positive")
  values: dict[tuple[str, str], np.ndarray] = {}
  all_values = []
  for branch in BRANCHES:
    for component in INSTANTANEOUS_COMPONENTS:
      array = _flatten_tree(last_step.instantaneous[branch][component])
      values[(branch, component)] = array
      all_values.append(array)
    for component in EMA_COMPONENTS:
      array = _flatten_tree(last_step.ema[branch][component])
      values[(branch, component)] = array
      all_values.append(array)
  combined = np.concatenate(all_values)
  low, high = float(np.min(combined)), float(np.max(combined))
  if low == high:
    padding = max(0.5, abs(low) * 0.01)
    low, high = low - padding, high + padding
  edges = np.linspace(low, high, bins + 1, dtype=np.float64)
  counts = np.zeros((len(BRANCHES), len(COMPONENTS), bins), dtype=np.int64)
  for branch_index, branch in enumerate(BRANCHES):
    for component_index, component in enumerate(COMPONENTS):
      counts[branch_index, component_index], _ = np.histogram(
          values[(branch, component)], bins=edges
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
      "bin_edges": edges,
      "counts": counts,
      "relative_frequency": relative,
  }


class Exp10Collector:
  """Collect every scalar step, exact stages, and selected histograms."""

  def __init__(
      self,
      params: Any,
      *,
      seed: int,
      horizon: int,
      histogram_bins: int = 64,
      window_size: int = WINDOW_SIZE,
  ) -> None:
    if horizon < 1 or histogram_bins < 1 or window_size < 1:
      raise ValueError("horizon, histogram_bins, and window_size must be positive")
    self.seed = int(seed)
    self.horizon = int(horizon)
    self.histogram_bins = int(histogram_bins)
    self.window_size = int(window_size)
    self._histogram_steps = set(histogram_checkpoint_steps(horizon, window_size))
    self._rows: list[dict[str, object]] = []
    self._histograms: list[dict[str, Any]] = []
    del params  # shapes are already represented by the first real step.

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
        row[field] = _host_scalar(metrics[field])
      row["private_v_decomposition_max_abs"] = _host_scalar(
          last_step.decomposition_error_max_abs[branch]
      )
      row["private_v_decomposition_rms"] = _host_scalar(
          last_step.decomposition_error_rms[branch]
      )
      self._rows.append(row)
    if int(step) in self._histogram_steps:
      self._histograms.append(_histogram_record(
          seed=self.seed, step=int(step), last_step=last_step,
          bins=self.histogram_bins,
      ))

  def stage_rows(self) -> list[dict[str, object]]:
    return stage_metrics_from_step_rows(self._rows, self.horizon)


def _weighted_mean(rows: list[Mapping[str, object]], field: str) -> float:
  if not rows:
    return 0.0
  weights = np.asarray([float(row["num_coordinates"]) for row in rows])
  values = np.asarray([float(row[field]) for row in rows])
  total = float(np.sum(weights))
  return float(np.sum(values * weights) / total) if total > 0 else 0.0


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
      "mean_g2_cross_minus_g2",
  ):
    result[field] = _weighted_mean(rows, field)
  if rows:
    weights = np.asarray([float(row["num_coordinates"]) for row in rows])
    rms_sq = np.asarray([float(row["rms_2gxi"]) ** 2 for row in rows])
    total = float(np.sum(weights))
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


def _safe_ratio_host(numerator: float, denominator: float) -> float:
  if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-30:
    return 0.0
  return float(numerator / denominator)


def stage_metrics_from_step_rows(
    rows: Iterable[Mapping[str, object]], horizon: int
) -> list[dict[str, object]]:
  """Aggregate raw step scalars into exact early/late/full stage rows."""
  rows = list(rows)
  result = []
  for stage, (start, end) in stage_bounds(horizon).items():
    for branch in BRANCHES:
      selected = [
          row for row in rows
          if row.get("branch") == branch and start <= int(row["step"]) <= end
      ] if start <= end else []
      seed = int(rows[0]["seed"]) if rows else 0
      result.append(_stage_row(
          selected, seed=seed, branch=branch, stage=stage,
          start=start, end=end,
      ))
  return result


def write_step_metrics(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  _write_csv(path, STEP_FIELDS, rows)


def write_stage_metrics(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  _write_csv(path, STAGE_FIELDS, rows)


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


def save_histograms(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    histogram_bins: int = 64,
) -> None:
  """Save compact shared-bin histograms; no coordinate arrays are persisted."""
  records = list(records)
  if records:
    edges = np.stack([np.asarray(record["bin_edges"], dtype=np.float64) for record in records])
    counts = np.stack([np.asarray(record["counts"], dtype=np.int64) for record in records])
    relative = np.stack([
        np.asarray(record["relative_frequency"], dtype=np.float32)
        for record in records
    ])
    seeds = np.asarray([int(record["seed"]) for record in records], dtype=np.int32)
    steps = np.asarray([int(record["step"]) for record in records], dtype=np.int32)
  else:
    edges = np.empty((0, histogram_bins + 1), dtype=np.float64)
    counts = np.empty((0, len(BRANCHES), len(COMPONENTS), histogram_bins), dtype=np.int64)
    relative = np.empty_like(counts, dtype=np.float32)
    seeds = np.empty((0,), dtype=np.int32)
    steps = np.empty((0,), dtype=np.int32)
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      path,
      seeds=seeds,
      steps=steps,
      bin_edges=edges,
      counts=counts,
      relative_frequency=relative,
      branch_names=np.asarray(BRANCHES),
      component_names=np.asarray(COMPONENTS),
      format_version=np.asarray("exp10-histograms-v1"),
  )


def aggregate_stage_rows(
    rows: Iterable[Mapping[str, object]]
) -> dict[str, dict[str, dict[str, float | int]]]:
  """Return cross-seed mean/std fields for the summary JSON."""
  grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
  for row in rows:
    grouped.setdefault((str(row["stage"]), str(row["branch"])), []).append(row)
  output: dict[str, dict[str, dict[str, float | int]]] = {}
  numeric_fields = (
      "num_steps", "num_coordinate_observations", "mean_phi_t",
  ) + METRIC_FIELDS + ERROR_FIELDS
  for (stage, branch), group in sorted(grouped.items()):
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
    "COMPONENTS",
    "ERROR_FIELDS",
    "Exp10Collector",
    "HISTOGRAM_COMPONENTS",
    "METRIC_FIELDS",
    "STAGE_FIELDS",
    "STEP_FIELDS",
    "WINDOW_SIZE",
    "aggregate_stage_rows",
    "histogram_checkpoint_steps",
    "save_histograms",
    "stage_bounds",
    "stage_metrics_from_step_rows",
    "write_stage_metrics",
    "write_step_metrics",
]
