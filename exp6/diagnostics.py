"""Pure aggregation helpers for Experiment 6's window diagnostics.

The online part lives in :mod:`exp6.online_shadow`.  This module deliberately
contains no model, optimizer, or privacy code so that the cancellation
definition can be tested independently of a training run.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


WINDOW_FIELDS = [
    "seed",
    "window_index",
    "start_step",
    "end_step",
    "mean_p_relative_change",
    "C_momentum",
    "C_frozen_p",
    "C_dynamic_clean_p",
    "C_real_adamw",
    "delta_p_cancellation",
    "extra_real_effect",
]

WINDOW_METRICS = [
    "mean_p_relative_change",
    "C_momentum",
    "C_frozen_p",
    "C_dynamic_clean_p",
    "C_real_adamw",
    "delta_p_cancellation",
    "extra_real_effect",
]


def window_ranges(horizon: int, window_size: int = 16) -> list[tuple[int, int, int]]:
  """Return ``(zero_based_index, one_based_start, one_based_end)`` windows."""
  if horizon < 1 or window_size < 1:
    raise ValueError("horizon and window_size must be positive")
  return [
      (index, start, min(start + window_size - 1, horizon))
      for index, start in enumerate(range(1, horizon + 1, window_size))
  ]


def cancellation_score(
    updates: Iterable[np.ndarray],
    *,
    learning_rate: float,
    weight_decay: float,
    eps_num: float = 1e-30,
) -> float:
  """Compute the local weighted cancellation score for one path.

  The recurrence ``y_t = a y_{t-1} + x_t`` is exactly the weighted sum in the
  experiment definition, and ``d_t = a^2 d_{t-1} + ||x_t||^2`` is its matching
  denominator.  Inputs may be scalars, vectors, or arbitrary array-shaped
  updates; all updates must have the same shape.
  """
  if learning_rate <= 0 or weight_decay < 0 or eps_num <= 0:
    raise ValueError("learning_rate must be positive, weight_decay non-negative, and eps_num positive")
  values = [np.asarray(value, dtype=np.float64) for value in updates]
  if not values:
    raise ValueError("updates must contain at least one value")
  shape = values[0].shape
  if any(value.shape != shape for value in values):
    raise ValueError("all updates must have the same shape")
  a = 1.0 - float(learning_rate) * float(weight_decay)
  weighted = np.zeros_like(values[0], dtype=np.float64)
  denominator = 0.0
  for value in values:
    weighted = a * weighted + value
    denominator = a * a * denominator + float(np.sum(value * value))
  return float(np.sum(weighted * weighted) / (denominator + eps_num))


def relative_p_change(
    current: np.ndarray, previous: np.ndarray | None, *, eps_num: float = 1e-30
) -> float:
  """Return the specified relative L2 change, with zero for the first step."""
  if eps_num <= 0:
    raise ValueError("eps_num must be positive")
  if previous is None:
    return 0.0
  current_array = np.asarray(current, dtype=np.float64)
  previous_array = np.asarray(previous, dtype=np.float64)
  if current_array.shape != previous_array.shape:
    raise ValueError("current and previous p must have the same shape")
  return float(
      np.linalg.norm(current_array - previous_array)
      / (np.linalg.norm(previous_array) + eps_num)
  )


def window_row(
    *,
    seed: int,
    window_index: int,
    start_step: int,
    end_step: int,
    mean_p_relative_change: float,
    C_momentum: float,
    C_frozen_p: float,
    C_dynamic_clean_p: float,
    C_real_adamw: float,
) -> dict[str, float | int]:
  """Build one output row and derive the two requested differences."""
  row: dict[str, float | int] = {
      "seed": int(seed),
      "window_index": int(window_index),
      "start_step": int(start_step),
      "end_step": int(end_step),
      "mean_p_relative_change": float(mean_p_relative_change),
      "C_momentum": float(C_momentum),
      "C_frozen_p": float(C_frozen_p),
      "C_dynamic_clean_p": float(C_dynamic_clean_p),
      "C_real_adamw": float(C_real_adamw),
  }
  row["delta_p_cancellation"] = float(row["C_dynamic_clean_p"] - row["C_frozen_p"])
  row["extra_real_effect"] = float(row["C_real_adamw"] - row["C_dynamic_clean_p"])
  return row


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
  array = np.asarray(values, dtype=np.float64)
  if array.size == 0:
    raise ValueError("cannot aggregate an empty group")
  return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def aggregate_window_rows(rows: Iterable[Mapping[str, float | int]]) -> list[dict[str, float | int]]:
  """Aggregate per-seed rows by window position using sample std deviation."""
  groups: dict[int, list[Mapping[str, float | int]]] = {}
  for row in rows:
    index = int(row["window_index"])
    groups.setdefault(index, []).append(row)
  result: list[dict[str, float | int]] = []
  for index in sorted(groups):
    group = groups[index]
    starts = {int(row["start_step"]) for row in group}
    ends = {int(row["end_step"]) for row in group}
    if len(starts) != 1 or len(ends) != 1:
      raise ValueError(f"window {index} has inconsistent boundaries")
    aggregate: dict[str, float | int] = {
        "window_index": index,
        "start_step": starts.pop(),
        "end_step": ends.pop(),
        "num_seeds": len(group),
    }
    for field in WINDOW_METRICS:
      mean, std = _mean_std([float(row[field]) for row in group])
      aggregate[f"{field}_mean"] = mean
      aggregate[f"{field}_std"] = std
    result.append(aggregate)
  return result


def _overlap(start: int, end: int, stage_start: int, stage_end: int) -> int:
  return max(0, min(end, stage_end) - max(start, stage_start) + 1)


def _weighted_stage_mean(
    rows: Sequence[Mapping[str, float | int]], *, stage_start: int, stage_end: int,
    field: str,
) -> tuple[float, int]:
  weighted_sum = 0.0
  total_weight = 0
  for row in rows:
    weight = _overlap(
        int(row["start_step"]), int(row["end_step"]), stage_start, stage_end
    )
    weighted_sum += weight * float(row[field])
    total_weight += weight
  return (
      float(weighted_sum / total_weight) if total_weight else float("nan"),
      total_weight,
  )


def stage_summary(
    rows: Iterable[Mapping[str, float | int]], *, early_end: int = 97,
    total_steps: int | None = None,
) -> dict[str, dict[str, float | int]]:
  """Compute strict step-weighted early/late means across all seed rows.

  A window crossing step 97 contributes only its actual number of steps to
  each stage.  Thus the [97, 112] window is not silently assigned to one side.
  """
  materialized = list(rows)
  if early_end < 1:
    raise ValueError("early_end must be positive")
  inferred_end = max((int(row["end_step"]) for row in materialized), default=early_end)
  final_step = inferred_end if total_steps is None else int(total_steps)
  if final_step < early_end:
    late_start = final_step + 1
  else:
    late_start = early_end + 1
  early_mean, early_weight = _weighted_stage_mean(
      materialized, stage_start=1, stage_end=early_end, field="delta_p_cancellation"
  )
  late_mean, late_weight = _weighted_stage_mean(
      materialized, stage_start=late_start, stage_end=final_step,
      field="delta_p_cancellation",
  ) if late_start <= final_step else (float("nan"), 0)
  return {
      "early_steps_1_97": {
          "start_step": 1, "end_step": early_end,
          "mean_delta_p_cancellation": early_mean,
          "covered_steps": early_weight,
      },
      "late_steps_98_488": {
          "start_step": 98, "end_step": final_step,
          "mean_delta_p_cancellation": late_mean,
          "covered_steps": late_weight,
      },
  }


def per_seed_stage_summary(
    rows: Iterable[Mapping[str, float | int]], *, early_end: int = 97,
    total_steps: int | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
  """Return the same strict stage means separately for each seed."""
  materialized = list(rows)
  seeds = sorted({int(row["seed"]) for row in materialized})
  return {
      str(seed): stage_summary(
          [row for row in materialized if int(row["seed"]) == seed],
          early_end=early_end, total_steps=total_steps,
      )
      for seed in seeds
  }


def spearman_rank_correlation(
    x: Iterable[float], y: Iterable[float]
) -> float:
  """Compute Spearman's rho with average ranks for ties.

  A constant input has no defined rank correlation and returns ``nan``.  The
  value is kept as a float so callers can serialize it in the summary JSON.
  """
  x_array, y_array = np.asarray(list(x), dtype=np.float64), np.asarray(list(y), dtype=np.float64)
  if x_array.shape != y_array.shape:
    raise ValueError("correlation inputs must have the same length")
  finite = np.isfinite(x_array) & np.isfinite(y_array)
  x_array, y_array = x_array[finite], y_array[finite]
  if x_array.size < 2:
    return float("nan")

  def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    rank_values = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
      end = position + 1
      while end < values.size and sorted_values[end] == sorted_values[position]:
        end += 1
      rank_values[order[position:end]] = (position + 1 + end) / 2.0
      position = end
    return rank_values

  x_rank, y_rank = ranks(x_array), ranks(y_array)
  if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
    return float("nan")
  return float(np.corrcoef(x_rank, y_rank)[0, 1])


def correlation_from_rows(rows: Iterable[Mapping[str, float | int]]) -> float:
  materialized = list(rows)
  return spearman_rank_correlation(
      [float(row["mean_p_relative_change"]) for row in materialized],
      [float(row["delta_p_cancellation"]) for row in materialized],
  )


def write_window_rows(path: str | Path, rows: Iterable[Mapping[str, float | int]]) -> None:
  """Write one seed's required window-level schema in a stable column order."""
  rows = list(rows)
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=WINDOW_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def write_window_summary(path: str | Path, rows: Iterable[Mapping[str, float | int]]) -> None:
  fields = ["window_index", "start_step", "end_step", "num_seeds"]
  fields.extend(field for metric in WINDOW_METRICS for field in (f"{metric}_mean", f"{metric}_std"))
  rows = list(rows)
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


__all__ = [
    "WINDOW_FIELDS",
    "WINDOW_METRICS",
    "aggregate_window_rows",
    "cancellation_score",
    "correlation_from_rows",
    "per_seed_stage_summary",
    "relative_p_change",
    "spearman_rank_correlation",
    "stage_summary",
    "window_ranges",
    "window_row",
    "write_window_rows",
    "write_window_summary",
]
