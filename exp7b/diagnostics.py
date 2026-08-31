"""CSV rows and early/late summaries for Experiment 7b."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from exp7.diagnostics import PATHS, factorial_effects, restoration_ratio


NORM_NAMES = (
    "raw_optimizer_update_l2", "applied_parameter_update_l2", "parameter_l2"
)
P_QUANTILE_METRICS = ("p_bc_median", "p_bc_q99", "p_bc_q99_9")
STABILITY_METRICS = (
    "corrected_v_nonpositive_fraction",
    "floor_activation_fraction",
    *P_QUANTILE_METRICS,
    "p_bc_max",
    *(f"{name}_{stat}" for name in NORM_NAMES
      for stat in ("mean", "std", "min", "max")),
)
BASE_METRICS = tuple(f"C_{path}" for path in PATHS) + (
    "C_real", "C_dynamic_clean_p", "gap",
    "E_cross", "E_square", "interaction", "R_BC",
) + tuple(f"nonpositive_v_fraction_{path}" for path in PATHS) + STABILITY_METRICS
WINDOW_FIELDS = (
    "seed", "algorithm", "window_index", "start_step", "end_step",
    "mean_p_relative_change", *BASE_METRICS,
)


def window_row(
    *, seed: int, algorithm: str, window_index: int, start_step: int,
    end_step: int, mean_p_relative_change: float,
    scores: Mapping[str, float], negative_fractions: Mapping[str, float],
    stability: Mapping[str, float],
) -> dict[str, float | int | str]:
  if algorithm not in {"baseline", "bc"}:
    raise ValueError("algorithm must be baseline or bc")
  row: dict[str, float | int | str] = {
      "seed": int(seed), "algorithm": algorithm,
      "window_index": int(window_index), "start_step": int(start_step),
      "end_step": int(end_step),
      "mean_p_relative_change": float(mean_p_relative_change),
  }
  for path in PATHS:
    row[f"C_{path}"] = float(scores[path])
    row[f"nonpositive_v_fraction_{path}"] = float(negative_fractions[path])
  row["C_dynamic_clean_p"] = float(scores["00"])
  row["C_real"] = float(scores["11"] if algorithm == "baseline" else scores["BC"])
  row["gap"] = float(row["C_real"]) - float(row["C_dynamic_clean_p"])
  row.update(factorial_effects(
      float(scores["00"]), float(scores["10"]),
      float(scores["01"]), float(scores["11"]),
  ))
  row["R_BC"] = restoration_ratio(scores["00"], scores["11"], scores["BC"])
  for metric in STABILITY_METRICS:
    row[metric] = float(stability[metric])
  return row


def write_window_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=WINDOW_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
  array = np.asarray(values, np.float64)
  return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate_window_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, float | int | str]]:
  groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
  for row in rows:
    groups.setdefault((str(row["algorithm"]), int(row["window_index"])), []).append(row)
  output: list[dict[str, float | int | str]] = []
  for (algorithm, index), group in sorted(groups.items()):
    result: dict[str, float | int | str] = {
        "algorithm": algorithm, "window_index": index,
        "start_step": int(group[0]["start_step"]),
        "end_step": int(group[0]["end_step"]), "num_seeds": len(group),
    }
    if any(int(row["start_step"]) != result["start_step"] or
           int(row["end_step"]) != result["end_step"] for row in group):
      raise ValueError(f"inconsistent boundaries for {algorithm} window {index}")
    excluded = {
        *P_QUANTILE_METRICS, "p_bc_max",
        *(f"{name}_{stat}" for name in NORM_NAMES
          for stat in ("mean", "std", "min", "max")),
    }
    for metric in (
        metric for metric in ("mean_p_relative_change", *BASE_METRICS)
        if metric not in excluded
    ):
      mean, std = _mean_std([float(row[metric]) for row in group])
      result[f"{metric}_mean"], result[f"{metric}_std"] = mean, std
    # A cross-seed average of window quantiles is not a pooled quantile.  Keep
    # it available for plots, but name the statistic honestly.
    for metric in P_QUANTILE_METRICS:
      mean, std = _mean_std([float(row[metric]) for row in group])
      result[f"mean_window_{metric}"] = mean
      result[f"std_window_{metric}"] = std
    result["p_bc_max"] = max(float(row["p_bc_max"]) for row in group)
    sample_count = int(result["end_step"]) - int(result["start_step"]) + 1
    total_count = sample_count * len(group)
    for name in NORM_NAMES:
      means = np.asarray([float(row[f"{name}_mean"]) for row in group], np.float64)
      stds = np.asarray([float(row[f"{name}_std"]) for row in group], np.float64)
      pooled_mean = float(np.sum(sample_count * means) / total_count)
      pooled_second = float(np.sum(sample_count * (stds ** 2 + means ** 2)) / total_count)
      result[f"{name}_mean"] = pooled_mean
      result[f"{name}_std"] = float(np.sqrt(max(0.0, pooled_second - pooled_mean ** 2)))
      result[f"{name}_min"] = min(float(row[f"{name}_min"]) for row in group)
      result[f"{name}_max"] = max(float(row[f"{name}_max"]) for row in group)
    output.append(result)
  return output


def write_window_summary(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  fields = ["algorithm", "window_index", "start_step", "end_step", "num_seeds"]
  for row in rows:
    fields.extend(field for field in row if field not in fields)
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _overlap(start: int, end: int, stage_start: int, stage_end: int) -> int:
  return max(0, min(end, stage_end) - max(start, stage_start) + 1)


def histogram_quantile(
    histogram: np.ndarray, q: float, implied_p_max: float,
) -> float:
  """Return the same conservative upper-bin quantile used by the collector."""
  counts = np.asarray(histogram, np.int64)
  if counts.ndim != 1 or counts.size < 2 or np.any(counts < 0):
    raise ValueError("histogram must be a non-negative one-dimensional array")
  if not 0 < q <= 1 or not np.isfinite(implied_p_max) or implied_p_max <= 0:
    raise ValueError("q and implied_p_max are invalid")
  total = int(np.sum(counts))
  if total < 1:
    return float("nan")
  target = int(np.ceil(q * total))
  index = int(np.searchsorted(np.cumsum(counts), target, side="left"))
  return float(implied_p_max * (index + 1) / counts.size)


def aggregate_step_stability(
    step_diagnostics: Iterable[Mapping[str, object]], *, stage_start: int,
    stage_end: int, implied_p_max: float,
) -> dict[str, float | int]:
  """Aggregate exact per-step stability records for one stage.

  Norms are one scalar sample per optimizer step.  Histograms retain all
  parameter coordinates, so their sum is the exact pooled binned population.
  """
  selected = [
      row for row in step_diagnostics
      if stage_start <= int(row["step"]) <= stage_end
  ]
  if not selected:
    return {"covered_stability_steps": 0}
  result: dict[str, float | int] = {"covered_stability_steps": len(selected)}
  mean_metrics = (
      *[f"nonpositive_v_fraction_{path}" for path in PATHS],
      "corrected_v_nonpositive_fraction", "floor_activation_fraction",
  )
  for metric in mean_metrics:
    result[metric] = float(np.mean([float(row[metric]) for row in selected]))
  histograms = [np.asarray(row["p_bc_histogram"], np.int64) for row in selected]
  if any(histogram.shape != histograms[0].shape for histogram in histograms):
    raise ValueError("per-step p_bc histograms must have identical shapes")
  pooled_histogram = np.sum(np.stack(histograms), axis=0, dtype=np.int64)
  result["p_bc_median"] = histogram_quantile(pooled_histogram, .5, implied_p_max)
  result["p_bc_q99"] = histogram_quantile(pooled_histogram, .99, implied_p_max)
  result["p_bc_q99_9"] = histogram_quantile(pooled_histogram, .999, implied_p_max)
  result["p_bc_max"] = max(float(row["p_bc_max"]) for row in selected)
  for name in NORM_NAMES:
    values = np.asarray([float(row[name]) for row in selected], np.float64)
    result[f"{name}_mean"] = float(np.mean(values))
    result[f"{name}_std"] = float(np.std(values))
    result[f"{name}_min"] = float(np.min(values))
    result[f"{name}_max"] = float(np.max(values))
  return result


def stage_metrics(
    rows: Iterable[Mapping[str, object]], *, stage_start: int, stage_end: int,
    step_diagnostics: Iterable[Mapping[str, object]] | None = None,
    implied_p_max: float | None = None,
) -> dict[str, float | int]:
  materialized = list(rows)
  result: dict[str, float | int] = {"start_step": stage_start, "end_step": stage_end}
  weights = [_overlap(int(row["start_step"]), int(row["end_step"]), stage_start, stage_end)
             for row in materialized]
  total = sum(weights)
  result["covered_seed_steps"] = total
  direct_metrics = (
      "C_00", "C_10", "C_01", "C_11", "C_BC", "C_real",
      "C_dynamic_clean_p", "gap",
  )
  for metric in direct_metrics:
    result[metric] = (
        float(sum(w * float(row[metric]) for row, w in zip(materialized, weights, strict=True)) / total)
        if total else float("nan")
    )
  result.update(factorial_effects(
      float(result["C_00"]), float(result["C_10"]),
      float(result["C_01"]), float(result["C_11"]),
  ))
  result["R_BC"] = restoration_ratio(
      float(result["C_00"]), float(result["C_11"]), float(result["C_BC"])
  )
  if step_diagnostics is not None:
    if implied_p_max is None:
      raise ValueError("implied_p_max is required with step_diagnostics")
    result.update(aggregate_step_stability(
        step_diagnostics, stage_start=stage_start, stage_end=stage_end,
        implied_p_max=implied_p_max,
    ))
  return result


def two_stage_summary(
    rows: Iterable[Mapping[str, object]], *, total_steps: int,
    step_diagnostics: Iterable[Mapping[str, object]] | None = None,
    implied_p_max: float | None = None,
) -> dict[str, dict[str, float | int]]:
  materialized = list(rows)
  exact_steps = list(step_diagnostics) if step_diagnostics is not None else None
  return {
      "early_steps_1_97": stage_metrics(
          materialized, stage_start=1, stage_end=min(97, total_steps),
          step_diagnostics=exact_steps, implied_p_max=implied_p_max,
      ),
      "late_steps_98_488": (
          stage_metrics(
              materialized, stage_start=98, stage_end=total_steps,
              step_diagnostics=exact_steps, implied_p_max=implied_p_max,
          )
          if total_steps >= 98 else {
              "start_step": 98, "end_step": total_steps, "covered_seed_steps": 0
          }
      ),
  }


__all__ = [
    "BASE_METRICS", "NORM_NAMES", "PATHS", "P_QUANTILE_METRICS",
    "STABILITY_METRICS", "WINDOW_FIELDS", "aggregate_step_stability",
    "aggregate_window_rows", "histogram_quantile", "stage_metrics",
    "two_stage_summary", "window_row", "write_window_rows",
    "write_window_summary",
]
