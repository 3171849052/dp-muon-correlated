"""Pure row construction and aggregation for Experiment 7."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PATHS = ("00", "10", "01", "11", "BC")
BASE_METRICS = tuple(f"C_{path}" for path in PATHS) + (
    "C_real", "C_dynamic_clean_p", "gap",
    "E_cross", "E_square", "interaction", "R_BC",
) + tuple(f"nonpositive_v_fraction_{path}" for path in PATHS)
WINDOW_FIELDS = (
    "seed", "algorithm", "window_index", "start_step", "end_step",
    "mean_p_relative_change", *BASE_METRICS,
)


def factorial_effects(c00: float, c10: float, c01: float, c11: float) -> dict[str, float]:
  return {
      "E_cross": 0.5 * ((c10 - c00) + (c11 - c01)),
      "E_square": 0.5 * ((c01 - c00) + (c11 - c10)),
      "interaction": c11 - c10 - c01 + c00,
  }


def restoration_ratio(
    c00: float, c11: float, c_bc: float, *, denominator_tol: float = 1e-12
) -> float:
  """Return the restoration ratio, or NaN when its denominator is unresolved."""
  denominator = float(c11) - float(c00)
  scale = max(1.0, abs(float(c00)), abs(float(c11)))
  if not np.isfinite(denominator) or abs(denominator) <= denominator_tol * scale:
    return float("nan")
  return float(1.0 - (float(c_bc) - float(c00)) / denominator)


def window_row(
    *, seed: int, algorithm: str, window_index: int, start_step: int,
    end_step: int, mean_p_relative_change: float,
    scores: Mapping[str, float], negative_fractions: Mapping[str, float],
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
        "algorithm": algorithm,
        "window_index": index,
        "start_step": int(group[0]["start_step"]),
        "end_step": int(group[0]["end_step"]),
        "num_seeds": len(group),
    }
    if any(
        int(row["start_step"]) != result["start_step"]
        or int(row["end_step"]) != result["end_step"] for row in group
    ):
      raise ValueError(f"inconsistent boundaries for {algorithm} window {index}")
    for metric in ("mean_p_relative_change", *BASE_METRICS):
      mean, std = _mean_std([float(row[metric]) for row in group])
      result[f"{metric}_mean"] = mean
      result[f"{metric}_std"] = std
    output.append(result)
  return output


def write_window_summary(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  fields = ["algorithm", "window_index", "start_step", "end_step", "num_seeds"]
  fields += [
      field for metric in ("mean_p_relative_change", *BASE_METRICS)
      for field in (f"{metric}_mean", f"{metric}_std")
  ]
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _overlap(start: int, end: int, stage_start: int, stage_end: int) -> int:
  return max(0, min(end, stage_end) - max(start, stage_start) + 1)


def stage_metrics(
    rows: Iterable[Mapping[str, object]], *, stage_start: int, stage_end: int,
) -> dict[str, float | int]:
  """Use Exp6's strict overlap weighting for a requested step range."""
  materialized = list(rows)
  result: dict[str, float | int] = {"start_step": stage_start, "end_step": stage_end}
  weights = [
      _overlap(int(row["start_step"]), int(row["end_step"]), stage_start, stage_end)
      for row in materialized
  ]
  total = sum(weights)
  result["covered_seed_steps"] = total
  for metric in ("C_00", "C_10", "C_01", "C_11", "C_BC", "C_real",
                 "C_dynamic_clean_p", "gap", *[f"nonpositive_v_fraction_{p}" for p in PATHS]):
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
  return result


def two_stage_summary(
    rows: Iterable[Mapping[str, object]], *, total_steps: int,
) -> dict[str, dict[str, float | int]]:
  materialized = list(rows)
  return {
      "early_steps_1_97": stage_metrics(
          materialized, stage_start=1, stage_end=min(97, total_steps)
      ),
      "late_steps_98_488": (
          stage_metrics(materialized, stage_start=98, stage_end=total_steps)
          if total_steps >= 98 else {
              "start_step": 98, "end_step": total_steps, "covered_seed_steps": 0
          }
      ),
  }


__all__ = [
    "BASE_METRICS", "PATHS", "WINDOW_FIELDS", "aggregate_window_rows",
    "factorial_effects", "restoration_ratio", "stage_metrics",
    "two_stage_summary", "window_row", "write_window_rows", "write_window_summary",
]
