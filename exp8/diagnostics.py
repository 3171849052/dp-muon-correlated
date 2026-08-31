"""Pure cancellation metrics, stage aggregation, and CSV rows for Exp8."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from exp8.core import BRANCHES, PATHS


METRIC_FIELDS = ("J", "D", "C", "G_C", "G_J")
DECOMP_FIELDS = (
    "A_energy", "B_energy", "I_energy", "AB_dot", "AI_dot", "BI_dot",
    "reconstruction_error",
)


def safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-30) -> float:
  """Return a finite zero when a ratio's denominator is unresolved."""
  numerator, denominator = float(numerator), float(denominator)
  if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= eps:
    return 0.0
  value = numerator / denominator
  return float(value) if np.isfinite(value) else 0.0


def safe_gain(corr: float, iid: float, *, eps: float = 1e-30) -> float:
  """Compute ``1 - corr / iid`` with safe zero-denominator handling."""
  if not np.isfinite(corr) or not np.isfinite(iid) or abs(float(iid)) <= eps:
    return 0.0
  value = 1.0 - float(corr) / float(iid)
  return float(value) if np.isfinite(value) else 0.0


def cancellation_metrics_from_jd(j: float, d: float) -> dict[str, float]:
  """Build ``J,D,C`` from an endpoint residual and matched energy."""
  j, d = float(j), float(d)
  return {"J": j if np.isfinite(j) else 0.0,
          "D": d if np.isfinite(d) else 0.0,
          "C": safe_ratio(j, d)}


def cancellation_statistics(
    x: np.ndarray, *, weight_decay: float, learning_rate: float
) -> dict[str, float]:
  """Compute ``J,D,C`` for one vector-valued step sequence.

  ``x`` is the already-scaled update contribution ``-learning_rate * N_t``;
  accepting it directly makes the function useful for hand-checkable tests.
  """
  values = np.asarray(x, dtype=np.float64)
  if values.ndim < 1:
    raise ValueError("x must have a time dimension")
  a = 1.0 - float(learning_rate) * float(weight_decay)
  d = np.zeros(values.shape[1:], dtype=np.float64)
  denominator = 0.0
  for value in values:
    d = a * d + value
    denominator = a * a * denominator + float(np.sum(value * value))
  endpoint = float(np.sum(d * d))
  return {"J": endpoint, "D": denominator, "C": safe_ratio(endpoint, denominator)}


def paired_gains(
    corr: Mapping[str, float], iid: Mapping[str, float]
) -> dict[str, float]:
  return {
      "G_C": safe_gain(float(corr.get("C", 0.0)), float(iid.get("C", 0.0))),
      "G_J": safe_gain(float(corr.get("J", 0.0)), float(iid.get("J", 0.0))),
  }


def add_paired_gains(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> dict[str, dict[str, dict[str, float]]]:
  """Add ``G_C`` and ``G_J`` to a corr/iid metric tree."""
  result = {
      branch: {
          path: {field: float(value) for field, value in values.items()}
          for path, values in paths.items()
      } for branch, paths in metrics.items()
  }
  for path in PATHS:
    gains = paired_gains(result["corr"][path], result["iid"][path])
    result["corr"][path].update(gains)
    result["iid"][path].update(gains)
  return result


def degradation(gains: Mapping[str, float]) -> dict[str, float]:
  """Return the three adjacent losses in ``G0 -> Gc -> Gphi -> Gp``."""
  return {
      "delta_clean": float(gains["G0"] - gains["Gc"]),
      "delta_bias": float(gains["Gc"] - gains["Gphi"]),
      "delta_nonlinear": float(gains["Gphi"] - gains["Gp"]),
  }


def make_window_row(
    *,
    seed: int,
    window_index: int,
    start_step: int,
    end_step: int,
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    decomp: Mapping[str, float],
) -> dict[str, int | float]:
  """Flatten one exact, non-overlapping window into a CSV row."""
  metrics = add_paired_gains(metrics)
  row: dict[str, int | float] = {
      "seed": int(seed), "window_index": int(window_index),
      "start_step": int(start_step), "end_step": int(end_step),
  }
  for branch in BRANCHES:
    for path in PATHS:
      for field in METRIC_FIELDS:
        row[f"{field}_{branch}_{path}"] = float(metrics[branch][path].get(field, 0.0))
  for field in DECOMP_FIELDS:
    row[field] = float(decomp.get(field, 0.0))
  return row


def write_window_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  base = ["seed", "window_index", "start_step", "end_step"]
  fields = base + [
      f"{field}_{branch}_{path_name}"
      for branch in BRANCHES for path_name in PATHS for field in METRIC_FIELDS
  ] + list(DECOMP_FIELDS)
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
  values = np.asarray(values, dtype=np.float64)
  if not len(values):
    return 0.0, 0.0
  return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate_window_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, int | float]]:
  """Aggregate same-index windows across seeds, without using it for stages."""
  groups: dict[int, list[Mapping[str, object]]] = {}
  for row in rows:
    groups.setdefault(int(row["window_index"]), []).append(row)
  fields = [
      f"{field}_{branch}_{path_name}"
      for branch in BRANCHES for path_name in PATHS for field in METRIC_FIELDS
  ] + list(DECOMP_FIELDS)
  output = []
  for index, group in sorted(groups.items()):
    result: dict[str, int | float] = {
        "window_index": index,
        "start_step": int(group[0]["start_step"]),
        "end_step": int(group[0]["end_step"]),
        "num_seeds": len(group),
    }
    for field in fields:
      mean, std = _mean_std([float(row[field]) for row in group])
      result[f"{field}_mean"], result[f"{field}_std"] = mean, std
    output.append(result)
  return output


def write_window_summary(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  fields = ["window_index", "start_step", "end_step", "num_seeds"]
  metrics = [
      f"{field}_{branch}_{path_name}"
      for branch in BRANCHES for path_name in PATHS for field in METRIC_FIELDS
  ] + list(DECOMP_FIELDS)
  fields += [item for field in metrics for item in (f"{field}_mean", f"{field}_std")]
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def attach_path_degradation(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> dict[str, object]:
  """Build the JSON-friendly stage payload, including both gain types."""
  metrics = add_paired_gains(metrics)
  result: dict[str, object] = {
      "metrics": {
          branch: {path: dict(values) for path, values in paths.items()}
          for branch, paths in metrics.items()
      }
  }
  degradation_output: dict[str, dict[str, float]] = {}
  for gain_field, output_name in (("G_C", "G_C"), ("G_J", "G_J")):
    gains = {path: float(metrics["corr"][path][gain_field]) for path in PATHS}
    degradation_output[output_name] = degradation({
        "G0": gains["P0"], "Gc": gains["P1"],
        "Gphi": gains["P2"], "Gp": gains["P3"],
    })
  result["degradation"] = degradation_output
  result["G0_to_Gc_to_Gphi_to_Gp"] = {
      gain: {path: float(metrics["corr"][path][gain]) for path in PATHS}
      for gain in ("G_C", "G_J")
  }
  return result


__all__ = [
    "DECOMP_FIELDS", "METRIC_FIELDS", "add_paired_gains", "attach_path_degradation",
    "cancellation_statistics", "cancellation_metrics_from_jd", "degradation",
    "make_window_row", "paired_gains", "safe_gain", "safe_ratio",
    "write_window_rows", "aggregate_window_rows", "write_window_summary",
]
