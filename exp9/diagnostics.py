"""Exact endpoint, window, and cross-seed aggregation for Experiment 9."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from exp9.core import BRANCHES, PATHS


METRIC_FIELDS = ("J", "D", "C", "G_C", "G_J")
DECOMP_FIELDS = (
    "state_gap_energy", "odd_gap_energy", "even_gap_energy",
    "odd_reconstruction_error",
)
BIAS_FIELDS = (
    "output_bias_norm_corr", "output_bias_norm_iid",
    "bias_endpoint_error_corr", "bias_endpoint_error_iid",
    "raw_private_clean_gap_endpoint_corr", "raw_private_clean_gap_endpoint_iid",
    "raw_private_clean_response_norm_corr", "raw_private_clean_response_norm_iid",
    "normalization_boundary_margin_min", "noise_signal_ratio_mean_corr",
    "noise_signal_ratio_mean_iid",
)


def safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-30) -> float:
  numerator, denominator = float(numerator), float(denominator)
  if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= eps:
    return 0.0
  value = numerator / denominator
  return float(value) if np.isfinite(value) else 0.0


def safe_gain(corr: float, iid: float, *, eps: float = 1e-30) -> float:
  if not np.isfinite(corr) or not np.isfinite(iid) or abs(float(iid)) <= eps:
    return 0.0
  value = 1.0 - float(corr) / float(iid)
  return float(value) if np.isfinite(value) else 0.0


def cancellation_metrics_from_jd(j: float, d: float) -> dict[str, float]:
  j, d = float(j), float(d)
  return {"J": j if np.isfinite(j) else 0.0,
          "D": d if np.isfinite(d) else 0.0,
          "C": safe_ratio(j, d)}


def cancellation_statistics(
    x: np.ndarray, *, weight_decay: float, learning_rate: float
) -> dict[str, float]:
  """Reference implementation of the Exp8 endpoint semantics."""
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


def paired_gains(corr: Mapping[str, float], iid: Mapping[str, float]) -> dict[str, float]:
  return {"G_C": safe_gain(corr.get("C", 0.0), iid.get("C", 0.0)),
          "G_J": safe_gain(corr.get("J", 0.0), iid.get("J", 0.0))}


def add_paired_gains(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> dict[str, dict[str, dict[str, float]]]:
  result = {
      branch: {path: {field: float(value) for field, value in values.items()}
               for path, values in paths.items()}
      for branch, paths in metrics.items()
  }
  for path in PATHS:
    gains = paired_gains(result["corr"][path], result["iid"][path])
    result["corr"][path].update(gains)
    result["iid"][path].update(gains)
  return result


def degradation(gains: Mapping[str, float]) -> dict[str, float]:
  """Return the requested adjacent losses ``P0 -> P1 -> P2 -> P3``."""
  return {
      "delta_state": float(gains["P0"] - gains["P1"]),
      "delta_odd": float(gains["P1"] - gains["P2"]),
      "delta_even": float(gains["P2"] - gains["P3"]),
  }


def _finite_mapping(values: Mapping[str, object], fields: Sequence[str]) -> dict[str, float]:
  return {field: float(values.get(field, 0.0)) for field in fields}


def make_window_row(
    *, seed: int, window_index: int, start_step: int, end_step: int,
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    decomposition: Mapping[str, float], bias: Mapping[str, float],
) -> dict[str, int | float]:
  metrics = add_paired_gains(metrics)
  row: dict[str, int | float] = {
      "seed": int(seed), "window_index": int(window_index),
      "start_step": int(start_step), "end_step": int(end_step),
  }
  for branch in BRANCHES:
    for path in PATHS:
      for field in METRIC_FIELDS:
        row[f"{field}_{branch}_{path}"] = float(metrics[branch][path].get(field, 0.0))
  row.update(_finite_mapping(decomposition, DECOMP_FIELDS))
  row.update(_finite_mapping(bias, BIAS_FIELDS))
  return row


def _window_fields() -> list[str]:
  return [
      f"{field}_{branch}_{path}"
      for branch in BRANCHES for path in PATHS for field in METRIC_FIELDS
  ] + list(DECOMP_FIELDS) + list(BIAS_FIELDS)


def write_window_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  fields = ["seed", "window_index", "start_step", "end_step"] + _window_fields()
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
  groups: dict[int, list[Mapping[str, object]]] = {}
  for row in rows:
    groups.setdefault(int(row["window_index"]), []).append(row)
  output = []
  for index, group in sorted(groups.items()):
    result: dict[str, int | float] = {
        "window_index": index, "start_step": int(group[0]["start_step"]),
        "end_step": int(group[0]["end_step"]), "num_seeds": len(group),
    }
    for field in _window_fields():
      mean, std = _mean_std([float(row.get(field, 0.0)) for row in group])
      result[f"{field}_mean"], result[f"{field}_std"] = mean, std
    output.append(result)
  return output


def write_window_summary(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  fields = ["window_index", "start_step", "end_step", "num_seeds"]
  fields += [item for field in _window_fields() for item in (f"{field}_mean", f"{field}_std")]
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _aggregate(values: Sequence[float]) -> dict[str, float]:
  mean, std = _mean_std(values)
  return {"mean": mean, "std": std}


def _aggregate_stage_responses(
    stages: Sequence[Mapping[str, object]], field: str
) -> dict[str, dict[str, dict[str, float]]]:
  output: dict[str, dict[str, dict[str, float]]] = {}
  for branch in BRANCHES:
    output[branch] = {}
    for stage_name in ("linear", "bf16", "norm", "ns", "scale"):
      values = [
          float(stage.get(field, {}).get(branch, {}).get(stage_name, 0.0))
          for stage in stages
      ]
      output[branch][stage_name] = _aggregate(values)
  return output


def cross_seed_aggregate(
    per_seed: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> dict[str, object]:
  """Aggregate stage values that were recomputed from raw steps per seed."""
  output: dict[str, object] = {}
  path_fields = ("C_corr", "C_iid", "J_corr", "J_iid", "D_corr", "D_iid", "G_C", "G_J")
  for stage in ("early", "late", "full"):
    stages = [run[stage] for run in per_seed.values() if stage in run]
    if not stages:
      output[stage] = {}
      continue
    paths = {}
    for path in PATHS:
      result = {}
      for field in path_fields:
        result[f"{field}_mean"], result[f"{field}_std"] = _mean_std(
            [float(stage["paths"][path][field]) for stage in stages]
        )
      paths[path] = result
    decomposition = {}
    for field in DECOMP_FIELDS:
      decomposition[field] = _aggregate(
          [float(stage.get("decomposition", {}).get(field, 0.0)) for stage in stages]
      )
    bias = {}
    for field in BIAS_FIELDS:
      bias[field] = _aggregate(
          [float(stage.get("bias", {}).get(field, 0.0)) for stage in stages]
      )
    degradation_result = {}
    for gain in ("G_C", "G_J"):
      gain_values = [
          {path: float(stage["paths"][path][gain]) for path in PATHS}
          for stage in stages
      ]
      for field in ("delta_state", "delta_odd", "delta_even"):
        degradation_result[f"{gain}_{field}_mean"], degradation_result[f"{gain}_{field}_std"] = _mean_std(
            [degradation(values)[field] for values in gain_values]
        )
    output[stage] = {
        "num_seeds": len(stages), "paths": paths,
        "decomposition": decomposition,
        "decomposition_flat": {
            f"{field}_{suffix}": value[suffix]
            for field, value in decomposition.items() for suffix in ("mean", "std")
        },
        "bias": bias,
        "bias_flat": {
            f"{field}_{suffix}": value[suffix]
            for field, value in bias.items() for suffix in ("mean", "std")
        },
        "stage_odd_response": _aggregate_stage_responses(stages, "stage_odd_response"),
        "secondary_stage_odd_response": _aggregate_stage_responses(
            stages, "secondary_stage_odd_response"
        ),
        "degradation": degradation_result,
    }
  return output


def attach_path_degradation(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> dict[str, object]:
  metrics = add_paired_gains(metrics)
  result: dict[str, object] = {
      "metrics": {branch: {path: dict(values) for path, values in paths.items()}
                  for branch, paths in metrics.items()}
  }
  result["degradation"] = {
      gain: degradation({path: float(metrics["corr"][path][gain]) for path in PATHS})
      for gain in ("G_C", "G_J")
  }
  result["P0_to_P1_to_P2_to_P3"] = {
      gain: {path: float(metrics["corr"][path][gain]) for path in PATHS}
      for gain in ("G_C", "G_J")
  }
  return result


__all__ = [
    "BIAS_FIELDS", "DECOMP_FIELDS", "METRIC_FIELDS", "add_paired_gains",
    "attach_path_degradation", "cancellation_statistics",
    "cancellation_metrics_from_jd", "cross_seed_aggregate", "degradation",
    "make_window_row", "paired_gains", "safe_gain", "safe_ratio",
    "write_window_rows", "aggregate_window_rows", "write_window_summary",
]
