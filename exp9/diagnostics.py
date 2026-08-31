"""Exact endpoint, window, and cross-seed aggregation for Experiment 9."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from exp9.core import BRANCHES, PATHS, PRIMARY_STAGES


Number = float | int | bool | None
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
    "probe_disagreement_norm_corr", "probe_disagreement_norm_iid",
    "probe_disagreement_relative_corr", "probe_disagreement_relative_iid",
    "probe_disagreement_endpoint_corr", "probe_disagreement_endpoint_iid",
    "P3_reliable_corr", "P3_reliable_iid",
    "clean_pre_q_norm_min",
    "global_noise_signal_ratio_mean_corr", "global_noise_signal_ratio_mean_iid",
    "block_noise_signal_ratio_mean_corr", "block_noise_signal_ratio_mean_iid",
    "block_noise_signal_ratio_max_corr", "block_noise_signal_ratio_max_iid",
)


def _finite_or_none(value: object, *, name: str) -> Number:
  if value is None:
    return None
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  result = float(value)
  if not np.isfinite(result):
    raise ValueError(f"non-finite diagnostic value in {name}: {value!r}")
  return result


def safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-30) -> float | None:
  """Return a ratio, or ``None`` when its denominator is invalid."""
  numerator_value = _finite_or_none(numerator, name="ratio numerator")
  denominator_value = _finite_or_none(denominator, name="ratio denominator")
  if numerator_value is None or denominator_value is None:
    return None
  if abs(float(denominator_value)) <= eps:
    return None
  value = float(numerator_value) / float(denominator_value)
  if not np.isfinite(value):
    raise ValueError("non-finite ratio")
  return value


def safe_gain(corr: float | None, iid: float | None, *, eps: float = 1e-30) -> float | None:
  """Return a paired gain, preserving invalid zero-denominator cases as null."""
  corr_value = _finite_or_none(corr, name="correlated gain input")
  iid_value = _finite_or_none(iid, name="IID gain input")
  if corr_value is None or iid_value is None or abs(float(iid_value)) <= eps:
    return None
  value = 1.0 - float(corr_value) / float(iid_value)
  if not np.isfinite(value):
    raise ValueError("non-finite gain")
  return float(value)


def cancellation_metrics_from_jd(j: float, d: float) -> dict[str, Number]:
  j_value = _finite_or_none(j, name="J")
  d_value = _finite_or_none(d, name="D")
  if j_value is None or d_value is None:
    raise ValueError("J and D must be finite")
  cancellation = safe_ratio(float(j_value), float(d_value))
  return {"J": float(j_value), "D": float(d_value), "C": cancellation,
          "valid": cancellation is not None}


def cancellation_statistics(
    x: np.ndarray, *, weight_decay: float, learning_rate: float
) -> dict[str, Number]:
  """Reference implementation of the Exp9 endpoint semantics."""
  values = np.asarray(x, dtype=np.float64)
  if values.ndim < 1:
    raise ValueError("x must have a time dimension")
  if not np.all(np.isfinite(values)):
    raise ValueError("x contains non-finite values")
  a = 1.0 - float(learning_rate) * float(weight_decay)
  d = np.zeros(values.shape[1:], dtype=np.float64)
  denominator = 0.0
  for value in values:
    d = a * d + value
    denominator = a * a * denominator + float(np.sum(value * value))
  endpoint = float(np.sum(d * d))
  return {"J": endpoint, "D": denominator, "C": safe_ratio(endpoint, denominator),
          "valid": denominator > 1e-30}


def paired_gains(corr: Mapping[str, Number], iid: Mapping[str, Number]) -> dict[str, Number]:
  return {"G_C": safe_gain(corr.get("C"), iid.get("C")),
          "G_J": safe_gain(corr.get("J"), iid.get("J"))}


def add_paired_gains(
    metrics: Mapping[str, Mapping[str, Mapping[str, Number]]]
) -> dict[str, dict[str, dict[str, Number]]]:
  result = {
      branch: {path: {
          field: _finite_or_none(value, name=f"{branch}/{path}/{field}")
          for field, value in values.items()
      } for path, values in paths.items()}
      for branch, paths in metrics.items()
  }
  for path in PATHS:
    gains = paired_gains(result["corr"][path], result["iid"][path])
    result["corr"][path].update(gains)
    result["iid"][path].update(gains)
  return result


def degradation(gains: Mapping[str, Number]) -> dict[str, Number]:
  """Return adjacent losses, preserving invalid gains as null."""
  def subtract(left: Number, right: Number) -> float | None:
    if left is None or right is None:
      return None
    value = float(left) - float(right)
    if not np.isfinite(value):
      raise ValueError("non-finite degradation")
    return value
  return {
      "delta_state": subtract(gains.get("P0"), gains.get("P1")),
      "delta_odd": subtract(gains.get("P1"), gains.get("P2")),
      "delta_even": subtract(gains.get("P2"), gains.get("P3")),
  }


def _finite_mapping(values: Mapping[str, object], fields: Sequence[str]) -> dict[str, Number]:
  return {
      field: _finite_or_none(values.get(field, 0.0), name=field)
      for field in fields
  }


def _stage_window_fields() -> list[str]:
  return [
      f"stage_{stage}_{branch}_{field}"
      for branch in BRANCHES for stage in PRIMARY_STAGES
      for field in METRIC_FIELDS
  ]


def make_window_row(
    *, seed: int, window_index: int, start_step: int, end_step: int,
    metrics: Mapping[str, Mapping[str, Mapping[str, Number]]],
    decomposition: Mapping[str, float], bias: Mapping[str, object],
    stage_metrics: Mapping[str, Mapping[str, Mapping[str, Number]]] | None = None,
) -> dict[str, object]:
  metrics = add_paired_gains(metrics)
  row: dict[str, object] = {
      "seed": int(seed), "window_index": int(window_index),
      "start_step": int(start_step), "end_step": int(end_step),
  }
  for branch in BRANCHES:
    for path in PATHS:
      for field in METRIC_FIELDS:
        row[f"{field}_{branch}_{path}"] = metrics[branch][path].get(field)
  row.update(_finite_mapping(decomposition, DECOMP_FIELDS))
  row.update(_finite_mapping(bias, BIAS_FIELDS))
  if stage_metrics is not None:
    for branch in BRANCHES:
      for stage in PRIMARY_STAGES:
        values = stage_metrics.get(branch, {}).get(stage, {})
        for field in METRIC_FIELDS:
          row[f"stage_{stage}_{branch}_{field}"] = values.get(field)
  return row


def _window_fields() -> list[str]:
  return [
      f"{field}_{branch}_{path}"
      for branch in BRANCHES for path in PATHS for field in METRIC_FIELDS
  ] + list(DECOMP_FIELDS) + list(BIAS_FIELDS) + _stage_window_fields()


def write_window_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
  rows = list(rows)
  fields = ["seed", "window_index", "start_step", "end_step"] + _window_fields()
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _mean_std(values: Sequence[Number]) -> tuple[Number, Number]:
  valid = []
  for value in values:
    checked = _finite_or_none(value, name="aggregate")
    if checked is not None:
      valid.append(float(checked))
  if not valid:
    return None, None
  return float(np.mean(valid)), float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def aggregate_window_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
  groups: dict[int, list[Mapping[str, object]]] = {}
  for row in rows:
    groups.setdefault(int(row["window_index"]), []).append(row)
  output = []
  for index, group in sorted(groups.items()):
    result: dict[str, object] = {
        "window_index": index, "start_step": int(group[0]["start_step"]),
        "end_step": int(group[0]["end_step"]), "num_seeds": len(group),
    }
    for field in _window_fields():
      mean, std = _mean_std([row.get(field) for row in group])
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


def _aggregate(values: Sequence[Number]) -> dict[str, Number]:
  mean, std = _mean_std(values)
  return {"mean": mean, "std": std}


def _aggregate_stage_responses(
    stages: Sequence[Mapping[str, object]], field: str
) -> dict[str, dict[str, dict[str, Number]]]:
  output: dict[str, dict[str, dict[str, Number]]] = {}
  for branch in BRANCHES:
    output[branch] = {}
    for stage_name in ("linear", "bf16", "norm", "ns", "scale"):
      values = [
          stage.get(field, {}).get(branch, {}).get(stage_name)  # type: ignore[union-attr]
          for stage in stages
      ]
      output[branch][stage_name] = _aggregate(values)
  return output


def _aggregate_stage_metrics(
    stages: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, dict[str, Number]]]]:
  output = {}
  for branch in BRANCHES:
    output[branch] = {}
    for stage_name in PRIMARY_STAGES:
      output[branch][stage_name] = {}
      for field in METRIC_FIELDS:
        values = [
            stage.get("stage_metrics", {}).get(branch, {}).get(stage_name, {}).get(field)
            for stage in stages
        ]
        output[branch][stage_name][field] = _aggregate(values)
  return output


def cross_seed_aggregate(
    per_seed: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> dict[str, object]:
  """Aggregate values recomputed from raw steps for each exact stage."""
  output: dict[str, object] = {}
  path_fields = ("C_corr", "C_iid", "J_corr", "J_iid", "D_corr", "D_iid", "G_C", "G_J")
  for stage_name in ("early", "late", "full"):
    stages = [run[stage_name] for run in per_seed.values() if stage_name in run]
    if not stages:
      output[stage_name] = {}
      continue
    paths = {}
    for path in PATHS:
      result = {}
      for field in path_fields:
        mean, std = _mean_std([stage["paths"][path].get(field) for stage in stages])
        result[f"{field}_mean"], result[f"{field}_std"] = mean, std
      paths[path] = result
    decomposition = {
        field: _aggregate([stage.get("decomposition", {}).get(field) for stage in stages])
        for field in DECOMP_FIELDS
    }
    bias = {
        field: _aggregate([stage.get("bias", {}).get(field) for stage in stages])
        for field in BIAS_FIELDS
    }
    degradation_result = {}
    for gain in ("G_C", "G_J"):
      gain_values = [
          {path: stage["paths"][path].get(gain) for path in PATHS}
          for stage in stages
      ]
      for field in ("delta_state", "delta_odd", "delta_even"):
        mean, std = _mean_std([degradation(values)[field] for values in gain_values])
        degradation_result[f"{gain}_{field}_mean"] = mean
        degradation_result[f"{gain}_{field}_std"] = std
    output[stage_name] = {
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
        "stage_metrics": _aggregate_stage_metrics(stages),
        "stage_odd_response": _aggregate_stage_responses(stages, "stage_odd_response"),
        "secondary_stage_odd_response": _aggregate_stage_responses(
            stages, "secondary_stage_odd_response"
        ),
        "degradation": degradation_result,
    }
  return output


def attach_path_degradation(
    metrics: Mapping[str, Mapping[str, Mapping[str, Number]]]
) -> dict[str, object]:
  metrics = add_paired_gains(metrics)
  result: dict[str, object] = {
      "metrics": {branch: {path: dict(values) for path, values in paths.items()}
                  for branch, paths in metrics.items()}
  }
  result["degradation"] = {
      gain: degradation({path: metrics["corr"][path].get(gain) for path in PATHS})
      for gain in ("G_C", "G_J")
  }
  result["P0_to_P1_to_P2_to_P3"] = {
      gain: {path: metrics["corr"][path].get(gain) for path in PATHS}
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
