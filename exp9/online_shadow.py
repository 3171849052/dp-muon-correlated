"""Online exact-window and exact-stage accumulation for Experiment 9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from exp9.core import BRANCHES, PATHS, Exp9DiagnosticStep
from exp9.diagnostics import (
    BIAS_FIELDS,
    DECOMP_FIELDS,
    cancellation_metrics_from_jd,
    make_window_row,
)


PyTree = Any
WINDOW_SIZE = 16


def _copy_blocks(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: np.asarray(value, dtype=np.float64).copy() for key, value in values.items()}


def _zeros_from(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: np.zeros_like(value, dtype=np.float64) for key, value in values.items()}


def _norm_sq(values: dict[str, np.ndarray]) -> float:
  return float(sum(np.sum(np.asarray(value, dtype=np.float64) ** 2) for value in values.values()))


def _sub(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: left[key] - right[key] for key in left}


def _metrics(bucket: "_Bucket") -> dict[str, dict[str, dict[str, float]]]:
  return {
      branch: {
          path: cancellation_metrics_from_jd(
              _norm_sq(bucket.d[branch][path]), bucket.D[branch][path]
          )
          for path in PATHS
      }
      for branch in BRANCHES
  }


@dataclass
class _Bucket:
  d: dict[str, dict[str, dict[str, np.ndarray]]]
  D: dict[str, dict[str, float]]
  bias_d: dict[str, dict[str, np.ndarray]]
  raw_d: dict[str, dict[str, np.ndarray]]
  decomposition: dict[str, float]
  bias_sum: dict[str, float]
  raw_response_sum: dict[str, float]
  ratio_sum: dict[str, float]
  ratio_count: int
  margin_min: float
  stage_odd_norm_sum: dict[str, dict[str, float]]
  secondary_stage_odd_norm_sum: dict[str, dict[str, float]]


def _new_bucket(template: dict[str, np.ndarray]) -> _Bucket:
  zero = _zeros_from(template)
  return _Bucket(
      d={branch: {path: _copy_blocks(zero) for path in PATHS} for branch in BRANCHES},
      D={branch: {path: 0.0 for path in PATHS} for branch in BRANCHES},
      bias_d={branch: _copy_blocks(zero) for branch in BRANCHES},
      raw_d={branch: _copy_blocks(zero) for branch in BRANCHES},
      decomposition={field: 0.0 for field in DECOMP_FIELDS},
      bias_sum={branch: 0.0 for branch in BRANCHES},
      raw_response_sum={branch: 0.0 for branch in BRANCHES},
      ratio_sum={branch: 0.0 for branch in BRANCHES},
      ratio_count=0, margin_min=float("inf"),
      stage_odd_norm_sum={
          branch: {stage: 0.0 for stage in ("linear", "bf16", "norm", "ns", "scale")}
          for branch in BRANCHES
      },
      secondary_stage_odd_norm_sum={
          branch: {stage: 0.0 for stage in ("linear", "bf16", "norm", "ns", "scale")}
          for branch in BRANCHES
      },
  )


def _apply_step(bucket: _Bucket, step: Exp9DiagnosticStep, *, learning_rate: float,
                weight_decay: float) -> None:
  a = 1.0 - float(learning_rate) * float(weight_decay)
  for branch in BRANCHES:
    for path in PATHS:
      x = {key: np.asarray(value, dtype=np.float64) for key, value in step.x[branch][path].items()}
      bucket.d[branch][path] = {
          key: a * bucket.d[branch][path][key] + x[key] for key in x
      }
      bucket.D[branch][path] = a * a * bucket.D[branch][path] + _norm_sq(x)
    bias = {key: np.asarray(value, dtype=np.float64) for key, value in step.bias[branch].items()}
    raw = {key: np.asarray(value, dtype=np.float64) for key, value in step.raw_response[branch].items()}
    bucket.bias_d[branch] = {
        key: a * bucket.bias_d[branch][key] - float(learning_rate) * bias[key]
        for key in bias
    }
    bucket.raw_d[branch] = {
        key: a * bucket.raw_d[branch][key] - float(learning_rate) * raw[key]
        for key in raw
    }
    bucket.bias_sum[branch] += float(sum(np.linalg.norm(value) for value in bias.values()))
    bucket.raw_response_sum[branch] += float(sum(np.linalg.norm(value) for value in raw.values()))
    bucket.ratio_sum[branch] += float(sum(np.asarray(value) for value in step.noise_signal_ratio[branch].values()))
    for stage, values in step.stage_odd[branch].items():
      bucket.stage_odd_norm_sum[branch][stage] += float(
          np.sqrt(sum(np.sum(np.asarray(value, dtype=np.float64) ** 2)
                      for value in values.values()))
      )
    for stage, values in step.secondary_stage_odd[branch].items():
      bucket.secondary_stage_odd_norm_sum[branch][stage] += float(
          np.sqrt(sum(np.sum(np.asarray(value, dtype=np.float64) ** 2)
                      for value in values.values()))
      )
  bucket.ratio_count += 1
  margins = [float(np.asarray(value)) for value in step.normalization_boundary_margin.values()]
  if margins:
    bucket.margin_min = min(bucket.margin_min, min(margins))
  bucket.decomposition["state_gap_energy"] += sum(
      _norm_sq(_sub(step.x[branch]["P0"], step.x[branch]["P1"])) for branch in ("corr",)
  )
  bucket.decomposition["odd_gap_energy"] += _norm_sq(
      _sub(step.x["corr"]["P1"], step.x["corr"]["P2"])
  )
  bucket.decomposition["even_gap_energy"] += _norm_sq(
      _sub(step.x["corr"]["P2"], step.x["corr"]["P3"])
  )
  reconstruction = max(
      float(np.asarray(step.odd_reconstruction_error[branch])) for branch in BRANCHES
  )
  bucket.decomposition["odd_reconstruction_error"] = max(
      bucket.decomposition["odd_reconstruction_error"], reconstruction
  )


def _bucket_bias(bucket: _Bucket) -> dict[str, float]:
  ratio = {
      branch: (bucket.ratio_sum[branch] / bucket.ratio_count
               if bucket.ratio_count else 0.0)
      for branch in BRANCHES
  }
  margin = bucket.margin_min if np.isfinite(bucket.margin_min) else 0.0
  return {
      "output_bias_norm_corr": bucket.bias_sum["corr"],
      "output_bias_norm_iid": bucket.bias_sum["iid"],
      "bias_endpoint_error_corr": float(np.sqrt(_norm_sq(bucket.bias_d["corr"]))),
      "bias_endpoint_error_iid": float(np.sqrt(_norm_sq(bucket.bias_d["iid"]))),
      "raw_private_clean_gap_endpoint_corr": float(np.sqrt(_norm_sq(bucket.raw_d["corr"]))),
      "raw_private_clean_gap_endpoint_iid": float(np.sqrt(_norm_sq(bucket.raw_d["iid"]))),
      "raw_private_clean_response_norm_corr": bucket.raw_response_sum["corr"],
      "raw_private_clean_response_norm_iid": bucket.raw_response_sum["iid"],
      "normalization_boundary_margin_min": margin,
      "noise_signal_ratio_mean_corr": ratio["corr"],
      "noise_signal_ratio_mean_iid": ratio["iid"],
  }


def _bucket_stage_response(bucket: _Bucket) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
  count = max(bucket.ratio_count, 1)
  primary = {
      branch: {stage: value / count for stage, value in values.items()}
      for branch, values in bucket.stage_odd_norm_sum.items()
  }
  secondary = {
      branch: {stage: value / count for stage, value in values.items()}
      for branch, values in bucket.secondary_stage_odd_norm_sum.items()
  }
  return primary, secondary


def _host_step(step: Exp9DiagnosticStep) -> Exp9DiagnosticStep:
  # The collector deliberately converts arrays at the boundary, keeping all
  # primary map computations inside the compiled training step.
  return step


class Exp9WindowCollector:
  """Collect exact non-overlapping windows and exact early/late/full stages."""

  def __init__(self, params: PyTree, *, seed: int, learning_rate: float,
               weight_decay: float, horizon: int, window_size: int = WINDOW_SIZE) -> None:
    del params
    if horizon < 1 or window_size < 1:
      raise ValueError("horizon and window_size must be positive")
    self.seed = int(seed)
    self.horizon = int(horizon)
    self.window_size = int(window_size)
    self.learning_rate = float(learning_rate)
    self.weight_decay = float(weight_decay)
    self._template: dict[str, np.ndarray] | None = None
    self._window: _Bucket | None = None
    self._full: _Bucket | None = None
    self._stage: _Bucket | None = None
    self._early: _Bucket | None = None
    self._late: _Bucket | None = None
    self._window_count = 0
    self._window_index = 0
    self._rows: list[dict[str, object]] = []
    self._finalized = False

  @property
  def rows(self) -> list[dict[str, object]]:
    return list(self._rows)

  def _ensure(self, step: Exp9DiagnosticStep) -> None:
    if self._template is not None:
      return
    self._template = {
        key: np.asarray(value, dtype=np.float64) for key, value in step.clean_pre_q.items()
    }
    self._window = _new_bucket(self._template)
    self._full = _new_bucket(self._template)
    self._stage = _new_bucket(self._template)

  def _append_bucket_row(self, bucket: _Bucket, *, start: int, end: int, index: int) -> None:
    self._rows.append(make_window_row(
        seed=self.seed, window_index=index, start_step=start, end_step=end,
        metrics=_metrics(bucket), decomposition=bucket.decomposition,
        bias=_bucket_bias(bucket),
    ))

  def after_step(self, state: Any, step: int) -> None:
    if step != int(np.asarray(state.step)):
      raise ValueError("callback step must equal Exp9 train state step")
    current = _host_step(state.last_step)
    self._ensure(current)
    assert self._window is not None and self._full is not None and self._stage is not None
    _apply_step(self._window, current, learning_rate=self.learning_rate, weight_decay=self.weight_decay)
    _apply_step(self._full, current, learning_rate=self.learning_rate, weight_decay=self.weight_decay)
    _apply_step(self._stage, current, learning_rate=self.learning_rate, weight_decay=self.weight_decay)
    self._window_count += 1
    if self._window_count == self.window_size:
      self._append_bucket_row(
          self._window, start=self._window_index * self.window_size + 1,
          end=step, index=self._window_index,
      )
      self._window = _new_bucket(self._template)  # type: ignore[arg-type]
      self._window_count = 0
      self._window_index += 1
    early_end = min(97, self.horizon)
    if step == early_end:
      self._early = self._stage
      self._stage = _new_bucket(self._template)  # type: ignore[arg-type]
    elif step == self.horizon and self.horizon > early_end:
      self._late = self._stage

  def finalize(self) -> list[dict[str, object]]:
    if self._finalized:
      return self.rows
    assert self._window is not None
    if self._window_count:
      start = self._window_index * self.window_size + 1
      self._append_bucket_row(
          self._window, start=start, end=start + self._window_count - 1,
          index=self._window_index,
      )
    self._finalized = True
    return self.rows

  def _stage_payload(self, bucket: _Bucket | None, *, start: int, end: int) -> dict[str, object]:
    if bucket is None:
      template = self._template or {}
      bucket = _new_bucket(template)
    metrics = _metrics(bucket)
    stage_odd, secondary_stage_odd = _bucket_stage_response(bucket)
    return {
        "start_step": int(start), "end_step": int(end),
        "num_steps": max(0, int(end) - int(start) + 1),
        "metrics": metrics, "decomposition": dict(bucket.decomposition),
        "bias": _bucket_bias(bucket),
        "stage_odd_response": stage_odd,
        "secondary_stage_odd_response": secondary_stage_odd,
    }

  def stage_summaries(self) -> dict[str, dict[str, object]]:
    early_end = min(97, self.horizon)
    return {
        "early": self._stage_payload(self._early, start=1, end=early_end),
        "late": self._stage_payload(self._late, start=98, end=self.horizon),
        "full": self._stage_payload(self._full, start=1, end=self.horizon),
    }


__all__ = ["Exp9WindowCollector", "WINDOW_SIZE"]
