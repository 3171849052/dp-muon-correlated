"""Online exact-window and exact-stage accumulation for Experiment 9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from exp9.core import BRANCHES, PATHS, PRIMARY_STAGES, STAGES, Exp9DiagnosticStep
from exp9.diagnostics import (
    BIAS_FIELDS, DECOMP_FIELDS, cancellation_metrics_from_jd, make_window_row,
    paired_gains, safe_ratio,
)


PyTree = Any
WINDOW_SIZE = 16
_EPS = 1e-12


def _copy_blocks(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: np.asarray(value, dtype=np.float64).copy() for key, value in values.items()}


def _zeros_from(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: np.zeros_like(value, dtype=np.float64) for key, value in values.items()}


def _array(value: Any, name: str) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if not np.all(np.isfinite(result)):
    raise ValueError(f"non-finite primary diagnostic value: {name}")
  return result


def _norm_sq(values: dict[str, np.ndarray]) -> float:
  result = float(sum(np.sum(_array(value, "block") ** 2) for value in values.values()))
  if not np.isfinite(result):
    raise ValueError("non-finite accumulated Frobenius norm")
  return result


def _sub(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: left[key] - right[key] for key in left}


def _metrics(bucket: "_Bucket") -> dict[str, dict[str, dict[str, object]]]:
  return {
      branch: {
          path: cancellation_metrics_from_jd(
              _norm_sq(bucket.d[branch][path]), bucket.D[branch][path]
          )
          for path in PATHS
      }
      for branch in BRANCHES
  }


def _stage_metrics(bucket: "_Bucket") -> dict[str, dict[str, dict[str, dict[str, object]]]]:
  result = {
      branch: {
          stage: cancellation_metrics_from_jd(
              _norm_sq(bucket.stage_d[branch][stage]), bucket.stage_D[branch][stage]
          )
          for stage in PRIMARY_STAGES
      }
      for branch in BRANCHES
  }
  for stage in PRIMARY_STAGES:
    gains = paired_gains(result["corr"][stage], result["iid"][stage])
    result["corr"][stage].update(gains)
    result["iid"][stage].update(gains)
  return result


@dataclass
class _Bucket:
  d: dict[str, dict[str, dict[str, np.ndarray]]]
  D: dict[str, dict[str, float]]
  stage_d: dict[str, dict[str, dict[str, np.ndarray]]]
  stage_D: dict[str, dict[str, dict[str, float]]]
  bias_d: dict[str, dict[str, np.ndarray]]
  raw_d: dict[str, dict[str, np.ndarray]]
  probe_disagreement_d: dict[str, dict[str, np.ndarray]]
  decomposition: dict[str, float]
  bias_sum: dict[str, float]
  bias_sq_sum: dict[str, float]
  raw_response_sum: dict[str, float]
  probe_disagreement_norm_sum: dict[str, float]
  probe_disagreement_sq_sum: dict[str, float]
  probe_error_energy: dict[str, float]
  global_ratio_sum: dict[str, float]
  block_ratio_mean_sum: dict[str, float]
  block_ratio_max_sum: dict[str, float]
  ratio_count: int
  clean_norm_min: float
  stage_odd_norm_sum: dict[str, dict[str, float]]
  secondary_stage_odd_norm_sum: dict[str, dict[str, float]]


def _new_bucket(template: dict[str, np.ndarray]) -> _Bucket:
  zero = _zeros_from(template)
  return _Bucket(
      d={branch: {path: _copy_blocks(zero) for path in PATHS} for branch in BRANCHES},
      D={branch: {path: 0.0 for path in PATHS} for branch in BRANCHES},
      stage_d={branch: {stage: _copy_blocks(zero) for stage in PRIMARY_STAGES}
               for branch in BRANCHES},
      stage_D={branch: {stage: 0.0 for stage in PRIMARY_STAGES} for branch in BRANCHES},
      bias_d={branch: _copy_blocks(zero) for branch in BRANCHES},
      raw_d={branch: _copy_blocks(zero) for branch in BRANCHES},
      probe_disagreement_d={branch: _copy_blocks(zero) for branch in BRANCHES},
      decomposition={field: 0.0 for field in DECOMP_FIELDS},
      bias_sum={branch: 0.0 for branch in BRANCHES},
      bias_sq_sum={branch: 0.0 for branch in BRANCHES},
      raw_response_sum={branch: 0.0 for branch in BRANCHES},
      probe_disagreement_norm_sum={branch: 0.0 for branch in BRANCHES},
      probe_disagreement_sq_sum={branch: 0.0 for branch in BRANCHES},
      probe_error_energy={branch: 0.0 for branch in BRANCHES},
      global_ratio_sum={branch: 0.0 for branch in BRANCHES},
      block_ratio_mean_sum={branch: 0.0 for branch in BRANCHES},
      block_ratio_max_sum={branch: 0.0 for branch in BRANCHES},
      ratio_count=0, clean_norm_min=float("inf"),
      stage_odd_norm_sum={
          branch: {stage: 0.0 for stage in STAGES} for branch in BRANCHES
      },
      secondary_stage_odd_norm_sum={
          branch: {stage: 0.0 for stage in STAGES} for branch in BRANCHES
      },
  )


def _apply_step(bucket: _Bucket, step: Exp9DiagnosticStep, *, learning_rate: float,
                weight_decay: float) -> None:
  a = 1.0 - float(learning_rate) * float(weight_decay)
  for branch in BRANCHES:
    for path in PATHS:
      x = {key: _array(value, f"x/{branch}/{path}/{key}")
           for key, value in step.x[branch][path].items()}
      bucket.d[branch][path] = {
          key: a * bucket.d[branch][path][key] + x[key] for key in x
      }
      bucket.D[branch][path] = a * a * bucket.D[branch][path] + _norm_sq(x)
    bias = {key: _array(value, f"bias/{branch}/{key}")
            for key, value in step.bias[branch].items()}
    raw = {key: _array(value, f"raw/{branch}/{key}")
           for key, value in step.raw_response[branch].items()}
    probe = {key: _array(value, f"probe_disagreement/{branch}/{key}")
             for key, value in step.probe_disagreement[branch].items()}
    bucket.bias_d[branch] = {
        key: a * bucket.bias_d[branch][key] - float(learning_rate) * bias[key]
        for key in bias
    }
    bucket.raw_d[branch] = {
        key: a * bucket.raw_d[branch][key] - float(learning_rate) * raw[key]
        for key in raw
    }
    bucket.probe_disagreement_d[branch] = {
        key: a * bucket.probe_disagreement_d[branch][key] - float(learning_rate) * probe[key]
        for key in probe
    }
    bucket.bias_sum[branch] += float(np.sqrt(_norm_sq(bias)))
    bucket.bias_sq_sum[branch] += _norm_sq(bias)
    bucket.raw_response_sum[branch] += float(np.sqrt(_norm_sq(raw)))
    bucket.probe_disagreement_norm_sum[branch] += float(np.sqrt(_norm_sq(probe)))
    bucket.probe_disagreement_sq_sum[branch] += _norm_sq(probe)
    # The standard error of B_hat=(B_A+B_B)/2 is half the A/B
    # disagreement.  This is the corresponding optimizer-space energy.
    bucket.probe_error_energy[branch] = (
        a * a * bucket.probe_error_energy[branch]
        + .25 * float(learning_rate) ** 2 * _norm_sq(probe)
    )
    for stage in STAGES:
      values = {key: _array(value, f"stage/{branch}/{stage}/{key}")
                for key, value in step.stage_odd[branch][stage].items()}
      secondary_values = {
          key: _array(value, f"secondary_stage/{branch}/{stage}/{key}")
          for key, value in step.secondary_stage_odd[branch][stage].items()
      }
      bucket.stage_odd_norm_sum[branch][stage] += float(np.sqrt(_norm_sq(values)))
      bucket.secondary_stage_odd_norm_sum[branch][stage] += float(
          np.sqrt(_norm_sq(secondary_values))
      )
      if stage in PRIMARY_STAGES:
        stage_x = {key: -float(learning_rate) * value for key, value in values.items()}
        bucket.stage_d[branch][stage] = {
            key: a * bucket.stage_d[branch][stage][key] + stage_x[key]
            for key in stage_x
        }
        bucket.stage_D[branch][stage] = (
            a * a * bucket.stage_D[branch][stage] + _norm_sq(stage_x)
        )
    global_ratio = float(_array(step.global_noise_signal_ratio[branch],
                                f"global_noise_signal_ratio/{branch}"))
    block_mean = float(_array(step.block_ratio_mean[branch], f"block_ratio_mean/{branch}"))
    block_max = float(_array(step.block_ratio_max[branch], f"block_ratio_max/{branch}"))
    bucket.global_ratio_sum[branch] += global_ratio
    bucket.block_ratio_mean_sum[branch] += block_mean
    bucket.block_ratio_max_sum[branch] += block_max
  clean_min = float(_array(step.clean_pre_q_norm_min, "clean_pre_q_norm_min"))
  bucket.clean_norm_min = min(bucket.clean_norm_min, clean_min)
  bucket.ratio_count += 1
  bucket.decomposition["state_gap_energy"] += _norm_sq(
      _sub(step.x["corr"]["P0"], step.x["corr"]["P1"])
  )
  bucket.decomposition["odd_gap_energy"] += _norm_sq(
      _sub(step.x["corr"]["P1"], step.x["corr"]["P2"])
  )
  bucket.decomposition["even_gap_energy"] += _norm_sq(
      _sub(step.x["corr"]["P2"], step.x["corr"]["P3"])
  )
  reconstruction = max(
      float(_array(step.odd_reconstruction_error[branch],
                   f"odd_reconstruction_error/{branch}")) for branch in BRANCHES
  )
  bucket.decomposition["odd_reconstruction_error"] = max(
      bucket.decomposition["odd_reconstruction_error"], reconstruction
  )
  scalar_values = [
      *bucket.D["corr"].values(), *bucket.D["iid"].values(),
      *bucket.stage_D["corr"].values(), *bucket.stage_D["iid"].values(),
      *bucket.bias_sum.values(), *bucket.bias_sq_sum.values(),
      *bucket.raw_response_sum.values(), *bucket.probe_disagreement_norm_sum.values(),
      *bucket.probe_disagreement_sq_sum.values(), *bucket.global_ratio_sum.values(),
      *bucket.probe_error_energy.values(),
      *bucket.block_ratio_mean_sum.values(), *bucket.block_ratio_max_sum.values(),
  ]
  if not np.all(np.isfinite(np.asarray(scalar_values, dtype=np.float64))):
    raise ValueError("non-finite accumulated primary diagnostic")


def _bucket_bias(bucket: _Bucket) -> dict[str, object]:
  if bucket.ratio_count == 0:
    return {
        field: (False if field.startswith("P3_reliable_") else None)
        for field in BIAS_FIELDS
    }
  count = bucket.ratio_count
  p3_metrics = _metrics(bucket)
  relative = {
      # Auxiliary only: preserve the requested +eps convention.  This value
      # never participates in the P3 reliability decision below.
      branch: float(
          np.sqrt(bucket.probe_disagreement_sq_sum[branch])
          / (np.sqrt(bucket.bias_sq_sum[branch]) + _EPS)
      ) for branch in BRANCHES
  }
  probe_error_to_d = {
      branch: safe_ratio(
          bucket.probe_error_energy[branch],
          float(p3_metrics[branch]["P3"]["D"]), eps=_EPS,
      ) for branch in BRANCHES
  }
  probe_error_endpoint_est = {
      branch: .5 * float(np.sqrt(_norm_sq(bucket.probe_disagreement_d[branch])))
      for branch in BRANCHES
  }
  p3_endpoint = {
      branch: float(np.sqrt(p3_metrics[branch]["P3"]["J"]))
      for branch in BRANCHES
  }
  probe_error_to_endpoint = {
      branch: safe_ratio(
          probe_error_endpoint_est[branch], p3_endpoint[branch], eps=_EPS
      ) for branch in BRANCHES
  }
  reliable = {
      branch: (
          probe_error_to_d[branch] is not None
          and probe_error_to_endpoint[branch] is not None
          and probe_error_to_d[branch] <= .1
          and probe_error_to_endpoint[branch] <= .1
      ) for branch in BRANCHES
  }
  clean_min = bucket.clean_norm_min if np.isfinite(bucket.clean_norm_min) else None
  return {
      "output_bias_norm_corr": bucket.bias_sum["corr"],
      "output_bias_norm_iid": bucket.bias_sum["iid"],
      "bias_endpoint_error_corr": float(np.sqrt(_norm_sq(bucket.bias_d["corr"]))),
      "bias_endpoint_error_iid": float(np.sqrt(_norm_sq(bucket.bias_d["iid"]))),
      "raw_private_clean_gap_endpoint_corr": float(np.sqrt(_norm_sq(bucket.raw_d["corr"]))),
      "raw_private_clean_gap_endpoint_iid": float(np.sqrt(_norm_sq(bucket.raw_d["iid"]))),
      "raw_private_clean_response_norm_corr": bucket.raw_response_sum["corr"],
      "raw_private_clean_response_norm_iid": bucket.raw_response_sum["iid"],
      "probe_disagreement_norm_corr": bucket.probe_disagreement_norm_sum["corr"] / count,
      "probe_disagreement_norm_iid": bucket.probe_disagreement_norm_sum["iid"] / count,
      "probe_disagreement_relative_to_bias_corr": relative["corr"],
      "probe_disagreement_relative_to_bias_iid": relative["iid"],
      "probe_disagreement_endpoint_corr": float(
          np.sqrt(_norm_sq(bucket.probe_disagreement_d["corr"]))
      ),
      "probe_disagreement_endpoint_iid": float(
          np.sqrt(_norm_sq(bucket.probe_disagreement_d["iid"]))
      ),
      "probe_error_to_P3_D_corr": probe_error_to_d["corr"],
      "probe_error_to_P3_D_iid": probe_error_to_d["iid"],
      "probe_error_to_P3_endpoint_corr": probe_error_to_endpoint["corr"],
      "probe_error_to_P3_endpoint_iid": probe_error_to_endpoint["iid"],
      "P3_reliable_corr": reliable["corr"],
      "P3_reliable_iid": reliable["iid"],
      "P3_reliable_paired": reliable["corr"] and reliable["iid"],
      "clean_pre_q_norm_min": clean_min,
      "global_noise_signal_ratio_mean_corr": bucket.global_ratio_sum["corr"] / count,
      "global_noise_signal_ratio_mean_iid": bucket.global_ratio_sum["iid"] / count,
      "block_noise_signal_ratio_mean_corr": bucket.block_ratio_mean_sum["corr"] / count,
      "block_noise_signal_ratio_mean_iid": bucket.block_ratio_mean_sum["iid"] / count,
      "block_noise_signal_ratio_max_corr": bucket.block_ratio_max_sum["corr"] / count,
      "block_noise_signal_ratio_max_iid": bucket.block_ratio_max_sum["iid"] / count,
  }


def _bucket_stage_response(bucket: _Bucket) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, float | None]]]:
  if bucket.ratio_count == 0:
    empty = {branch: {stage: None for stage in STAGES} for branch in BRANCHES}
    return empty, {branch: dict(values) for branch, values in empty.items()}
  count = bucket.ratio_count
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
        key: _array(value, f"clean_pre_q/{key}")
        for key, value in step.clean_pre_q.items()
    }
    self._window = _new_bucket(self._template)
    self._full = _new_bucket(self._template)
    self._stage = _new_bucket(self._template)

  def _append_bucket_row(self, bucket: _Bucket, *, start: int, end: int, index: int) -> None:
    self._rows.append(make_window_row(
        seed=self.seed, window_index=index, start_step=start, end_step=end,
        metrics=_metrics(bucket), decomposition=bucket.decomposition,
        bias=_bucket_bias(bucket), stage_metrics=_stage_metrics(bucket),
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

  def stage_summaries(self) -> dict[str, dict[str, object]]:
    early_end = min(97, self.horizon)
    result = {}
    for name, bucket, start, end in (
        ("early", self._early, 1, early_end),
        ("late", self._late, 98, self.horizon),
        ("full", self._full, 1, self.horizon),
    ):
      if bucket is None:
        template = self._template or {}
        bucket = _new_bucket(template)
      metrics = _metrics(bucket)
      stage_odd, secondary_stage_odd = _bucket_stage_response(bucket)
      result[name] = {
          "start_step": int(start), "end_step": int(end),
          "num_steps": max(0, int(end) - int(start) + 1),
          "metrics": metrics, "paths": {
              path: {
                  "C_corr": metrics["corr"][path].get("C"),
                  "C_iid": metrics["iid"][path].get("C"),
                  "J_corr": metrics["corr"][path].get("J"),
                  "J_iid": metrics["iid"][path].get("J"),
                  "D_corr": metrics["corr"][path].get("D"),
                  "D_iid": metrics["iid"][path].get("D"),
                  "G_C": metrics["corr"][path].get("G_C"),
                  "G_J": metrics["corr"][path].get("G_J"),
              } for path in PATHS
          },
          "decomposition": dict(bucket.decomposition),
          "bias": _bucket_bias(bucket),
          "stage_metrics": _stage_metrics(bucket),
          "stage_odd_response": stage_odd,
          "secondary_stage_odd_response": secondary_stage_odd,
      }
    return result


__all__ = ["Exp9WindowCollector", "WINDOW_SIZE"]
