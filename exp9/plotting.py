"""Plots for Experiment 9's primary cancellation and bias diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from exp9.core import PATHS


def _plot_value(value: object) -> float:
  if value is None:
    return np.nan
  value = float(value)
  return value if np.isfinite(value) else np.nan


def _mean_window_values(rows: Iterable[Mapping[str, object]], field: str):
  groups: dict[int, list[Mapping[str, object]]] = {}
  for row in rows:
    groups.setdefault(int(row["window_index"]), []).append(row)
  values = []
  for _, group in sorted(groups.items()):
    midpoint = np.mean([(float(row["start_step"]) + float(row["end_step"])) / 2 for row in group])
    numeric = [_plot_value(row.get(field)) for row in group]
    numeric = [value for value in numeric if np.isfinite(value)]
    if numeric:
      values.append((midpoint, np.mean(numeric)))
  return values


def plot_cancellation_paths(
    rows: Iterable[Mapping[str, object]], output: str | Path, *, gain: str = "G_C"
) -> None:
  rows = list(rows)
  figure, axis = plt.subplots(figsize=(7.4, 4.4))
  for path in PATHS:
    values = _mean_window_values(rows, f"{gain}_corr_{path}")
    if values:
      axis.plot([x for x, _ in values], [y for _, y in values], marker="o", label=path)
  axis.set(xlabel="training step (window midpoint)", ylabel=gain)
  axis.grid(alpha=.25)
  axis.legend(ncol=4)
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_gain_over_steps(rows: Iterable[Mapping[str, object]], output: str | Path, *, gain: str) -> None:
  plot_cancellation_paths(rows, output, gain=gain)


def plot_path_gain_summary(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
  stages = ("early", "late", "full")
  for axis, gain in zip(axes, ("G_C", "G_J"), strict=True):
    for stage in stages:
      if stage not in summaries or "paths" not in summaries[stage]:
        continue
      values = summaries[stage]["paths"]
      y = [_plot_value(values[path].get(f"{gain}_mean")) for path in PATHS]
      error = [_plot_value(values[path].get(f"{gain}_std")) for path in PATHS]
      axis.errorbar(range(4), y, yerr=error, marker="o", capsize=3, label=stage)
    axis.set_xticks(range(4), ("P0", "P1", "P2", "P3"))
    axis.set_title(gain)
    axis.grid(alpha=.25)
  axes[0].set_ylabel("gain")
  axes[-1].legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_bias_diagnostics(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)
  stages = ("early", "late", "full")
  x = np.arange(len(stages))
  for axis, metric, title in zip(
      axes,
      ("output_bias_norm", "raw_private_clean_gap_endpoint"),
      ("output bias (sum of Frobenius norms)", "raw private-clean gap endpoint"),
      strict=True,
  ):
    for branch in ("corr", "iid"):
      field = f"{metric}_{branch}"
      y = [
          _plot_value(summaries[stage].get("bias_flat", {}).get(f"{field}_mean"))
          if stage in summaries else np.nan for stage in stages
      ]
      error = [
          _plot_value(summaries[stage].get("bias_flat", {}).get(f"{field}_std"))
          if stage in summaries else np.nan for stage in stages
      ]
      axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=branch)
    axis.set_title(title)
    axis.grid(alpha=.25)
  axes[0].set_xticks(x, stages)
  axes[0].set_ylabel("diagnostic quantity")
  axes[-1].legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_decomposition(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  """Plot the three path-gap energies; these are not bias/cancellation gains."""
  figure, axis = plt.subplots(figsize=(7.2, 4.2))
  stages = ("early", "late", "full")
  x = np.arange(len(stages))
  for field, label in (("state_gap_energy", "state"),
                       ("odd_gap_energy", "odd"),
                       ("even_gap_energy", "even")):
    y = [_plot_value(summaries[stage].get("decomposition_flat", {}).get(field + "_mean"))
         if stage in summaries else np.nan for stage in stages]
    error = [_plot_value(summaries[stage].get("decomposition_flat", {}).get(field + "_std"))
             if stage in summaries else np.nan for stage in stages]
    axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=label)
  axis.set_xticks(x, stages)
  axis.set_ylabel("correlated path-gap energy")
  axis.grid(alpha=.25)
  axis.legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_stage_diagnostics(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  """Plot primary and production-precision odd response by Q stage."""
  figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
  stage_names = ("norm", "ns", "scale")
  for axis, field, title in zip(
      axes,
      ("stage_odd_response", "secondary_stage_odd_response"),
      ("primary float32 Q", "secondary production Q"),
      strict=True,
  ):
    for branch in ("corr", "iid"):
      y = [
          _plot_value(summaries["full"].get(field, {}).get(branch, {}).get(stage, {}).get("mean"))
          if "full" in summaries else np.nan for stage in stage_names
      ]
      axis.plot(stage_names, y, marker="o", label=branch)
    axis.set_title(title)
    axis.grid(alpha=.25)
  axes[0].set_ylabel("mean antithetic odd-response norm")
  axes[-1].legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_stage_cancellation(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  """Plot float32 stage-wise G_C/G_J; invalid zero-D cases remain gaps."""
  figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
  stages = ("linear", "norm", "ns", "scale")
  full = summaries.get("full", {})
  stage_metrics = full.get("stage_metrics", {})
  for axis, gain in zip(axes, ("G_C", "G_J"), strict=True):
    for branch in ("corr", "iid"):
      values = stage_metrics.get(branch, {})
      y = [_plot_value(values.get(stage, {}).get(gain, {}).get("mean")) for stage in stages]
      error = [_plot_value(values.get(stage, {}).get(gain, {}).get("std")) for stage in stages]
      axis.errorbar(stages, y, yerr=error, marker="o", capsize=3, label=branch)
    axis.set_title(gain + " (primary float32)")
    axis.grid(alpha=.25)
  axes[0].set_ylabel("paired cancellation gain")
  axes[-1].legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


__all__ = [
    "plot_bias_diagnostics", "plot_cancellation_paths", "plot_decomposition",
    "plot_gain_over_steps", "plot_path_gain_summary", "plot_stage_cancellation",
    "plot_stage_diagnostics",
]
