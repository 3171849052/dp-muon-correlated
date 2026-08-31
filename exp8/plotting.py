"""Small plots for the four-layer Exp8 mechanism diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import matplotlib.pyplot as plt

from exp8.core import PATHS


def _mean_window_values(rows: Iterable[Mapping[str, object]], field: str):
  groups: dict[int, list[Mapping[str, object]]] = {}
  for row in rows:
    groups.setdefault(int(row["window_index"]), []).append(row)
  output = []
  for index, group in sorted(groups.items()):
    x = np.mean([(float(row["start_step"]) + float(row["end_step"])) / 2 for row in group])
    y = np.mean([float(row[field]) for row in group])
    output.append((x, y))
  return output


def plot_gain_over_steps(
    rows: Iterable[Mapping[str, object]], output: str | Path, *, gain: str
) -> None:
  rows = list(rows)
  figure, axis = plt.subplots(figsize=(7.2, 4.3))
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


def plot_path_gain_summary(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  """Plot cross-seed stage means with one-standard-deviation error bars."""
  figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
  stages = ("early", "late", "full")
  labels = ("G0", "Gc", "Gphi", "Gp")
  for axis, gain in zip(axes, ("G_C", "G_J"), strict=True):
    for stage in stages:
      if stage not in summaries or "paths" not in summaries[stage]:
        continue
      values = summaries[stage]["paths"]  # type: ignore[index]
      y = [float(values[path][f"{gain}_mean"]) for path in PATHS]
      error = [float(values[path][f"{gain}_std"]) for path in PATHS]
      axis.errorbar(range(4), y, yerr=error, marker="o", capsize=3, label=stage)
    axis.set_xticks(range(4), labels)
    axis.set_title(gain)
    axis.grid(alpha=.25)
  axes[0].set_ylabel("gain")
  axes[-1].legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_decomposition(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], output: str | Path
) -> None:
  """Plot cross-seed decomposition means with standard-deviation bars."""
  figure, axis = plt.subplots(figsize=(7.2, 4.2))
  stages = ("early", "late", "full")
  x = np.arange(len(stages))
  for field, label in (("A_energy", "A"), ("B_energy", "B"), ("I_energy", "I")):
    values = [
        float(summaries[stage]["decomposition_flat"][field + "_mean"])
        if stage in summaries and "decomposition_flat" in summaries[stage] else 0.0
        for stage in stages
    ]
    errors = [
        float(summaries[stage]["decomposition_flat"][field + "_std"])
        if stage in summaries and "decomposition_flat" in summaries[stage] else 0.0
        for stage in stages
    ]
    axis.errorbar(x, values, yerr=errors, marker="o", capsize=3, label=label)
  axis.set_xticks(x, stages)
  axis.set_ylabel("sum squared energy (correlated branch)")
  axis.grid(alpha=.25)
  axis.legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


__all__ = ["plot_decomposition", "plot_gain_over_steps", "plot_path_gain_summary"]
