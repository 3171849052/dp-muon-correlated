"""Exp10 pooled histogram and paired-statistic plotting helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exp10.core import BRANCHES
from exp10.diagnostics import HISTOGRAM_GROUPS, HISTOGRAM_GROUP_COMPONENTS


def resolve_histogram_artifact(
    histograms: str | Path, *, seed: int | None
) -> Path:
  """Prefer pooled data by default, while honoring an explicit seed request."""
  path = Path(histograms)
  sibling_pooled = path.with_name("pooled_histograms.npz")
  sibling_per_seed = path.with_name("histograms.npz")
  if seed is None and path.name != sibling_pooled.name and sibling_pooled.is_file():
    return sibling_pooled
  if seed is not None and path.name == sibling_pooled.name and sibling_per_seed.is_file():
    return sibling_per_seed
  return path


def _confidence_interval(values: list[float]) -> tuple[float, float, float]:
  array = np.asarray(values, dtype=np.float64)
  if not len(array):
    return 0.0, 0.0, 0.0
  mean = float(np.mean(array))
  if len(array) < 2:
    return mean, mean, mean
  se = float(np.std(array, ddof=1) / np.sqrt(len(array)))
  critical = 1.96
  try:
    from scipy.stats import t as student_t
    critical = float(student_t.ppf(.975, len(array) - 1))
  except (ImportError, ValueError):
    pass
  return mean, mean - critical * se, mean + critical * se


def plot_histograms(
    histograms: str | Path,
    output: str | Path,
    *,
    seed: int | None = None,
    step: int | None = None,
    xscale: str = "linear",
) -> None:
  """Plot grouped MF/IID histograms, pooled by default.

  Histogram counts always come from original linear-domain coordinate values.
  ``xscale='symlog'`` changes only the display axis and is useful when the
  shared linear range spans several orders of magnitude.
  """
  if xscale not in {"linear", "symlog"}:
    raise ValueError("xscale must be 'linear' or 'symlog'")
  import matplotlib.pyplot as plt

  selected_path = resolve_histogram_artifact(histograms, seed=seed)
  with np.load(selected_path, allow_pickle=False) as data:
    steps = np.asarray(data["steps"])
    edges = np.asarray(data["group_bin_edges"])
    frequencies = np.asarray(data["relative_frequency"])
    stored_groups = tuple(str(value) for value in data["group_names"])
    stored_components = np.asarray(data["group_component_names"]).astype(str)
    kind = str(np.asarray(data["format_kind"]).item())
    stored_seeds = None if kind == "pooled" else np.asarray(data["seeds"])
  if not len(steps):
    raise ValueError("histogram artifact contains no checkpoints")

  candidates = np.arange(len(steps))
  if stored_seeds is not None and seed is not None:
    candidates = candidates[stored_seeds[candidates] == int(seed)]
  if step is not None:
    candidates = candidates[steps[candidates] == int(step)]
  if not len(candidates):
    raise ValueError("requested histogram checkpoint is not present")
  index = int(candidates[-1])
  figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), squeeze=False)
  for group_index, group in enumerate(HISTOGRAM_GROUPS):
    axis = axes.flat[group_index]
    stored_group_index = stored_groups.index(group)
    # Groups B/D intentionally have independent ranges from A/C, so the
    # x-coordinates must come from the selected group's own edge vector.
    group_edges = edges[index, stored_group_index]
    centers = (group_edges[:-1] + group_edges[1:]) / 2.0
    components = HISTOGRAM_GROUP_COMPONENTS[group]
    for branch_index, branch in enumerate(BRANCHES):
      for slot, component in enumerate(components):
        label = f"{branch} {component}"
        axis.plot(
            centers,
            frequencies[index, branch_index, stored_group_index, slot],
            label=label,
        )
    axis.set_title(group)
    axis.set_xlabel("coordinate value")
    axis.set_ylabel("relative frequency")
    axis.set_xscale(xscale, **({"linthresh": 1e-6} if xscale == "symlog" else {}))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
  source = "pooled" if kind == "pooled" else f"seed={int(stored_seeds[index])}"
  figure.suptitle(f"Exp10 grouped histograms: {source}, step={int(steps[index])}")
  figure.tight_layout()
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_paired_stage_statistics(
    paired_stage_metrics: str | Path,
    output: str | Path,
) -> None:
  """Plot paired MF-IID deltas with cross-seed 95% confidence intervals."""
  import matplotlib.pyplot as plt

  rows = []
  with Path(paired_stage_metrics).open(encoding="utf-8", newline="") as stream:
    rows.extend(csv.DictReader(stream))
  if not rows:
    raise ValueError("paired stage metrics contain no rows")
  order = {"early": 0, "late": 1, "full": 2}
  stages = sorted({str(row["stage"]) for row in rows}, key=lambda value: order.get(value, 99))
  fields = ("delta_feedback", "delta_traj", "delta_noise")
  figure, axis = plt.subplots(figsize=(8.5, 4.8))
  x = np.arange(len(stages))
  offsets = (-.18, 0.0, .18)
  for field, offset in zip(fields, offsets, strict=True):
    means, lows, highs = [], [], []
    for stage in stages:
      values = [float(row[field]) for row in rows if row["stage"] == stage]
      mean, low, high = _confidence_interval(values)
      means.append(mean)
      lows.append(mean - low)
      highs.append(high - mean)
    axis.errorbar(
        x + offset,
        means,
        yerr=np.asarray([lows, highs]),
        marker="o",
        capsize=3,
        label=field,
    )
  axis.axhline(0.0, color="black", linewidth=.8)
  axis.set_xticks(x, stages)
  axis.set_ylabel("MF - IID paired stage mean")
  axis.set_title("Exp10 paired MF-vs-IID statistics (95% CI)")
  axis.grid(alpha=0.25)
  axis.legend()
  figure.tight_layout()
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)


__all__ = [
    "plot_histograms",
    "plot_paired_stage_statistics",
    "resolve_histogram_artifact",
]
