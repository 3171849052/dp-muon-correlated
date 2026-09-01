"""Plotting helpers for the compact Exp10 histogram artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from exp10.core import BRANCHES, COMPONENTS


def plot_histograms(
    histograms: str | Path,
    output: str | Path,
    *,
    seed: int | None = None,
    step: int | None = None,
) -> None:
  """Plot one checkpoint's MF/IID relative-frequency histograms.

  ``histograms.npz`` stores one shared set of edges per ``(seed, step)``.
  The helper selects the requested checkpoint (or the last one) and never
  reconstructs or requires full parameter-coordinate arrays.
  """
  import matplotlib.pyplot as plt

  with np.load(histograms, allow_pickle=False) as data:
    seeds = np.asarray(data["seeds"])
    steps = np.asarray(data["steps"])
    edges = np.asarray(data["bin_edges"])
    frequencies = np.asarray(data["relative_frequency"])
    stored_branches = tuple(str(value) for value in data["branch_names"])
    stored_components = tuple(str(value) for value in data["component_names"])
  if not len(steps):
    raise ValueError("histogram artifact contains no checkpoints")
  candidates = np.arange(len(steps))
  if seed is not None:
    candidates = candidates[seeds[candidates] == int(seed)]
  if step is not None:
    candidates = candidates[steps[candidates] == int(step)]
  if not len(candidates):
    raise ValueError("requested histogram checkpoint is not present")
  index = int(candidates[-1])
  edge = edges[index]
  centers = (edge[:-1] + edge[1:]) / 2.0

  figure, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), squeeze=False)
  for component_index, component in enumerate(COMPONENTS):
    axis = axes.flat[component_index]
    stored_component_index = stored_components.index(component)
    for branch in BRANCHES:
      stored_branch_index = stored_branches.index(branch)
      axis.plot(
          centers,
          frequencies[index, stored_branch_index, stored_component_index],
          label=branch,
      )
    axis.set_title(component)
    axis.set_xlabel("coordinate value")
    axis.set_ylabel("relative frequency")
    axis.grid(alpha=0.25)
    axis.legend()
  figure.suptitle(f"Exp10 histograms: seed={int(seeds[index])}, step={int(steps[index])}")
  figure.tight_layout()
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)


__all__ = ["plot_histograms"]
