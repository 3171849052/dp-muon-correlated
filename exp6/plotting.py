"""Small mechanism-diagnostic plots for Experiment 6."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import correlation_from_rows


def plot_over_steps(
    summary_rows: Iterable[Mapping[str, float | int]], output: str | Path
) -> None:
  """Plot the window-average preconditioner change and cancellation delta."""
  rows = list(summary_rows)
  if not rows:
    return
  x = np.asarray(
      [(float(row["start_step"]) + float(row["end_step"])) / 2.0 for row in rows]
  )
  p_change = np.asarray([float(row["mean_p_relative_change_mean"]) for row in rows])
  delta = np.asarray([float(row["delta_p_cancellation_mean"]) for row in rows])
  figure, left = plt.subplots(figsize=(7.0, 4.0))
  right = left.twinx()
  first = left.plot(x, p_change, "o-", color="tab:blue", label="mean p relative change")
  second = right.plot(x, delta, "s-", color="tab:orange", label="delta p cancellation")
  left.set_xlabel("training step (window midpoint)")
  left.set_ylabel("mean_p_relative_change", color="tab:blue")
  right.set_ylabel("C_dynamic_clean_p - C_frozen_p", color="tab:orange")
  left.grid(alpha=0.25)
  left.legend(first + second, [line.get_label() for line in first + second], loc="best")
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_scatter(
    rows: Iterable[Mapping[str, float | int]], output: str | Path
) -> float:
  """Plot all seed/window points and return the pooled Spearman rho."""
  rows = list(rows)
  rho = correlation_from_rows(rows)
  figure, axis = plt.subplots(figsize=(6.0, 4.5))
  seeds = sorted({int(row["seed"]) for row in rows})
  for seed in seeds:
    subset = [row for row in rows if int(row["seed"]) == seed]
    axis.scatter(
        [float(row["mean_p_relative_change"]) for row in subset],
        [float(row["delta_p_cancellation"]) for row in subset],
        s=24,
        alpha=0.8,
        label=f"seed {seed}",
    )
  axis.set_xlabel("mean_p_relative_change")
  axis.set_ylabel("delta_p_cancellation")
  axis.set_title(f"p change vs cancellation delta (Spearman rho={rho:.4g})")
  if seeds:
    axis.legend()
  axis.grid(alpha=0.25)
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)
  return rho


__all__ = ["plot_over_steps", "plot_scatter"]
