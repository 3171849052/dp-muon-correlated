"""Diagnostic plots for Experiment 7."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt


def plot_cancellation(rows: Iterable[Mapping[str, object]], output: str | Path) -> None:
  baseline = [row for row in rows if row["algorithm"] == "baseline"]
  if not baseline:
    return
  figure, axis = plt.subplots(figsize=(7.2, 4.3))
  x = [(float(row["start_step"]) + float(row["end_step"])) / 2 for row in baseline]
  for field in ("C_00", "C_10", "C_01", "C_11", "C_BC"):
    key = field if field in baseline[0] else f"{field}_mean"
    axis.plot(x, [float(row[key]) for row in baseline], marker="o", label=field)
  axis.set(xlabel="training step (window midpoint)", ylabel="16-step cancellation score")
  axis.grid(alpha=.25)
  axis.legend(ncol=3)
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_paired_gaps(rows: Iterable[Mapping[str, object]], output: str | Path) -> None:
  materialized = list(rows)
  if not materialized:
    return
  figure, axis = plt.subplots(figsize=(7.0, 4.1))
  for algorithm in ("baseline", "bc"):
    subset = [row for row in materialized if row["algorithm"] == algorithm]
    if subset:
      key = "gap" if "gap" in subset[0] else "gap_mean"
      axis.plot(
          [(float(row["start_step"]) + float(row["end_step"])) / 2 for row in subset],
          [float(row[key]) for row in subset], marker="o", label=algorithm,
      )
  axis.set(xlabel="training step (window midpoint)", ylabel="C_real - C_dynamic_clean_p")
  axis.grid(alpha=.25)
  axis.legend()
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


__all__ = ["plot_cancellation", "plot_paired_gaps"]
