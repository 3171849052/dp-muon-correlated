"""Plots required by Experiment 7b."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt


def _key(row: Mapping[str, object], metric: str) -> str:
  return metric if metric in row else f"{metric}_mean"


def _x(rows):
  return [(float(row["start_step"]) + float(row["end_step"])) / 2 for row in rows]


def plot_cancellation(rows: Iterable[Mapping[str, object]], output: str | Path) -> None:
  baseline = [row for row in rows if row["algorithm"] == "baseline"]
  if not baseline:
    return
  figure, axis = plt.subplots(figsize=(7.2, 4.3))
  for field in ("C_00", "C_10", "C_01", "C_11", "C_BC"):
    key = _key(baseline[0], field)
    axis.plot(_x(baseline), [float(row[key]) for row in baseline], marker="o", label=field)
  axis.set(xlabel="training step (window midpoint)", ylabel="16-step cancellation score")
  axis.grid(alpha=.25)
  axis.legend(ncol=3)
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_bc_preconditioner(rows: Iterable[Mapping[str, object]], output: str | Path) -> None:
  materialized = list(rows)
  if not materialized:
    return
  figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
  for algorithm in ("baseline", "bc"):
    subset = [row for row in materialized if row["algorithm"] == algorithm]
    if not subset:
      continue
    top.plot(_x(subset), [float(row[_key(row, "floor_activation_fraction")])
                          for row in subset], marker="o", label=algorithm)
    for metric, style in (("p_bc_median", "-"), ("p_bc_q99", "--"),
                          ("p_bc_q99_9", ":"), ("p_bc_max", "-.")):
      bottom.plot(_x(subset), [float(row[_key(row, metric)]) for row in subset],
                  linestyle=style, label=f"{algorithm} {metric.removeprefix('p_bc_')}")
  top.set(ylabel="floor activation fraction")
  bottom.set(xlabel="training step (window midpoint)", ylabel="BC preconditioner")
  for axis in (top, bottom):
    axis.grid(alpha=.25)
    axis.legend(ncol=2, fontsize=8)
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


def plot_update_norms(rows: Iterable[Mapping[str, object]], output: str | Path) -> None:
  materialized = list(rows)
  if not materialized:
    return
  figure, axis = plt.subplots(figsize=(7.2, 4.3))
  for algorithm in ("baseline", "bc"):
    subset = [row for row in materialized if row["algorithm"] == algorithm]
    if not subset:
      continue
    for metric, style in (("raw_optimizer_update_l2_mean", "--"),
                          ("applied_parameter_update_l2_mean", "-")):
      axis.plot(_x(subset), [float(row[_key(row, metric)]) for row in subset],
                linestyle=style, marker="o",
                label=f"{algorithm} {metric.removesuffix('_l2_mean')}")
  axis.set(xlabel="training step (window midpoint)", ylabel="L2 norm")
  axis.grid(alpha=.25)
  axis.legend(fontsize=8)
  figure.tight_layout()
  figure.savefig(output, dpi=150)
  plt.close(figure)


__all__ = ["plot_bc_preconditioner", "plot_cancellation", "plot_update_norms"]
