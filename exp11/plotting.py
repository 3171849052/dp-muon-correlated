"""Plotting and artifact helpers for Experiment 11."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SPECTRA_KEYS = (
    "steps",
    "parameter_name",
    "clean_singular_values",
    "dp_singular_values",
)
_PANEL_LABELS = ("Early", "Middle", "Late")


def _as_spectra_arrays(
    steps: np.ndarray,
    clean_singular_values: np.ndarray,
    dp_singular_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  steps = np.asarray(steps, dtype=np.int32)
  clean = np.asarray(clean_singular_values, dtype=np.float64)
  dp = np.asarray(dp_singular_values, dtype=np.float64)
  if steps.ndim != 1 or len(steps) != 3:
    raise ValueError("Exp11 requires exactly three spectrum checkpoints")
  if clean.ndim != 2 or dp.ndim != 2 or clean.shape != dp.shape:
    raise ValueError("clean and DP spectra must be matching rank-two arrays")
  if clean.shape[0] != len(steps) or clean.shape[1] < 1:
    raise ValueError("spectrum rows must match the three checkpoint steps")
  if np.any(np.diff(steps) <= 0):
    raise ValueError("spectrum steps must be strictly increasing")
  if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(dp)):
    raise ValueError("spectra must contain only finite values")
  if np.any(clean < 0) or np.any(dp < 0):
    raise ValueError("singular values must be non-negative")
  return steps, clean, dp


def save_spectra(
    output: str | Path,
    *,
    steps: np.ndarray,
    parameter_name: str,
    clean_singular_values: np.ndarray,
    dp_singular_values: np.ndarray,
) -> Path:
  """Writes the small Exp11 NPZ schema, containing spectra but no matrices."""
  if not isinstance(parameter_name, str) or not parameter_name:
    raise ValueError("parameter_name must be a non-empty string")
  steps, clean, dp = _as_spectra_arrays(
      steps, clean_singular_values, dp_singular_values
  )
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      output,
      steps=steps,
      parameter_name=np.asarray(parameter_name),
      clean_singular_values=clean,
      dp_singular_values=dp,
  )
  return output


def load_spectra(path: str | Path) -> dict[str, np.ndarray]:
  """Loads and validates an Exp11 spectrum artifact."""
  with np.load(path, allow_pickle=False) as data:
    missing = set(SPECTRA_KEYS).difference(data.files)
    if missing:
      raise ValueError(f"spectra artifact is missing keys: {sorted(missing)}")
    result = {key: np.asarray(data[key]) for key in SPECTRA_KEYS}
  _as_spectra_arrays(
      result["steps"], result["clean_singular_values"], result["dp_singular_values"]
  )
  if result["parameter_name"].ndim != 0:
    raise ValueError("parameter_name must be a scalar string")
  return result


def plot_singular_spectra(
    spectra: str | Path,
    output: str | Path,
) -> Path:
  """Plots Early/Middle/Late clean and IID-DP spectra on shared log y axes."""
  import matplotlib.pyplot as plt

  values = load_spectra(spectra)
  steps = values["steps"].astype(int)
  clean = values["clean_singular_values"].astype(np.float64)
  dp = values["dp_singular_values"].astype(np.float64)
  positive = np.concatenate((clean[clean > 0], dp[dp > 0]))
  if not len(positive):
    raise ValueError("cannot use a log scale for an all-zero spectrum")

  figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
  x = np.arange(1, clean.shape[1] + 1)
  lower = max(float(np.min(positive)) * 0.8, np.finfo(np.float64).tiny)
  upper = float(np.max(positive)) * 1.25
  if upper <= lower:
    upper = lower * 10.0
  for index, (axis, panel) in enumerate(zip(axes, _PANEL_LABELS, strict=True)):
    axis.plot(x, clean[index], label="Clean")
    axis.plot(x, dp[index], label="IID DP")
    axis.set_title(f"{panel}: step {int(steps[index])}")
    axis.set_xlabel("Singular value index")
    axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    axis.grid(alpha=0.25)
    axis.legend()
  axes[0].set_ylabel("Singular value")
  figure.tight_layout()
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)
  return output


__all__ = ["SPECTRA_KEYS", "load_spectra", "plot_singular_spectra", "save_spectra"]
