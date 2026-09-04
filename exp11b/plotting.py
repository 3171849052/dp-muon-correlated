"""Artifact and plotting helpers for Exp11b."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


SPECTRA_KEYS = (
    "epsilons",
    "steps",
    "layers",
    "clean_singular_values",
    "dp_singular_values",
)
TARGET_EPSILONS = (3, 8)
TARGET_BLOCKS = (0, 5, 11)


def _validate_arrays(
    epsilons: np.ndarray,
    steps: np.ndarray,
    layers: np.ndarray,
    clean: np.ndarray,
    dp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  epsilons = np.asarray(epsilons, dtype=np.int32)
  steps = np.asarray(steps, dtype=np.int32)
  layers = np.asarray(layers)
  clean = np.asarray(clean, dtype=np.float64)
  dp = np.asarray(dp, dtype=np.float64)
  if epsilons.ndim != 1 or tuple(epsilons.tolist()) != TARGET_EPSILONS:
    raise ValueError("Exp11b must contain epsilon=3 and epsilon=8 in that order")
  if steps.ndim != 1 or len(steps) != 3 or np.any(np.diff(steps) <= 0):
    raise ValueError("Exp11b requires three strictly increasing steps")
  if layers.ndim != 1 or len(layers) != 3:
    raise ValueError("Exp11b requires exactly three target layers")
  if any(not isinstance(layer, str) or not layer for layer in layers.tolist()):
    raise ValueError("layer names must be non-empty strings")
  if clean.ndim != 4 or dp.ndim != 4 or clean.shape != dp.shape:
    raise ValueError(
        "clean and DP spectra must be matching [epsilon, step, layer, index] arrays"
    )
  if clean.shape[:3] != (len(epsilons), len(steps), len(layers)):
    raise ValueError("spectrum dimensions do not match epsilon, step, and layer axes")
  if clean.shape[-1] < 1:
    raise ValueError("spectrum index axis must be non-empty")
  if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(dp)):
    raise ValueError("spectra must contain only finite values")
  if np.any(clean < 0) or np.any(dp < 0):
    raise ValueError("singular values must be non-negative")
  if np.any(np.diff(clean, axis=-1) > 0) or np.any(np.diff(dp, axis=-1) > 0):
    raise ValueError("singular values must be in descending order")
  return epsilons, steps, layers, clean, dp


def save_spectra(output: str | Path, *, result: Any) -> Path:
  """Write the only numerical artifact used by both CSV and plots."""
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  epsilons, steps, layers, clean, dp = _validate_arrays(
      result.epsilons,
      result.steps,
      np.asarray(result.layers),
      result.clean_singular_values,
      result.dp_singular_values,
  )
  np.savez(
      output,
      epsilons=epsilons,
      steps=steps,
      layers=layers,
      clean_singular_values=clean,
      dp_singular_values=dp,
  )
  return output


def load_spectra(path: str | Path) -> dict[str, np.ndarray]:
  """Load and validate an Exp11b NPZ artifact."""
  with np.load(path, allow_pickle=False) as data:
    missing = set(SPECTRA_KEYS).difference(data.files)
    if missing:
      raise ValueError(f"spectra artifact is missing keys: {sorted(missing)}")
    result = {key: np.asarray(data[key]) for key in SPECTRA_KEYS}
  _validate_arrays(
      result["epsilons"], result["steps"], result["layers"],
      result["clean_singular_values"], result["dp_singular_values"],
  )
  return result


def save_spectra_csv(spectra: str | Path, output: str | Path) -> Path:
  """Export the validated NPZ arrays as the requested long-form table."""
  values = load_spectra(spectra)
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("epsilon", "step", "layer", "index", "clean", "dp"))
    for epsilon_index, epsilon in enumerate(values["epsilons"]):
      for step_index, step in enumerate(values["steps"]):
        for layer_index, layer in enumerate(values["layers"]):
          clean = values["clean_singular_values"][epsilon_index, step_index, layer_index]
          dp = values["dp_singular_values"][epsilon_index, step_index, layer_index]
          for singular_index, (clean_value, dp_value) in enumerate(
              zip(clean, dp, strict=True), start=1
          ):
            writer.writerow((
                int(epsilon), int(step), layer, singular_index,
                format(float(clean_value), ".17e"),
                format(float(dp_value), ".17e"),
            ))
  return output


def plot_singular_spectra(
    spectra: str | Path,
    output: str | Path,
    *,
    epsilon: int,
) -> Path:
  """Plot one epsilon as a 3x3 layer-by-step grid with shared log y limits."""
  import matplotlib.pyplot as plt

  values = load_spectra(spectra)
  matches = np.flatnonzero(values["epsilons"] == int(epsilon))
  if len(matches) != 1:
    raise ValueError(f"epsilon={epsilon} is not present exactly once")
  epsilon_index = int(matches[0])
  clean = values["clean_singular_values"][epsilon_index]
  dp = values["dp_singular_values"][epsilon_index]
  positive = np.concatenate((clean[clean > 0], dp[dp > 0]))
  if not len(positive):
    raise ValueError("cannot use a log scale for an all-zero spectrum")
  lower = max(float(np.min(positive)) * 0.8, np.finfo(np.float64).tiny)
  upper = float(np.max(positive)) * 1.25
  if upper <= lower:
    upper = lower * 10.0

  figure, axes = plt.subplots(3, 3, figsize=(15.2, 11.5), sharex=True, sharey=True)
  x = np.arange(1, clean.shape[-1] + 1)
  for layer_index, block in enumerate(TARGET_BLOCKS):
    for step_index, step in enumerate(values["steps"]):
      axis = axes[layer_index, step_index]
      axis.plot(x, clean[step_index, layer_index], label="Clean")
      axis.plot(x, dp[step_index, layer_index], label="IID DP")
      axis.set_title(
          f"layer={values['layers'][layer_index]}\n"
          f"step={int(step)}, epsilon={int(epsilon)}"
      )
      axis.set_xlim(1, clean.shape[-1])
      axis.set_yscale("log")
      axis.set_ylim(lower, upper)
      axis.grid(alpha=0.25)
      if layer_index == 2:
        axis.set_xlabel("Singular value index")
      if step_index == 0:
        axis.set_ylabel("Singular value")
  axes[0, 0].legend()
  figure.suptitle(f"Exp11b pre-Q singular spectra (epsilon={int(epsilon)})")
  figure.tight_layout()
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)
  return output


__all__ = [
    "SPECTRA_KEYS",
    "load_spectra",
    "plot_singular_spectra",
    "save_spectra",
    "save_spectra_csv",
]
