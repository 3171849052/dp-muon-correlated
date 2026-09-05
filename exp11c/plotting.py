"""Artifacts and plotting helpers for Exp11c."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORIES = ("clean", "eps3", "eps8")
PAIR_NAMES = ("clean_eps3", "clean_eps8", "eps3_eps8")
TARGET_BLOCKS = (0, 5, 11)
ARTIFACT_KEYS = (
    "format_version",
    "epsilons",
    "steps",
    "layers",
    "trajectories",
    "pair_names",
    "noise_stds",
    "matrix_frobenius_norms",
    "matrix_scale_ratios",
    "ideal_q_pairwise_frobenius_distances",
    "ideal_q_pairwise_cosines",
)


def _validate_arrays(
    epsilons: Any,
    steps: Any,
    layers: Any,
    trajectories: Any,
    pair_names: Any,
    noise_stds: Any,
    matrix_frobenius_norms: Any,
    matrix_scale_ratios: Any,
    q_distances: Any,
    q_cosines: Any,
) -> dict[str, np.ndarray]:
  values = {
      "epsilons": np.asarray(epsilons, dtype=np.int32),
      "steps": np.asarray(steps, dtype=np.int32),
      "layers": np.asarray(layers),
      "trajectories": np.asarray(trajectories),
      "pair_names": np.asarray(pair_names),
      "noise_stds": np.asarray(noise_stds, dtype=np.float64),
      "matrix_frobenius_norms": np.asarray(matrix_frobenius_norms, dtype=np.float64),
      "matrix_scale_ratios": np.asarray(matrix_scale_ratios, dtype=np.float64),
      "ideal_q_pairwise_frobenius_distances": np.asarray(q_distances, dtype=np.float64),
      "ideal_q_pairwise_cosines": np.asarray(q_cosines, dtype=np.float64),
  }
  if tuple(values["epsilons"].tolist()) != (3, 8):
    raise ValueError("Exp11c must contain epsilon=3 and epsilon=8 in that order")
  if values["steps"].ndim != 1 or not len(values["steps"]):
    raise ValueError("Exp11c requires at least one recording step")
  if np.any(np.diff(values["steps"]) <= 0):
    raise ValueError("Exp11c recording steps must be strictly increasing")
  if tuple(values["trajectories"].tolist()) != TRAJECTORIES:
    raise ValueError("Exp11c trajectories must be clean, eps3, eps8")
  if tuple(values["pair_names"].tolist()) != PAIR_NAMES:
    raise ValueError("Exp11c pair names are not in the required order")
  if values["layers"].ndim != 1 or len(values["layers"]) != len(TARGET_BLOCKS):
    raise ValueError("Exp11c requires exactly three target layers")
  if values["noise_stds"].shape != (3,):
    raise ValueError("noise_stds must have shape [trajectory]")
  expected = (3, len(values["steps"]), len(values["layers"]))
  if values["matrix_frobenius_norms"].shape != expected:
    raise ValueError("matrix_frobenius_norms must have shape [trajectory, step, layer]")
  if values["matrix_scale_ratios"].shape != expected[1:]:
    raise ValueError("matrix_scale_ratios must have shape [step, layer]")
  pair_expected = (len(values["steps"]), len(values["layers"]), len(PAIR_NAMES))
  for name in (
      "ideal_q_pairwise_frobenius_distances",
      "ideal_q_pairwise_cosines",
  ):
    if values[name].shape != pair_expected:
      raise ValueError(f"{name} must have shape [step, layer, pair]")
  for name in (
      "noise_stds",
      "matrix_frobenius_norms",
      "matrix_scale_ratios",
      "ideal_q_pairwise_frobenius_distances",
      "ideal_q_pairwise_cosines",
  ):
    if not np.all(np.isfinite(values[name])):
      raise ValueError(f"{name} must contain only finite values")
  if np.any(values["noise_stds"] < 0) or np.any(values["matrix_frobenius_norms"] < 0):
    raise ValueError("noise and matrix norms must be non-negative")
  if np.any(values["ideal_q_pairwise_frobenius_distances"] < 0):
    raise ValueError("ideal-Q distances must be non-negative")
  if np.any(values["ideal_q_pairwise_cosines"] < -1.000001) or np.any(
      values["ideal_q_pairwise_cosines"] > 1.000001
  ):
    raise ValueError("ideal-Q cosine similarities must lie in [-1, 1]")
  return values


def save_scale_blindness(output: str | Path, *, result: Any) -> Path:
  """Write metrics only; no pre-Q or ideal-Q matrix is persisted."""
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  values = _validate_arrays(
      result.epsilons,
      result.steps,
      np.asarray(result.layers),
      np.asarray(result.trajectories),
      np.asarray(result.pair_names),
      result.noise_stds,
      result.matrix_frobenius_norms,
      result.matrix_scale_ratios,
      result.ideal_q_pairwise_frobenius_distances,
      result.ideal_q_pairwise_cosines,
  )
  np.savez(
      output,
      format_version=np.asarray("exp11c-scale-blindness-v1"),
      **values,
  )
  return output


def load_scale_blindness(path: str | Path) -> dict[str, np.ndarray]:
  """Load and validate the compact Exp11c metrics artifact."""
  with np.load(path, allow_pickle=False) as data:
    missing = set(ARTIFACT_KEYS).difference(data.files)
    if missing:
      raise ValueError(f"scale-blindness artifact is missing keys: {sorted(missing)}")
    result = {key: np.asarray(data[key]) for key in ARTIFACT_KEYS}
  if str(result["format_version"].item()) != "exp11c-scale-blindness-v1":
    raise ValueError("unsupported Exp11c artifact version")
  _validate_arrays(
      result["epsilons"], result["steps"], result["layers"],
      result["trajectories"], result["pair_names"], result["noise_stds"],
      result["matrix_frobenius_norms"], result["matrix_scale_ratios"],
      result["ideal_q_pairwise_frobenius_distances"],
      result["ideal_q_pairwise_cosines"],
  )
  return result


def save_scale_blindness_csv(scalars: str | Path, output: str | Path) -> Path:
  """Export one compact row per step/layer, without matrix coordinates."""
  values = load_scale_blindness(scalars)
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow((
        "step", "layer", "clean_norm", "eps3_norm", "eps8_norm",
        "eps3_over_eps8_norm", "q_distance_clean_eps3",
        "q_distance_clean_eps8", "q_distance_eps3_eps8",
        "q_cosine_clean_eps3", "q_cosine_clean_eps8", "q_cosine_eps3_eps8",
    ))
    norms = values["matrix_frobenius_norms"]
    ratios = values["matrix_scale_ratios"]
    distances = values["ideal_q_pairwise_frobenius_distances"]
    cosines = values["ideal_q_pairwise_cosines"]
    for step_index, step in enumerate(values["steps"]):
      for layer_index, layer in enumerate(values["layers"]):
        writer.writerow((
            int(step), str(layer),
            *(format(float(norms[trajectory, step_index, layer_index]), ".17e")
              for trajectory in range(3)),
            format(float(ratios[step_index, layer_index]), ".17e"),
            *(format(float(distances[step_index, layer_index, pair]), ".17e")
              for pair in range(3)),
            *(format(float(cosines[step_index, layer_index, pair]), ".17e")
              for pair in range(3)),
        ))
  return output


def plot_scale_blindness(scalars: str | Path, output: str | Path) -> Path:
  """Plot matrix scale and ideal-Q disagreement at all three target layers."""
  import matplotlib.pyplot as plt

  values = load_scale_blindness(scalars)
  figure, axes = plt.subplots(2, 3, figsize=(15.0, 7.8), squeeze=False)
  steps = values["steps"]
  x = np.arange(len(steps))
  for layer_index, block in enumerate(TARGET_BLOCKS):
    norm_axis = axes[0, layer_index]
    for trajectory_index, trajectory in enumerate(TRAJECTORIES):
      norm_axis.plot(
          x, values["matrix_frobenius_norms"][trajectory_index, :, layer_index],
          marker="o", label=trajectory,
      )
    norm_axis.set_title(f"block={block}: pre-Q matrix norm")
    norm_axis.set_xticks(x, [str(int(step)) for step in steps])
    norm_axis.set_xlabel("step")
    norm_axis.set_ylabel("Frobenius norm")
    norm_axis.grid(alpha=0.25)
    norm_axis.legend()

    q_axis = axes[1, layer_index]
    for pair_index, pair in enumerate(PAIR_NAMES):
      q_axis.plot(
          x,
          values["ideal_q_pairwise_frobenius_distances"][:, layer_index, pair_index],
          marker="o", label=pair,
      )
    q_axis.set_title(f"block={block}: ideal-Q disagreement")
    q_axis.set_xticks(x, [str(int(step)) for step in steps])
    q_axis.set_xlabel("step")
    q_axis.set_ylabel(r"$\|Q_i-Q_j\|_F$")
    q_axis.grid(alpha=0.25)
    q_axis.legend(fontsize=8)
  figure.suptitle("Exp11c ideal Muon scale-blindness diagnostics")
  figure.tight_layout()
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=150)
  plt.close(figure)
  return output


__all__ = [
    "ARTIFACT_KEYS",
    "PAIR_NAMES",
    "TRAJECTORIES",
    "load_scale_blindness",
    "plot_scale_blindness",
    "save_scale_blindness",
    "save_scale_blindness_csv",
]
