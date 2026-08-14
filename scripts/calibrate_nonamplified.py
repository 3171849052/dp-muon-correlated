#!/usr/bin/env python3
"""Calibrate full-transcript iid Gaussian noise from a BandInvMF artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dp_muon.privacy import calibrate_nonamplified_bandinv


def load_sensitivity_squared(artifact_path: Path) -> float:
  """Loads the required matrix sensitivity value from a strategy artifact."""
  try:
    with np.load(artifact_path, allow_pickle=False) as artifact:
      if "sensitivity_squared" not in artifact:
        raise ValueError("strategy artifact has no sensitivity_squared field")
      return float(artifact["sensitivity_squared"])
  except OSError as error:
    raise ValueError(f"could not read strategy artifact: {artifact_path}") from error


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("strategy_artifact", type=Path)
  parser.add_argument("--epsilon", type=float, required=True)
  parser.add_argument("--delta", type=float, required=True)
  parser.add_argument("--clip-norm", type=float, required=True)
  parser.add_argument("--normalize-by", type=float, required=True)
  parser.add_argument(
      "--adjacency", choices=("add_remove", "replace_one"), required=True
  )
  args = parser.parse_args()

  sensitivity_squared = load_sensitivity_squared(args.strategy_artifact)
  result = calibrate_nonamplified_bandinv(
      epsilon=args.epsilon,
      delta=args.delta,
      clip_norm=args.clip_norm,
      normalize_by=args.normalize_by,
      adjacency=args.adjacency,
      sensitivity_squared=sensitivity_squared,
  )
  for field in (
      "epsilon",
      "delta",
      "adjacency",
      "clip_norm",
      "normalize_by",
      "matrix_sensitivity",
      "query_sensitivity",
      "total_sensitivity",
      "mu",
      "noise_multiplier",
      "iid_noise_std",
  ):
    print(f"{field}: {getattr(result, field)}")


if __name__ == "__main__":
  main()
