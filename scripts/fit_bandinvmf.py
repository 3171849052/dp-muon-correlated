#!/usr/bin/env python3
"""Fit and save a public prefix-sum BandInvMF strategy artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from dp_muon.bandinvmf import fit_bandinv_strategy


def default_artifact_path(
    root: Path,
    *,
    horizon: int,
    bandwidth: int,
    min_sep: int,
    max_participations: int | None,
) -> Path:
  """Returns a stable name that identifies all public strategy parameters.

  ``p`` is the number of optimized Toeplitz coefficients (the API's
  ``bandwidth``); ``b`` is the minimum separation; and ``k`` is the maximum
  number of participations. ``kmax`` unambiguously represents no cap.
  """
  k = "max" if max_participations is None else str(max_participations)
  return root / "artifacts" / "strategies" / (
      f"prefix_n{horizon}_p{bandwidth}_b{min_sep}_k{k}.npz"
  )


def jax_privacy_version() -> str:
  """Returns the available installed-package version without modifying it."""
  for distribution in ("jax-privacy", "jax_privacy"):
    try:
      dist = importlib.metadata.distribution(distribution)
      name = dist.metadata.get("Name", distribution)
      return f"{name} {dist.version}"
    except importlib.metadata.PackageNotFoundError:
      continue
  try:
    import jax_privacy

    return str(getattr(jax_privacy, "__version__", "version unavailable"))
  except ImportError:
    return "version unavailable"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--horizon", type=int, required=True)
  parser.add_argument("--bandwidth", type=int, required=True)
  parser.add_argument("--min-sep", type=int, required=True)
  parser.add_argument("--max-participations", type=int)
  parser.add_argument("--max-optimizer-steps", type=int, default=1000)
  parser.add_argument("--reduction", choices=("mean", "max", "last"), default="mean")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  result = fit_bandinv_strategy(
      args.horizon,
      args.bandwidth,
      args.min_sep,
      max_participations=args.max_participations,
      max_optimizer_steps=args.max_optimizer_steps,
      reduction=args.reduction,
  )
  output = args.output or default_artifact_path(
      ROOT,
      horizon=args.horizon,
      bandwidth=args.bandwidth,
      min_sep=args.min_sep,
      max_participations=args.max_participations,
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      output,
      horizon=np.asarray(result.horizon),
      bandwidth=np.asarray(result.bandwidth),
      min_sep=np.asarray(result.min_sep),
      max_participations=np.asarray(-1 if result.max_participations is None else result.max_participations),
      workload_coef=np.asarray(result.workload_coef),
      noising_coef=np.asarray(result.noising_coef),
      strategy_coef=np.asarray(result.strategy_coef),
      sensitivity_squared=np.asarray(result.sensitivity_squared),
      objective=np.asarray(result.objective),
      reduction=np.asarray(args.reduction),
      max_optimizer_steps=np.asarray(args.max_optimizer_steps),
      jax_privacy_version=np.asarray(jax_privacy_version()),
  )
  print(f"saved strategy artifact: {output}")


if __name__ == "__main__":
  main()
