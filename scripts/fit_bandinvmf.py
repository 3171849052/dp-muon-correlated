#!/usr/bin/env python3
"""Fit and save a public BandInvMF strategy artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from dp_muon.bandinvmf import fit_bandinv_strategy, load_bandinv_strategy
from dp_muon.optim import fixed_lr_nesterov_trajectory_workload_coef
from dp_muon.training.file_locking import atomic_replace, atomic_temporary_path, file_lock
from dp_muon.bandinvmf import save_bandinv_strategy


def default_artifact_path(
    root: Path,
    *,
    horizon: int,
    bandwidth: int,
    min_sep: int,
    max_participations: int | None,
    workload: str = "prefix",
    momentum: float | None = None,
    learning_rate: float | None = None,
) -> Path:
  """Returns a stable name that identifies all public strategy parameters.

  ``p`` is the number of optimized Toeplitz coefficients (the API's
  ``bandwidth``); ``b`` is the minimum separation; and ``k`` is the maximum
  number of participations. ``kmax`` unambiguously represents no cap.
  """
  k = "max" if max_participations is None else str(max_participations)
  if workload == "prefix":
    name = f"prefix_n{horizon}_p{bandwidth}_b{min_sep}_k{k}.npz"
  else:
    name = (
        f"nesterov-trajectory_n{horizon}_p{bandwidth}_b{min_sep}_k{k}"
        f"_m{momentum}_lr{learning_rate}.npz"
    )
  return root / "artifacts" / "strategies" / name


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


def publish_strategy_artifact(
    output: Path,
    strategy,
    *,
    reduction: str,
    workload_type: str,
    momentum: float | None,
    learning_rate: float | None,
    max_optimizer_steps: int,
) -> Path:
  """Publishes a manually fitted strategy through the shared lock protocol."""
  with file_lock(output):
    with atomic_temporary_path(output) as temporary:
      save_bandinv_strategy(
          temporary, strategy, reduction=reduction, workload_type=workload_type,
          momentum=momentum, learning_rate=learning_rate,
          max_optimizer_steps=max_optimizer_steps,
      )
      load_bandinv_strategy(temporary)
      atomic_replace(temporary, output)
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--horizon", type=int, required=True)
  parser.add_argument("--bandwidth", type=int, required=True)
  parser.add_argument("--min-sep", type=int, required=True)
  parser.add_argument("--max-participations", type=int)
  parser.add_argument("--max-optimizer-steps", type=int, default=1000)
  parser.add_argument("--reduction", choices=("mean", "max", "last"), default="mean")
  parser.add_argument("--workload", choices=("prefix", "nesterov-trajectory"), default="prefix")
  parser.add_argument("--momentum", type=float)
  parser.add_argument("--learning-rate", type=float)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if args.workload == "nesterov-trajectory":
    if args.momentum is None:
      parser.error("--momentum is required for --workload nesterov-trajectory")
    if args.learning_rate is None:
      parser.error("--learning-rate is required for --workload nesterov-trajectory")
    workload_coef = fixed_lr_nesterov_trajectory_workload_coef(
        args.horizon, args.momentum, args.learning_rate
    )
  else:
    workload_coef = None

  result = fit_bandinv_strategy(
      args.horizon,
      args.bandwidth,
      args.min_sep,
      max_participations=args.max_participations,
      workload_coef=workload_coef,
      max_optimizer_steps=args.max_optimizer_steps,
      reduction=args.reduction,
  )
  output = args.output or default_artifact_path(
      ROOT,
      horizon=args.horizon,
      bandwidth=args.bandwidth,
      min_sep=args.min_sep,
      max_participations=args.max_participations,
      workload=args.workload,
      momentum=args.momentum,
      learning_rate=args.learning_rate,
  )
  publish_strategy_artifact(
      output, result, reduction=args.reduction, workload_type=args.workload,
      momentum=args.momentum, learning_rate=args.learning_rate,
      max_optimizer_steps=args.max_optimizer_steps,
  )
  print(f"saved strategy artifact: {output}")


if __name__ == "__main__":
  main()
