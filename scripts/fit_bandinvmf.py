#!/usr/bin/env python3
"""Fit and save a public prefix-sum BandInvMF strategy artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dp_muon.bandinvmf import fit_bandinv_strategy


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--horizon", type=int, required=True)
  parser.add_argument("--bandwidth", type=int, required=True)
  parser.add_argument("--min-sep", type=int, required=True)
  parser.add_argument("--max-participations", type=int)
  parser.add_argument("--max-optimizer-steps", type=int, default=100)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  result = fit_bandinv_strategy(
      args.horizon,
      args.bandwidth,
      args.min_sep,
      max_participations=args.max_participations,
      max_optimizer_steps=args.max_optimizer_steps,
  )
  output = args.output or ROOT / "artifacts" / "strategies" / (
      f"prefix_h{args.horizon}_b{args.bandwidth}_sep{args.min_sep}.npz"
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
  )
  print(f"saved strategy artifact: {output}")


if __name__ == "__main__":
  main()
