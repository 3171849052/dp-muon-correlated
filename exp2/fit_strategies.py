#!/usr/bin/env python3
"""Fit and publish both full-horizon Experiment 2 BandInvMF artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exp2.common import contract_dict, load_config_and_contract, resolve_repo_path
from exp2.run import load_trajectory
from exp2.strategies import (
    ADAM_M_AWARE,
    DECAYED_PREFIX,
    StrategySpec,
    fit_strategy,
    save_strategy,
)


def fit_full_horizon_strategies(
    *, config_path: str | Path, trajectory_path: str | Path, strategy_dir: str | Path,
    force_refit: bool = True,
) -> dict[str, object]:
  config, contract = load_config_and_contract(config_path)
  trajectory = load_trajectory(resolve_repo_path(trajectory_path))
  expected = contract_dict(contract)
  for field, value in expected.items():
    if trajectory[field] != value:
      raise ValueError(f"trajectory {field}={trajectory[field]} does not match derived {value}")
  strategy_dir = resolve_repo_path(strategy_dir)
  strategy_dir.mkdir(parents=True, exist_ok=True)
  strategy_config = {
      "horizon": contract.horizon,
      "bandwidth": config.bandwidth,
      "min_sep": contract.min_sep,
      "max_participations": contract.max_participations,
      "learning_rate": config.learning_rate,
      "beta1": config.beta1,
      "weight_decay": config.weight_decay,
      "reduction": config.reduction,
      "max_optimizer_steps": config.max_optimizer_steps,
  }
  records: dict[str, object] = {}
  for name in (DECAYED_PREFIX, ADAM_M_AWARE):
    spec = StrategySpec(name=name, **strategy_config)
    path = strategy_dir / f"{name}.npz"
    if force_refit or not path.exists():
      strategy = fit_strategy(spec)
      save_strategy(path, strategy, spec)
      action = "fit"
    else:
      from dp_muon.bandinvmf import load_bandinv_strategy
      strategy = load_bandinv_strategy(path)
      action = "reuse"
    if name == DECAYED_PREFIX and strategy.workload_matrix is not None:
      raise ValueError("decayed-prefix must use the decayed-prefix workload coefficient")
    if name == ADAM_M_AWARE and strategy.workload_matrix is None:
      raise ValueError("adam-m-aware must use the general causal workload matrix")
    records[name] = {
        "artifact": str(path.resolve()),
        "action": action,
        "workload_type": name,
        "workload_representation": (
            "general-causal-matrix" if strategy.workload_matrix is not None
            else "decayed-prefix-coef"
        ),
        "horizon": strategy.horizon,
        "min_sep": strategy.min_sep,
        "max_participations": strategy.max_participations,
        "bandwidth": strategy.bandwidth,
        "sensitivity_squared": float(strategy.sensitivity_squared),
        "objective": float(strategy.objective),
    }
  output = strategy_dir.parent / "strategy_summary.json"
  summary = {
      "config": str(resolve_repo_path(config_path)),
      "contract": expected,
      "adamw": {
          "learning_rate": config.learning_rate,
          "beta1": config.beta1,
          "beta2": config.beta2,
          "eps": config.eps,
          "weight_decay": config.weight_decay,
      },
      "strategies": records,
  }
  output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=ROOT / "config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--trajectory", type=Path, default=ROOT / "exp2/results/replay_full_horizon/trajectory.npz")
  parser.add_argument("--strategy-dir", type=Path, default=ROOT / "exp2/results/replay_full_horizon/strategies")
  parser.add_argument("--reuse", action="store_true", help="reuse compatible artifacts instead of refitting")
  args = parser.parse_args()
  summary = fit_full_horizon_strategies(
      config_path=args.config, trajectory_path=args.trajectory,
      strategy_dir=args.strategy_dir, force_refit=not args.reuse,
  )
  for name, record in summary["strategies"].items():
    print(name, record["action"], record["artifact"])


if __name__ == "__main__":
  main()
