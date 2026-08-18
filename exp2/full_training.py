#!/usr/bin/env python3
"""Run the full DP-AdamW comparison for Exp2's two fitted workloads."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dp_muon.bandinvmf import load_bandinv_strategy
from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
    epsilon_spent_for_bandinv_prefix,
)
from dp_muon.training.bandinvmf_strategy_manager import load_strategy_snapshot
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    load_cifar10_bandinv_dpadamw_config,
)
from dp_muon.training.cifar10_driver import (
    Cifar10BandInvDPAdamWTrainConfig,
    build_logical_schedule,
    cross_entropy_loss,
    evaluate_classifier_metrics,
    run_training,
)
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot

from exp2.common import contract_dict, derive_contract, resolve_repo_path
from exp2.strategies import (
    ADAM_M_AWARE,
    DECAYED_PREFIX,
    StrategySpec,
    load_or_fit_strategy,
)


def _strategy_spec(config: Any, contract: Any, name: str) -> StrategySpec:
  return StrategySpec(
      name=name,
      horizon=contract.horizon,
      bandwidth=config.bandwidth,
      min_sep=contract.min_sep,
      max_participations=contract.max_participations,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      weight_decay=config.weight_decay,
      reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps,
  )


def calibration_metadata(config: Any, strategy: Any) -> dict[str, Any]:
  """Calibrate one strategy at the common public (epsilon, delta) target."""
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  result = asdict(calibration)
  result.update({
      "sensitivity_squared": float(strategy.sensitivity_squared),
      "strategy_objective": float(strategy.objective),
      "calibrated_noise_stddev": float(calibration.iid_noise_std),
      "calibrated_noise_multiplier": float(calibration.noise_multiplier),
  })
  return result


def _safe_name(name: str) -> str:
  return name.replace("-", "_")


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
  fields = [
      "strategy", "workload_type", "seed", "sensitivity_squared", "strategy_objective",
      "epsilon", "delta", "adjacency", "calibrated_noise_stddev",
      "calibrated_noise_multiplier", "epoch", "step", "effective_epoch",
      "epsilon_spent", "train_loss", "test_loss", "test_accuracy",
      "elapsed_seconds", "eval_seconds",
  ]
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _train_one_strategy(
    *, config: Any, contract: Any, strategy_name: str, strategy_path: Path,
    strategy: Any, seed: int, train_images: Any, train_labels: Any,
    test_images: Any, test_labels: Any, schedule: list[Any], output_dir: Path,
) -> dict[str, Any]:
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(seed))
  pretrained_snapshot = load_pretrained_snapshot(
      resolve_repo_path(config.pretrained), key=parameter_key
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(
      strategy.horizon, strategy.min_sep, strategy.max_participations
  )
  train_step, optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      lambda parameters, batch: cross_entropy_loss(parameters, batch, model),
      strategy,
      calibration,
      participation,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      beta2=config.beta2,
      eps=config.eps,
      weight_decay=config.weight_decay,
      microbatch_size=config.microbatch_size,
  )
  initial_state = init_nonamplified_bandinv_dpadamw_state(
      pretrained_snapshot.params, strategy, noise_key, optimizer
  )
  train_config = Cifar10BandInvDPAdamWTrainConfig(
      strategy=str(strategy_path),
      pretrained=str(resolve_repo_path(config.pretrained)),
      data_dir=str(resolve_repo_path(config.data_dir)),
      batch_size=config.batch_size,
      microbatch_size=config.microbatch_size,
      clip_norm=config.clip_norm,
      epsilon=config.epsilon,
      delta=config.delta,
      learning_rate=config.learning_rate,
      beta1=config.beta1,
      beta2=config.beta2,
      eps=config.eps,
      weight_decay=config.weight_decay,
      seed=seed,
      checkpoint_dir=str(output_dir / f"checkpoints_{_safe_name(strategy_name)}_seed{seed}"),
      eval_every=config.eval_every,
      adjacency=config.adjacency,
  )

  def evaluate(state: Any) -> dict[str, float]:
    train_metrics = evaluate_classifier_metrics(
        state.params, model, train_images, train_labels, batch_size=config.batch_size
    )
    test_metrics = evaluate_classifier_metrics(
        state.params, model, test_images, test_labels, batch_size=config.batch_size
    )
    return {
        "train_loss": train_metrics["test_loss"],
        "test_loss": test_metrics["test_loss"],
        "test_accuracy": test_metrics["test_accuracy"],
    }

  checkpoint_path = output_dir / f"checkpoints_{_safe_name(strategy_name)}_seed{seed}" / "latest.pkl"
  _, history = run_training(
      initial_state=initial_state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=contract.horizon,
      experiment_config=asdict(train_config),
      artifact_identifiers={
          "algorithm": "dp-adamw-correlated-naive",
          "strategy_path": str(strategy_path.resolve()),
          "pretrained_path": str(pretrained_snapshot.path),
          "pretrained_sha256": pretrained_snapshot.sha256,
      },
      checkpoint_path=checkpoint_path,
      eval_every=config.eval_every,
      evaluate=evaluate,
      num_train_examples=contract.num_examples,
      logical_batch_size=contract.batch_size,
      privacy_accountant=lambda step: epsilon_spent_for_bandinv_prefix(
          prefix_steps=step,
          noising_coef=strategy.noising_coef,
          horizon=strategy.horizon,
          min_sep=strategy.min_sep,
          max_participations=strategy.max_participations,
          calibration=calibration,
          full_sensitivity_squared=float(strategy.sensitivity_squared),
      ),
  )
  calibration_info = calibration_metadata(config, strategy)
  metric_rows = []
  for record in history:
    metric_rows.append({
        "strategy": strategy_name,
        "workload_type": strategy_name,
        "seed": seed,
        "sensitivity_squared": calibration_info["sensitivity_squared"],
        "strategy_objective": calibration_info["strategy_objective"],
        "epsilon": config.epsilon,
        "delta": config.delta,
        "adjacency": config.adjacency,
        "calibrated_noise_stddev": calibration_info["calibrated_noise_stddev"],
        "calibrated_noise_multiplier": calibration_info["calibrated_noise_multiplier"],
        **{field: record.get(field, float("nan")) for field in (
            "epoch", "step", "effective_epoch", "epsilon_spent", "train_loss",
            "test_loss", "test_accuracy", "elapsed_seconds", "eval_seconds",
        )},
    })
  metrics_path = output_dir / f"metrics_{_safe_name(strategy_name)}_seed{seed}.csv"
  _write_metrics(metrics_path, metric_rows)
  final = metric_rows[-1]
  best_accuracy = max(metric_rows, key=lambda row: row["test_accuracy"])
  best_loss = min(metric_rows, key=lambda row: row["test_loss"])
  return {
      "strategy": strategy_name,
      "seed": seed,
      "metrics": str(metrics_path.resolve()),
      "checkpoint": str(checkpoint_path.resolve()),
      "final_train_loss": final["train_loss"],
      "final_test_loss": final["test_loss"],
      "final_test_accuracy": final["test_accuracy"],
      "best_test_accuracy": best_accuracy["test_accuracy"],
      "best_accuracy_epoch": best_accuracy["epoch"],
      "best_test_loss": best_loss["test_loss"],
      "best_loss_epoch": best_loss["epoch"],
      "strategy_metadata": {
          "workload_type": strategy_name,
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
          "privacy_calibration": calibration_info,
      },
  }


def run_full_training(
    *, config_path: str | Path, strategy_dir: str | Path, output_dir: str | Path,
    seeds: list[int], force_refit: bool = False,
) -> dict[str, Any]:
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  train_images, train_labels = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  test_images, test_labels = load_cifar10(resolve_repo_path(config.data_dir), train=False)
  contract = derive_contract(config, num_examples=len(train_images))
  output_dir = resolve_repo_path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  strategy_dir = resolve_repo_path(strategy_dir)
  strategies = {}
  for name in (DECAYED_PREFIX, ADAM_M_AWARE):
    spec = _strategy_spec(config, contract, name)
    path = strategy_dir / f"{name}.npz"
    strategy = load_or_fit_strategy(path, spec, force_refit=force_refit)
    strategies[name] = (path, strategy)
  for seed in seeds:
    schedule = build_logical_schedule(
        num_examples=contract.num_examples,
        batch_size=contract.batch_size,
        strategy=strategies[DECAYED_PREFIX][1],
        seed=seed,
    )
    other_schedule = build_logical_schedule(
        num_examples=contract.num_examples,
        batch_size=contract.batch_size,
        strategy=strategies[ADAM_M_AWARE][1],
        seed=seed,
    )
    if len(schedule) != len(other_schedule) or any(
        not np.array_equal(left, right)
        for left, right in zip(schedule, other_schedule, strict=True)
    ):
      raise ValueError("the two Exp2 strategies do not share the same fixed-cycle schedule")
  results = []
  for seed in seeds:
    schedule = build_logical_schedule(
        num_examples=contract.num_examples, batch_size=contract.batch_size,
        strategy=strategies[DECAYED_PREFIX][1], seed=seed,
    )
    for name in (DECAYED_PREFIX, ADAM_M_AWARE):
      path, strategy = strategies[name]
      results.append(_train_one_strategy(
          config=config, contract=contract, strategy_name=name, strategy_path=path,
          strategy=strategy, seed=seed, train_images=train_images,
          train_labels=train_labels, test_images=test_images, test_labels=test_labels,
          schedule=schedule, output_dir=output_dir,
      ))
  calibration_budgets = {
      name: {
          "epsilon": config.epsilon,
          "delta": config.delta,
          "adjacency": config.adjacency,
      }
      for name in (DECAYED_PREFIX, ADAM_M_AWARE)
  }
  comparison_fields = [
      "strategy", "seed", "final_train_loss", "final_test_loss", "final_test_accuracy",
      "best_test_loss", "best_test_accuracy", "best_accuracy_epoch", "best_loss_epoch",
      "sensitivity_squared", "calibrated_noise_stddev", "calibrated_noise_multiplier",
      "strategy_objective", "workload_type",
  ]
  comparison_rows = []
  for record in results:
    metadata = record["strategy_metadata"]
    calibration = metadata["privacy_calibration"]
    comparison_rows.append({
        "strategy": record["strategy"], "seed": record["seed"],
        "final_train_loss": record["final_train_loss"],
        "final_test_loss": record["final_test_loss"],
        "final_test_accuracy": record["final_test_accuracy"],
        "best_test_loss": record["best_test_loss"],
        "best_test_accuracy": record["best_test_accuracy"],
        "best_accuracy_epoch": record["best_accuracy_epoch"],
        "best_loss_epoch": record["best_loss_epoch"],
        "sensitivity_squared": metadata["sensitivity_squared"],
        "calibrated_noise_stddev": calibration["calibrated_noise_stddev"],
        "calibrated_noise_multiplier": calibration["calibrated_noise_multiplier"],
        "strategy_objective": metadata["objective"],
        "workload_type": metadata["workload_type"],
    })
  with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=comparison_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(comparison_rows)
  summary = {
      "question_A_replay": "Delta R = R_m-aware - R_naive; negative means improved temporal cancellation.",
      "question_B_full_training": "Compare final and best utility at the same (epsilon, delta); no accuracy direction is assumed.",
      "config": str(resolve_repo_path(config_path)),
      "contract": contract_dict(contract),
      "adamw": {
          "learning_rate": config.learning_rate, "beta1": config.beta1,
          "beta2": config.beta2, "eps": config.eps, "weight_decay": config.weight_decay,
      },
      "privacy_budget_equal": calibration_budgets[DECAYED_PREFIX] == calibration_budgets[ADAM_M_AWARE],
      "privacy_budget": calibration_budgets,
      "same_schedule_and_initialization": True,
      "seeds": seeds,
      "results": results,
      "comparison": comparison_rows,
  }
  (output_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=ROOT / "config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--strategy-dir", type=Path, default=ROOT / "exp2/results/replay_full_horizon/strategies")
  parser.add_argument("--output-dir", type=Path, default=ROOT / "exp2/results/full_training")
  parser.add_argument("--seeds", type=int, nargs="+", default=[0])
  parser.add_argument("--force-refit", action="store_true")
  args = parser.parse_args()
  summary = run_full_training(
      config_path=args.config, strategy_dir=args.strategy_dir,
      output_dir=args.output_dir, seeds=args.seeds, force_refit=args.force_refit,
  )
  for row in summary["comparison"]:
    print(row, flush=True)


if __name__ == "__main__":
  main()
