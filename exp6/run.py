#!/usr/bin/env python3
"""Run Experiment 6: local cancellation and clean-AdamW-p diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import fit_bandinv_strategy
from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
    epsilon_spent_for_bandinv_prefix,
)
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule,
    cross_entropy_loss,
    evaluate_classifier_metrics,
    run_training,
)
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    load_cifar10_bandinv_dpadamw_config,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from exp2.common import derive_contract, resolve_repo_path
from exp2.strategies import DECAYED_PREFIX
from exp3.online_shadow import init_online_shadow_state, make_online_shadow_train_step
from exp3.run import _strategy as exp3_strategy

from exp6.diagnostics import (
    aggregate_window_rows,
    correlation_from_rows,
    per_seed_stage_summary,
    stage_summary,
    write_window_rows,
    write_window_summary,
)
from exp6.online_shadow import WindowDiagnosticsCollector
from exp6.plotting import plot_over_steps, plot_scatter


WINDOW_SIZE = 16


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--output-dir", default="exp6/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args(argv)


def build_schedule_for_seed(contract, seed):
  """Reuse Experiment 3's fixed-cycle schedule contract."""
  return build_fixed_cycle_logical_schedule(
      num_examples=contract.num_examples,
      batch_size=contract.batch_size,
      horizon=contract.horizon,
      min_sep=contract.min_sep,
      max_participations=contract.max_participations,
      seed=seed,
  )


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, (np.integer, np.floating)):
    value = value.item()
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def _write_run_rows(output: Path, rows: list[dict[str, float | int]], seed: int) -> Path:
  path = output / f"window_diagnostics_seed{seed}.csv"
  write_window_rows(path, rows)
  return path


def _write_outputs(
    output: Path,
    rows: list[dict[str, float | int]],
    *,
    horizon: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
  summary_rows = aggregate_window_rows(rows)
  write_window_summary(output / "window_summary.csv", summary_rows)
  rho = plot_scatter(rows, output / "p_change_vs_cancellation.png")
  plot_over_steps(summary_rows, output / "diagnostics_over_steps.png")
  summary = {
      **metadata,
      "window_size": WINDOW_SIZE,
      "per_seed_stage_summary": per_seed_stage_summary(
          rows, early_end=97, total_steps=horizon
      ),
      "stage_summary": stage_summary(rows, early_end=97, total_steps=horizon),
      "spearman_mean_p_relative_change_vs_delta_p_cancellation": rho,
      "num_window_rows": len(rows),
  }
  (output / "summary.json").write_text(
      json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
  )
  return summary


def run_one(
    config,
    contract,
    strategy,
    seed: int,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    schedule: list[np.ndarray],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
  """Run one continuous seed using Experiment 3's exact private step."""
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(seed))
  snapshot = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(
      contract.horizon, contract.min_sep, contract.max_participations
  )
  train_step, optimizer = make_online_shadow_train_step(
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
  state = init_online_shadow_state(snapshot.params, strategy, noise_key, optimizer)
  collector = WindowDiagnosticsCollector(
      snapshot.params,
      seed=seed,
      beta1=config.beta1,
      beta2=config.beta2,
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      eps=config.eps,
      window_size=WINDOW_SIZE,
  )

  def evaluate(current_state):
    train_metrics = evaluate_classifier_metrics(
        current_state.params, model, train_images, train_labels,
        batch_size=config.batch_size,
    )
    test_metrics = evaluate_classifier_metrics(
        current_state.params, model, test_images, test_labels,
        batch_size=config.batch_size,
    )
    return {
        "train_loss": train_metrics["test_loss"],
        "test_loss": test_metrics["test_loss"],
        "test_accuracy": test_metrics["test_accuracy"],
    }

  checkpoint = output / f"checkpoint_decayed_prefix_seed{seed}.pkl"
  final, history = run_training(
      initial_state=state,
      train_step=train_step,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule),
      horizon=contract.horizon,
      experiment_config=asdict(config),
      artifact_identifiers={
          "experiment": "exp6",
          "strategy": DECAYED_PREFIX,
          "strategy_path": str(resolve_repo_path(config.strategy_dir) / "decayed-prefix.npz"),
          "pretrained_sha256": snapshot.sha256,
      },
      checkpoint_path=checkpoint,
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
      after_step=collector.after_step,
  )
  rows = collector.finalize()
  _write_run_rows(output, rows, seed)
  last = history[-1] if history else {}
  result = {
      "seed": seed,
      "strategy": DECAYED_PREFIX,
      "num_windows": len(rows),
      "final_test_loss": last.get("test_loss"),
      "final_test_accuracy": last.get("test_accuracy"),
      "final_step": int(final.step),
      "calibration": asdict(calibration),
  }
  return result, rows


def run_real(config_path: str | Path, output: Path, seeds: list[int]) -> None:
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  train_images, train_labels = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  test_images, test_labels = load_cifar10(resolve_repo_path(config.data_dir), train=False)
  contract = derive_contract(config, num_examples=len(train_images))
  strategy = exp3_strategy(config, contract, DECAYED_PREFIX)
  rows: list[dict[str, float | int]] = []
  per_run = []
  output.mkdir(parents=True, exist_ok=True)
  for seed in seeds:
    schedule = build_schedule_for_seed(contract, seed)
    result, seed_rows = run_one(
        config, contract, strategy, seed, train_images, train_labels,
        test_images, test_labels, schedule, output,
    )
    per_run.append(result)
    rows.extend(seed_rows)
  _write_outputs(
      output,
      rows,
      horizon=contract.horizon,
      metadata={
          "smoke": False,
          "contract": asdict(contract),
          "config": asdict(config),
          "per_run": per_run,
      },
  )


def run_smoke(output: Path, seeds: list[int]) -> None:
  """Start the same real BandInvMF/AdamW/diagnostic path on a tiny workload."""
  horizon = 20
  learning_rate, weight_decay = 0.02, 0.01
  beta1, beta2, eps = 0.9, 0.9, 1e-6
  strategy = fit_bandinv_strategy(
      horizon,
      bandwidth=2,
      min_sep=1,
      max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(
          horizon, learning_rate, weight_decay
      ),
      max_optimizer_steps=3,
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0,
      delta=1e-5,
      clip_norm=1.0,
      normalize_by=2.0,
      adjacency="add_remove",
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(horizon, 1, 1)

  def loss(params, batch):
    prediction = jnp.dot(params["w"], batch["x"][0])
    return 0.5 * (prediction - batch["y"][0]) ** 2

  batches = [
      {
          "x": np.asarray([[1.0, step / horizon], [1.0, -step / horizon]], np.float32),
          "y": np.asarray([0.5, -0.25], np.float32),
      }
      for step in range(horizon)
  ]
  output.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, float | int]] = []
  per_run = []
  for seed in seeds:
    train_step, optimizer = make_online_shadow_train_step(
        loss,
        strategy,
        calibration,
        participation,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )
    params = {"w": jax.random.normal(jax.random.key(seed), (2,)) * 0.01}
    state = init_online_shadow_state(
        params, strategy, jax.random.key(seed + 10_000), optimizer
    )
    collector = WindowDiagnosticsCollector(
        params,
        seed=seed,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eps=eps,
        window_size=WINDOW_SIZE,
    )
    final, _ = run_training(
        initial_state=state,
        train_step=train_step,
        logical_batches=list(batches),
        horizon=horizon,
        experiment_config={"experiment": "exp6-smoke"},
        artifact_identifiers={"experiment": "exp6-smoke"},
        eval_every=1,
        evaluate=None,
        after_step=collector.after_step,
    )
    seed_rows = collector.finalize()
    _write_run_rows(output, seed_rows, seed)
    rows.extend(seed_rows)
    per_run.append({"seed": seed, "final_step": int(final.step), "num_windows": len(seed_rows)})
  _write_outputs(
      output,
      rows,
      horizon=horizon,
      metadata={"smoke": True, "per_run": per_run},
  )


def main(argv=None):
  args = parse_args(argv)
  output = resolve_repo_path(args.output_dir)
  if args.smoke:
    run_smoke(output, args.seeds)
  else:
    run_real(args.config, output, args.seeds)


if __name__ == "__main__":
  main()
