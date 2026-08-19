#!/usr/bin/env python3
"""Run Experiment 4A and 4B without duplicating continuous training."""
from __future__ import annotations

import argparse
import csv
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
    ParticipationSpec, calibrate_nonamplified_bandinv,
    epsilon_spent_for_bandinv_prefix,
)
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    load_cifar10_bandinv_dpadamw_config,
)
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule, cross_entropy_loss,
    evaluate_classifier_metrics, run_training,
)
from dp_muon.training.nonamplified_bandinv_dpadamw import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from dp_muon.training.run_logging import MetricsCSVWriter
from exp2.common import derive_contract, resolve_repo_path
from exp2.strategies import DECAYED_PREFIX, StrategySpec, load_or_fit_strategy
from exp4.diagnostics import compute_p_tree, diagnostics_row_dict, p_tree_statistics
from exp4.plotting import plot_comparison, plot_p_diagnostics, plot_p_summary
from exp4.segmented_strategy import (
    begin_segment, fit_segmented_plan, make_segmented_train_step,
)

CONDITIONS = ("continuous", "seg97", "seg16")


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--output-dir", default="exp4/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args(argv)


def _diagnostics_hook(beta2: float, eps: float):
  rows, previous = [], None
  def hook(state, step):
    nonlocal previous
    p_tree = compute_p_tree(state.optimizer_state, beta2=beta2, eps=eps)
    row, previous = p_tree_statistics(p_tree, previous, step=step)
    rows.append(diagnostics_row_dict(row))
  return hook, rows


def _write_diagnostics(rows: list[dict], path: Path) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)


def _result(seed: int, condition: str, history: list[dict]) -> dict[str, Any]:
  losses = [float(row["test_loss"]) for row in history]
  accuracies = [float(row["test_accuracy"]) for row in history]
  return {"seed": seed, "condition": condition,
          "final_test_loss": losses[-1], "final_test_accuracy": accuracies[-1],
          "best_test_loss": min(losses), "best_test_accuracy": max(accuracies)}


def _write_outputs(out: Path, results: list[dict], metadata: dict,
                   diagnostic_paths: list[tuple[int, Path]]) -> None:
  fields = ["seed", "condition", "final_test_loss", "final_test_accuracy",
            "best_test_loss", "best_test_accuracy"]
  with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(results)
  aggregate = {}
  for condition in CONDITIONS:
    subset = [row for row in results if row["condition"] == condition]
    aggregate[condition] = {}
    for field in fields[2:]:
      values = np.asarray([row[field] for row in subset], dtype=float)
      aggregate[condition][field] = {"mean": float(values.mean()),
                                     "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
  (out / "summary.json").write_text(json.dumps(
      {**metadata, "per_run": results, "aggregate": aggregate}, indent=2), encoding="utf-8")
  plot_comparison(results, out / "comparison_final_accuracy.png")
  plot_p_summary(diagnostic_paths, out / "p_median_summary.png")


def _continuous_strategy(config, contract, *, max_optimizer_steps=None):
  spec = StrategySpec(
      DECAYED_PREFIX, contract.horizon, min(config.bandwidth, contract.horizon),
      contract.min_sep, contract.max_participations, config.learning_rate,
      config.beta1, config.weight_decay, config.reduction,
      max_optimizer_steps or config.max_optimizer_steps)
  path = resolve_repo_path(config.strategy_dir) / "exp4_continuous_decayed_prefix.npz"
  return load_or_fit_strategy(path, spec, force_refit=config.force_refit)


def _privacy_accountant(condition, strategy, calibration):
  """Only Continuous has a mathematically supported prefix accountant."""
  if condition != "continuous":
    return None
  return lambda step: epsilon_spent_for_bandinv_prefix(
      prefix_steps=step, noising_coef=strategy.noising_coef,
      horizon=strategy.horizon, min_sep=strategy.min_sep,
      max_participations=strategy.max_participations,
      calibration=calibration,
      full_sensitivity_squared=float(strategy.sensitivity_squared))


def run_real(config_path: str | Path, out: Path, seeds: list[int]) -> None:
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  train_x, train_y = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  test_x, test_y = load_cifar10(resolve_repo_path(config.data_dir), train=False)
  contract = derive_contract(config, num_examples=len(train_x))
  continuous = _continuous_strategy(config, contract)
  plans = {name: fit_segmented_plan(
      horizon=contract.horizon, block_size=size, bandwidth=config.bandwidth,
      min_sep=contract.min_sep, max_participations=contract.max_participations,
      max_optimizer_steps=config.max_optimizer_steps, reduction=config.reduction,
      learning_rate=config.learning_rate, weight_decay=config.weight_decay,
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency)
      for name, size in (("seg97", 97), ("seg16", 16))}
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(continuous.sensitivity_squared))
  results, diagnostic_paths = [], []
  model = ViTTiny()
  for seed in seeds:
    parameter_key, noise_key = jax.random.split(jax.random.key(seed))
    params = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key).params
    schedule = build_fixed_cycle_logical_schedule(
        num_examples=len(train_x), batch_size=config.batch_size,
        horizon=contract.horizon, min_sep=contract.min_sep,
        max_participations=contract.max_participations, seed=seed)
    for condition in CONDITIONS:
      if condition == "continuous":
        step_fn, optimizer = make_nonamplified_bandinv_dpadamw_train_step(
            lambda p, b: cross_entropy_loss(p, b, model), continuous, calibration,
            ParticipationSpec(contract.horizon, contract.min_sep, contract.max_participations),
            learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2,
            eps=config.eps, weight_decay=config.weight_decay,
            microbatch_size=config.microbatch_size)
        strategy_for_init = continuous; before_step = None
        hook, diagnostics = _diagnostics_hook(config.beta2, config.eps)
      else:
        plan = plans[condition]
        step_fn, optimizer = make_segmented_train_step(
            lambda p, b: cross_entropy_loss(p, b, model), plan,
            learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2,
            eps=config.eps, weight_decay=config.weight_decay,
            microbatch_size=config.microbatch_size)
        strategy_for_init = plan.strategies[0]
        before_step = lambda state, step, p=plan: begin_segment(state, step, p)
        hook, diagnostics = None, []
      state = init_nonamplified_bandinv_dpadamw_state(
          params, strategy_for_init, noise_key, optimizer)
      metrics_path = out / f"metrics_{condition}_seed{seed}.csv"
      _, history = run_training(
          initial_state=state, train_step=step_fn,
          logical_batches=iter_logical_batches(train_x, train_y, schedule),
          horizon=contract.horizon, experiment_config={**asdict(config), "condition": condition},
          artifact_identifiers={"experiment": "exp4", "condition": condition},
          # The shared checkpoint schema requires BandInvMF's step to be
          # global. Segmented noise steps are deliberately block-local.
          checkpoint_path=(out / f"checkpoint_{condition}_seed{seed}.pkl"
                           if condition == "continuous" else None),
          eval_every=config.eval_every,
          evaluate=lambda s: evaluate_classifier_metrics(
              s.params, model, test_x, test_y, batch_size=config.batch_size),
          num_train_examples=len(train_x), logical_batch_size=config.batch_size,
          metrics_writer=MetricsCSVWriter(metrics_path), before_step=before_step,
          after_step=hook,
          privacy_accountant=_privacy_accountant(
              condition, continuous, calibration))
      results.append(_result(seed, condition, history))
      if condition == "continuous":
        path = out / f"diagnostics_continuous_seed{seed}.csv"
        _write_diagnostics(diagnostics, path); plot_p_diagnostics(path, out, seed)
        diagnostic_paths.append((seed, path))
  metadata = {"smoke": False, "contract": asdict(contract),
      "privacy": {"epsilon": config.epsilon, "delta": config.delta,
                  "continuous_sensitivity_squared": float(continuous.sensitivity_squared),
                  **{f"{name}_sensitivity_squared": plan.sensitivity_squared for name, plan in plans.items()}}}
  _write_outputs(out, results, metadata, diagnostic_paths)


def run_smoke(out: Path, seeds: list[int]) -> None:
  """Real DP/AdamW path over a small synthetic differentiable problem."""
  horizon, min_sep, max_participations = 18, 18, 1
  learning_rate, weight_decay, beta2, eps = .02, .01, .9, 1e-6
  continuous = fit_bandinv_strategy(
      horizon, 2, min_sep, max_participations=max_participations,
      workload_coef=decayed_prefix_sum_workload_coef(horizon, learning_rate, weight_decay),
      max_optimizer_steps=2)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=3., delta=1e-5, clip_norm=1., normalize_by=2.,
      adjacency="add_remove", sensitivity_squared=float(continuous.sensitivity_squared))
  plans = {name: fit_segmented_plan(
      horizon=horizon, block_size=size, bandwidth=2, min_sep=min_sep,
      max_participations=max_participations, max_optimizer_steps=2, reduction="mean",
      learning_rate=learning_rate, weight_decay=weight_decay, epsilon=3., delta=1e-5,
      clip_norm=1., normalize_by=2., adjacency="add_remove")
      for name, size in (("seg97", 97), ("seg16", 16))}
  def loss(params, batch):
    prediction = jnp.dot(params["w"], batch["x"][0])
    return .5 * (prediction - batch["y"][0]) ** 2
  batches = [{"x": np.asarray([[1., step / horizon], [1., -step / horizon]], np.float32),
              "y": np.asarray([.5, -.25], np.float32)} for step in range(horizon)]
  results, diagnostic_paths = [], []
  for seed in seeds:
    params = {"w": jax.random.normal(jax.random.key(seed), (2,)) * .01}
    noise_key = jax.random.key(seed + 10_000)
    for condition in CONDITIONS:
      if condition == "continuous":
        step_fn, optimizer = make_nonamplified_bandinv_dpadamw_train_step(
            loss, continuous, calibration,
            ParticipationSpec(horizon, min_sep, max_participations),
            learning_rate=learning_rate, beta2=beta2, eps=eps,
            weight_decay=weight_decay)
        strategy, before = continuous, None
        hook, diagnostics = _diagnostics_hook(beta2, eps)
      else:
        plan = plans[condition]
        step_fn, optimizer = make_segmented_train_step(
            loss, plan, learning_rate=learning_rate, beta1=.9, beta2=beta2,
            eps=eps, weight_decay=weight_decay)
        strategy = plan.strategies[0]
        before = lambda state, step, p=plan: begin_segment(state, step, p)
        hook, diagnostics = None, []
      initial = init_nonamplified_bandinv_dpadamw_state(params, strategy, noise_key, optimizer)
      def evaluate(state):
        value = float(jnp.sum(state.params["w"] ** 2))
        return {"test_loss": value, "test_accuracy": 1. / (1. + value)}
      _, history = run_training(
          initial_state=initial, train_step=step_fn, logical_batches=list(batches),
          horizon=horizon, experiment_config={"smoke": True, "condition": condition},
          artifact_identifiers={"experiment": "exp4-smoke"}, eval_every=1,
          evaluate=evaluate, before_step=before, after_step=hook,
          num_train_examples=12, logical_batch_size=2,
          metrics_writer=MetricsCSVWriter(out / f"metrics_{condition}_seed{seed}.csv"),
          privacy_accountant=_privacy_accountant(condition, continuous, calibration))
      results.append(_result(seed, condition, history))
      if condition == "continuous":
        path = out / f"diagnostics_continuous_seed{seed}.csv"
        _write_diagnostics(diagnostics, path); plot_p_diagnostics(path, out, seed)
        diagnostic_paths.append((seed, path))
  _write_outputs(out, results, {"smoke": True, "privacy": {
      "epsilon": 3., "delta": 1e-5,
      "continuous_sensitivity_squared": float(continuous.sensitivity_squared),
      **{f"{name}_sensitivity_squared": plan.sensitivity_squared
         for name, plan in plans.items()}}}, diagnostic_paths)


def main(argv=None):
  args = parse_args(argv)
  out = resolve_repo_path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
  if args.smoke:
    run_smoke(out, args.seeds)
  else:
    run_real(args.config, out, args.seeds)


if __name__ == "__main__":
  main()
