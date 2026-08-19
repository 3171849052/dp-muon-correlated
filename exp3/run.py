#!/usr/bin/env python3
"""Run real CIFAR-10 DP-AdamW training with online shadow diagnostics."""
from __future__ import annotations

import argparse, csv, json, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import jax
import numpy as np

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv, epsilon_spent_for_bandinv_prefix
from dp_muon.training.cifar10_driver import (build_fixed_cycle_logical_schedule, cross_entropy_loss,
    evaluate_classifier_metrics, run_training)
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import load_cifar10_bandinv_dpadamw_config
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from exp2.common import derive_contract, resolve_repo_path
from exp2.strategies import ADAM_M_AWARE, DECAYED_PREFIX, StrategySpec, load_or_fit_strategy
from exp3.online_shadow import init_online_shadow_state, make_online_shadow_train_step


def build_schedule_for_seed(contract, seed):
  """Build one fixed-cycle schedule; both strategies for this seed reuse it."""
  return build_fixed_cycle_logical_schedule(
      num_examples=contract.num_examples, batch_size=contract.batch_size,
      horizon=contract.horizon, min_sep=contract.min_sep,
      max_participations=contract.max_participations, seed=seed)


def _mean_std(values):
  values = [float(value) for value in values]
  if len(values) == 1:
    return values[0], 0.0
  array = np.asarray(values, dtype=float)
  return float(np.mean(array)), float(np.std(array, ddof=1))


def paired_seed_aggregation(results):
  """Pair naive and m-aware runs by seed before computing deltas/statistics."""
  by_seed = {}
  for result in results:
    seed = int(result["seed"])
    strategy = result["strategy"]
    if strategy not in (DECAYED_PREFIX, ADAM_M_AWARE):
      raise ValueError(f"unknown strategy {strategy!r}")
    if seed in by_seed and strategy in by_seed[seed]:
      raise ValueError(f"duplicate {strategy!r} result for seed {seed}")
    by_seed.setdefault(seed, {})[strategy] = result
  paired = []
  for seed in sorted(by_seed):
    runs = by_seed[seed]
    if set(runs) != {DECAYED_PREFIX, ADAM_M_AWARE}:
      raise ValueError(f"seed {seed} must contain both strategies")
    naive, aware = runs[DECAYED_PREFIX], runs[ADAM_M_AWARE]
    delta_linear = float(aware["R_linear"] - naive["R_linear"])
    delta_adamw = float(aware["R_adamw"] - naive["R_adamw"])
    paired.append({
        "seed": seed,
        "delta_R_linear": delta_linear,
        "delta_R_adamw": delta_adamw,
        "gamma_R": delta_adamw - delta_linear,
        "delta_accuracy": float(aware["final_accuracy"] - naive["final_accuracy"]),
        "delta_test_loss": float(naive["final_test_loss"] - aware["final_test_loss"]),
    })
  fields = ("delta_R_linear", "delta_R_adamw", "gamma_R", "delta_accuracy", "delta_test_loss")
  aggregate = {"num_seeds": len(paired)}
  for field in fields:
    mean, std = _mean_std([row[field] for row in paired])
    aggregate[f"{field}_mean"] = mean
    aggregate[f"{field}_std"] = std
  return paired, aggregate


def write_comparison(path, paired):
  fields = ["seed", "delta_R_linear", "delta_R_adamw", "gamma_R", "delta_accuracy", "delta_test_loss"]
  with Path(path).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(paired)


def _strategy(config, contract, name):
  return load_or_fit_strategy(resolve_repo_path(config.strategy_dir) / (name + ".npz"), StrategySpec(
      name=name, horizon=contract.horizon, bandwidth=config.bandwidth, min_sep=contract.min_sep,
      max_participations=contract.max_participations, learning_rate=config.learning_rate,
      beta1=config.beta1, weight_decay=config.weight_decay, reduction=config.reduction,
      max_optimizer_steps=config.max_optimizer_steps), force_refit=config.force_refit)


def run_one(config, contract, name, strategy, seed, train_images, train_labels, test_images, test_labels, schedule, output):
  output.mkdir(parents=True, exist_ok=True)
  model = ViTTiny()
  parameter_key, noise_key = jax.random.split(jax.random.key(seed))
  snapshot = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key)
  calibration = calibrate_nonamplified_bandinv(epsilon=config.epsilon, delta=config.delta,
      clip_norm=config.clip_norm, normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared))
  participation = ParticipationSpec(contract.horizon, contract.min_sep, contract.max_participations)
  step_fn, optimizer = make_online_shadow_train_step(
      lambda p, b: cross_entropy_loss(p, b, model), strategy, calibration, participation,
      learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2, eps=config.eps,
      weight_decay=config.weight_decay, microbatch_size=config.microbatch_size)
  state = init_online_shadow_state(snapshot.params, strategy, noise_key, optimizer)
  def evaluate(s):
    tr = evaluate_classifier_metrics(s.params, model, train_images, train_labels, batch_size=config.batch_size)
    te = evaluate_classifier_metrics(s.params, model, test_images, test_labels, batch_size=config.batch_size)
    return {"train_loss": tr["test_loss"], "test_loss": te["test_loss"], "test_accuracy": te["test_accuracy"],
            "amplitude_distortion": s.amplitude, "direction_distortion": s.direction,
            "R_linear_prefix": s.j_linear/(s.d_prefix_linear+1e-30),
            "R_adamw_prefix": s.j_adamw/(s.d_prefix_adamw+1e-30),
            "R_linear_aggregate": s.sum_j_linear/(s.sum_d_linear+1e-30),
            "R_adamw_aggregate": s.sum_j_adamw/(s.sum_d_adamw+1e-30)}
  ckpt = output / f"checkpoint_{name.replace('-', '_')}_seed{seed}.pkl"
  final, history = run_training(initial_state=state, train_step=step_fn,
      logical_batches=iter_logical_batches(train_images, train_labels, schedule), horizon=contract.horizon,
      experiment_config=asdict(config), artifact_identifiers={"strategy": name, "pretrained_sha256": snapshot.sha256},
      checkpoint_path=ckpt, eval_every=config.eval_every, evaluate=evaluate,
      num_train_examples=contract.num_examples, logical_batch_size=contract.batch_size,
      privacy_accountant=lambda step: epsilon_spent_for_bandinv_prefix(prefix_steps=step,
          noising_coef=strategy.noising_coef, horizon=strategy.horizon, min_sep=strategy.min_sep,
          max_participations=strategy.max_participations, calibration=calibration,
          full_sensitivity_squared=float(strategy.sensitivity_squared)))
  rows = [{"strategy": name, "seed": seed, **row} for row in history]
  with (output / f"diagnostics_{name.replace('-', '_')}_seed{seed}.csv").open("w", newline="") as f:
    fields = sorted({k for r in rows for k in r}); writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
  last = rows[-1]
  return {"strategy": name, "seed": seed, "final_train_loss": last["train_loss"], "final_test_loss": last["test_loss"],
      "final_accuracy": last["test_accuracy"], "R_linear": last["R_linear_aggregate"], "R_adamw": last["R_adamw_aggregate"],
      "amplitude_distortion": last["amplitude_distortion"], "direction_distortion": last["direction_distortion"],
      "calibration": asdict(calibration), "horizon": contract.horizon, "min_sep": contract.min_sep,
      "max_participations": contract.max_participations}


def main():
  p = argparse.ArgumentParser(); p.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml"); p.add_argument("--output-dir", default="exp3/results"); p.add_argument("--seeds", nargs="+", type=int, default=None)
  a = p.parse_args(); config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(a.config)); train_x, train_y = load_cifar10(resolve_repo_path(config.data_dir), train=True); test_x, test_y = load_cifar10(resolve_repo_path(config.data_dir), train=False)
  contract = derive_contract(config, num_examples=len(train_x))
  output = resolve_repo_path(a.output_dir); output.mkdir(parents=True, exist_ok=True); results = []
  for seed in a.seeds or [config.seed]:
    schedule = build_schedule_for_seed(contract, seed)
    for name in (DECAYED_PREFIX, ADAM_M_AWARE):
      results.append(run_one(config, contract, name, _strategy(config, contract, name), seed, train_x, train_y, test_x, test_y, schedule, output))
  paired, aggregate = paired_seed_aggregation(results)
  write_comparison(output / "comparison.csv", paired)
  summary = {"contract": asdict(contract), "per_run": results,
             "paired_by_seed": paired, "aggregate": aggregate}
  (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
