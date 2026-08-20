#!/usr/bin/env python3
"""Run Experiment 5A -> 5B -> 5C."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    load_cifar10_bandinv_dpadamw_config,
)
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer
from exp2.common import resolve_repo_path
from exp5.hybrid_optimizer import FrozenPAdamW, freeze_optax_adamw, p_star_from_optax
from exp5.hybrid_strategy import (
    HybridPlan, fit_hybrid_plan, hybrid_noising_matrix,
    p_aware_hybrid_objective, share_conservative_calibration,
)
from exp5.replay import paired_replay

DEFAULT_TAUS = [32, 48, 64, 80, 97]
CONDITIONS = ("dynamic_cont", "dynamic_seg97", "frozen_cont", "frozen_seg97")


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--output-dir", default="exp5/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--tau-candidates", nargs="+", type=int, default=DEFAULT_TAUS)
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args(argv)


def select_tau_star(rows: list[dict[str, Any]]) -> int:
  """Maximize 3-seed final accuracy; break ties by lower final loss."""
  candidates = sorted({int(row["tau"]) for row in rows})
  if not candidates:
    raise ValueError("switch comparison is empty")
  scores = {}
  for tau in candidates:
    subset = [row for row in rows if int(row["tau"]) == tau]
    scores[tau] = (float(np.mean([r["final_test_accuracy"] for r in subset])),
                   -float(np.mean([r["final_test_loss"] for r in subset])))
  return max(candidates, key=lambda tau: scores[tau])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    raise ValueError(f"cannot write empty result {path}")
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _aggregate(rows, key, fields):
  result = {}
  for value in sorted({row[key] for row in rows}, key=str):
    subset = [row for row in rows if row[key] == value]
    result[str(value)] = {}
    for field in fields:
      values = np.asarray([row[field] for row in subset], dtype=np.float64)
      result[str(value)][field] = {
          "mean": float(values.mean()),
          "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
      }
  return result


def _p_stats(p, previous):
  value = np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(p)])
  old = np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(previous)])
  return {
      "switch_relative_change": float(np.linalg.norm(value - old) / max(np.linalg.norm(old), 1e-30)),
      "switch_p_mean": float(np.mean(value)),
      "switch_p_median": float(np.median(value)),
      "switch_p_p10": float(np.percentile(value, 10)),
      "switch_p_p90": float(np.percentile(value, 90)),
      "switch_p_rms": float(np.sqrt(np.mean(value * value))),
  }


def _clip(gradient, norm=1.0):
  gradient = np.asarray(gradient, dtype=np.float64)
  return gradient * min(1.0, norm / max(float(np.linalg.norm(gradient)), 1e-30))


def _clean_gradient(theta, step, target):
  forcing = .08 * np.asarray([np.sin(.7 * step), np.cos(.4 * step), np.sin(.3 * step + .2)])
  return _clip(np.asarray(theta) - target + forcing)


def _evaluate(theta, target):
  loss = float(.5 * np.sum((np.asarray(theta) - target) ** 2))
  return loss, float(np.exp(-loss))


def _plans(horizon, tau, *, segmented, max_steps=8):
  return fit_hybrid_plan(
      horizon=horizon, tau=tau, block_size=4 if segmented else None,
      bandwidth=2, min_sep=4, max_participations=3,
      learning_rate=.04, beta1=.8, weight_decay=.02,
      epsilon=3., delta=1e-5, clip_norm=1., normalize_by=2.,
      adjacency="add_remove", max_optimizer_steps=max_steps)


def _noise(plan: HybridPlan, latent: np.ndarray) -> np.ndarray:
  return plan.calibration.iid_noise_std * (hybrid_noising_matrix(
      plan.tau, plan.strategies) @ latent)


def _warmup(seed, tau, noise, target):
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=.04, beta1=.8, beta2=.9, eps=1e-6, weight_decay=.02)
  params = jnp.asarray(np.random.default_rng(seed + 101).normal(0, .05, 3))
  state = optimizer.init(params)
  previous_p = None
  history = []
  for step in range(tau):
    gradient = jnp.asarray(_clean_gradient(params, step, target) + noise[step])
    updates, state = optimizer.update(gradient, state, params)
    params = optax.apply_updates(params, updates)
    p = p_star_from_optax(state, beta2=.9, eps=1e-6)
    if step == tau - 2:
      previous_p = p
    loss, accuracy = _evaluate(params, target)
    history.append({"step": step + 1, "test_loss": loss, "test_accuracy": accuracy})
  if previous_p is None:
    raise ValueError("smoke tau must be at least two")
  return params, state, previous_p, history


def _phase(params, optax_state, noise, tau, horizon, target, *, frozen):
  history = []
  if frozen:
    optimizer = FrozenPAdamW(.04, .8, .02)
    state = freeze_optax_adamw(optax_state, beta2=.9, eps=1e-6)
  else:
    optimizer = make_nonamplified_dpadamw_optimizer(
        learning_rate=.04, beta1=.8, beta2=.9, eps=1e-6, weight_decay=.02)
    state = optax_state
  for step in range(tau, horizon):
    gradient = jnp.asarray(_clean_gradient(params, step, target) + noise[step])
    updates, state = optimizer.update(gradient, state, params)
    params = optax.apply_updates(params, updates)
    loss, accuracy = _evaluate(params, target)
    history.append({"step": step + 1, "test_loss": loss, "test_accuracy": accuracy})
  return params, state, history


def _summary_row(seed, tau, switch_stats, switch_metrics, history):
  losses = [row["test_loss"] for row in history]
  accuracies = [row["test_accuracy"] for row in history]
  return {
      "seed": seed, "tau": tau, **switch_stats,
      "switch_test_accuracy": switch_metrics["test_accuracy"],
      "final_test_loss": losses[-1], "final_test_accuracy": accuracies[-1],
      "best_test_loss": min(losses), "best_test_accuracy": max(accuracies),
      "post_switch_accuracy_gain": accuracies[-1] - switch_metrics["test_accuracy"],
  }


def run_smoke(out: Path, seeds: list[int], tau_candidates: list[int]) -> int:
  horizon = 10
  candidates = sorted(set(tau_candidates))
  if candidates == DEFAULT_TAUS:
    candidates = [2, 4]
  if len(candidates) < 2 or any(tau < 2 or tau >= horizon for tau in candidates):
    raise ValueError("smoke requires at least two tau candidates in [2, 9]")
  target = np.asarray([.45, -.25, .15])
  switch_rows = []
  for seed in seeds:
    latent = np.random.default_rng(seed + 5000).normal(size=(horizon, 3))
    for tau in candidates:
      plan = _plans(horizon, tau, segmented=True)
      noise = _noise(plan, latent)
      params, state, previous_p, warm_history = _warmup(seed, tau, noise, target)
      p = p_star_from_optax(state, beta2=.9, eps=1e-6)
      switch_stats = _p_stats(p, previous_p)
      switch_stats["switch_p_weighted_objective"] = p_aware_hybrid_objective(
          plan, p, beta1=.8, learning_rate=.04, weight_decay=.02)
      _, _, phase_history = _phase(params, state, noise, tau, horizon, target, frozen=True)
      history = warm_history + phase_history
      _write_csv(out / f"metrics_switch_tau{tau}_seed{seed}.csv", history)
      switch_rows.append(_summary_row(seed, tau, switch_stats, warm_history[-1], history))
  _write_csv(out / "switch_comparison.csv", switch_rows)
  tau_star = select_tau_star(switch_rows)
  threshold = float(np.mean([row["switch_relative_change"] for row in switch_rows
                             if row["tau"] == tau_star]))
  switch_summary = {
      "smoke": True, "tau_candidates": candidates, "tau_star": tau_star,
      "selection": "highest mean final_test_accuracy; lower mean final_test_loss breaks ties",
      "empirical_stability_threshold": threshold,
      "aggregate": _aggregate(switch_rows, "tau", ["final_test_accuracy", "final_test_loss",
                                                     "switch_relative_change"]),
      "privacy": {"epsilon": 3., "delta": 1e-5, "adjacency": "add_remove",
                  "calibration_scope": "one full hybrid transcript per run"},
  }
  (out / "switch_summary.json").write_text(json.dumps(switch_summary, indent=2), encoding="utf-8")

  comparison = []
  replay_rows = []
  plans = share_conservative_calibration({
      "cont": _plans(horizon, tau_star, segmented=False),
      "seg97": _plans(horizon, tau_star, segmented=True)})
  for seed in seeds:
    latent = np.random.default_rng(seed + 9000).normal(size=(horizon, 3))
    for mechanism, plan in plans.items():
      noise = _noise(plan, latent)
      params, state, _, warm_history = _warmup(seed, tau_star, noise, target)
      # Dynamic and frozen receive bit-identical Phase-II perturbations.
      for kind in ("dynamic", "frozen"):
        condition = f"{kind}_{mechanism}"
        _, _, phase_history = _phase(
            params, state, noise, tau_star, horizon, target, frozen=kind == "frozen")
        history = warm_history + phase_history
        _write_csv(out / f"metrics_{condition}_seed{seed}.csv", history)
        losses = [row["test_loss"] for row in history]
        accuracies = [row["test_accuracy"] for row in history]
        comparison.append({
            "seed": seed, "condition": condition,
            "final_test_loss": losses[-1], "final_test_accuracy": accuracies[-1],
            "best_test_loss": min(losses), "best_test_accuracy": max(accuracies),
        })
      frozen_state = freeze_optax_adamw(state, beta2=.9, eps=1e-6)
      clean = []
      clean_params = np.asarray(params)
      clean_m = np.asarray(frozen_state.mu)
      for step in range(tau_star, horizon):
        gradient = _clean_gradient(clean_params, step, target)
        clean.append(gradient)
        global_t = step + 1
        clean_m = .8 * clean_m + .2 * gradient
        clean_params = .9992 * clean_params - .04 * np.asarray(frozen_state.p_star) \
            * clean_m / (1. - .8 ** global_t)
      phase_noise = noise[tau_star:]
      result = paired_replay(
          np.asarray(clean), phase_noise, params=np.asarray(params),
          mu=np.asarray(frozen_state.mu), nu=np.asarray(frozen_state.frozen_nu),
          count=tau_star, beta1=.8, beta2=.9, eps=1e-6,
          learning_rate=.04, weight_decay=.02)
      replay_rows.append({"seed": seed, "mechanism": mechanism,
                          "G_dynamic": result.g_dynamic, "G_frozen": result.g_frozen})
  _write_csv(out / "nonlinearity_comparison.csv", comparison)
  deltas = []
  for seed in seeds:
    values = {row["condition"]: row["final_test_accuracy"] for row in comparison
              if row["seed"] == seed}
    deltas.append({"seed": seed,
                   "delta_seg_dynamic": values["dynamic_seg97"] - values["dynamic_cont"],
                   "delta_seg_frozen": values["frozen_seg97"] - values["frozen_cont"]})
  nonlinear_summary = {
      "smoke": True, "tau_star": tau_star,
      "aggregate": _aggregate(comparison, "condition", ["final_test_accuracy", "final_test_loss"]),
      "segmentation_deltas": deltas,
      "paired_phase_latent_draws": True,
      "identical_iid_warmup_draws": True,
      "shared_worst_case_sensitivity_squared": max(
          plan.sensitivity_squared for plan in plans.values()),
  }
  (out / "nonlinearity_summary.json").write_text(
      json.dumps(nonlinear_summary, indent=2), encoding="utf-8")
  _write_csv(out / "replay_results.csv", replay_rows)
  (out / "replay_summary.json").write_text(json.dumps({
      "smoke": True, "tau_star": tau_star,
      "aggregate": _aggregate(replay_rows, "mechanism", ["G_dynamic", "G_frozen"]),
      "meaning": "Only dynamic second-moment/preconditioner nonlinearity is measured.",
  }, indent=2), encoding="utf-8")
  return tau_star


def run_real(config_path: str | Path, out: Path, seeds: list[int], tau_candidates: list[int]):
  """Run the formal CIFAR-10 experiment after validating its fixed contract."""
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  if (config.epsilon, config.delta, config.adjacency) != (3.0, 1e-5, "add_remove"):
    raise ValueError("Exp5 formal runs require epsilon=3, delta=1e-5, add_remove")
  if tau_candidates != DEFAULT_TAUS:
    raise ValueError("formal Exp5 requires tau candidates 32,48,64,80,97")
  from exp5.full_training import run
  run(config, out, seeds, tau_candidates)


def main(argv=None):
  args = parse_args(argv)
  out = resolve_repo_path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  if args.smoke:
    tau_star = run_smoke(out, args.seeds, args.tau_candidates)
    print(f"Exp5 smoke complete: tau_star={tau_star}, output={out}")
  else:
    run_real(args.config, out, args.seeds, args.tau_candidates)


if __name__ == "__main__":
  main()
