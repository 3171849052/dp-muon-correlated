"""Formal CIFAR-10 execution for Experiment 5.

This module is intentionally separate from the small smoke driver so importing
the mathematical helpers never loads the model or dataset.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dp_muon.bandinvmf import filter_latent_noise, init_bandinv_noise_state
from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.privacy import make_clipped_gradient_query
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule, cross_entropy_loss,
    evaluate_classifier_metrics,
)
from dp_muon.training.nonamplified_dpadamw import make_nonamplified_dpadamw_optimizer
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from exp2.common import derive_contract, resolve_repo_path
from exp5.hybrid_optimizer import FrozenPAdamW, freeze_optax_adamw, p_star_from_optax
from exp5.hybrid_strategy import (
    HybridPlan, fit_hybrid_plan, p_aware_hybrid_objective,
)
from exp5.replay import FullPyTreeReplayAccumulator
from exp5.run import _aggregate, _p_stats, _write_csv, select_tau_star


def _latent_like(key, tree, std):
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  keys = jax.random.split(key, len(leaves))
  values = [jax.random.normal(k, x.shape, x.dtype) * jnp.asarray(std, x.dtype)
            for k, x in zip(keys, leaves, strict=True)]
  return jax.tree_util.tree_unflatten(treedef, values)


def _result(seed, condition, history):
  losses = [row["test_loss"] for row in history]
  accuracies = [row["test_accuracy"] for row in history]
  return {"seed": seed, "condition": condition,
          "final_test_loss": losses[-1], "final_test_accuracy": accuracies[-1],
          "best_test_loss": min(losses), "best_test_accuracy": max(accuracies)}


def _log(message: str) -> None:
  print(f"[Exp5] {message}", flush=True)


def _fit_switch_plans(
    common: dict[str, Any], tau_candidates: list[int]
) -> dict[int, HybridPlan]:
  """Fit each deterministic 5A temporal plan once for all seeds."""
  _log("fitting 5A plans...")
  plans = {}
  for tau in tau_candidates:
    plans[tau] = fit_hybrid_plan(tau=tau, block_size=97, **common)
    _log(f"fitted tau={tau}")
  return plans


def _train(config, contract, plan, seed, *, dynamic, train_x, train_y, test_x,
           test_y, model, schedule, collect_replay=False):
  parameter_key = jax.random.key(seed)
  params = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key).params
  optimizer = make_nonamplified_dpadamw_optimizer(
      learning_rate=config.learning_rate, beta1=config.beta1, beta2=config.beta2,
      eps=config.eps, weight_decay=config.weight_decay)
  optimizer_state = optimizer.init(params)
  frozen_optimizer = FrozenPAdamW(config.learning_rate, config.beta1, config.weight_decay)
  frozen_state = None
  query = jax.jit(make_clipped_gradient_query(
      lambda p, b: cross_entropy_loss(p, b, model), clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), batch_argnums=1, keep_batch_dim=True,
      microbatch_size=config.microbatch_size))
  batches = iter_logical_batches(train_x, train_y, schedule)
  noise_root = jax.random.key(seed + 1_000_003)
  noise_state = None
  block_index = 0
  block_end = plan.tau + plan.block_lengths[0]
  history, previous_p = [], None
  switch_params = switch_state = None
  replay = None
  for step, batch in enumerate(batches):
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    clipped = query(params, batch)
    latent = _latent_like(jax.random.fold_in(noise_root, step), clipped,
                          plan.calibration.iid_noise_std)
    if step < plan.tau:
      perturbation = latent
    else:
      if step == plan.tau or step == block_end:
        if step == block_end:
          block_index += 1
          block_end += plan.block_lengths[block_index]
        noise_state = init_bandinv_noise_state(params, plan.strategies[block_index].bandwidth)
      perturbation, noise_state = filter_latent_noise(
          noise_state, latent, plan.strategies[block_index].noising_coef)
    private = jax.tree_util.tree_map(lambda g, n: g + n, clipped, perturbation)
    if step < plan.tau or dynamic:
      updates, optimizer_state = optimizer.update(private, optimizer_state, params)
    else:
      assert frozen_state is not None
      updates, frozen_state = frozen_optimizer.update(private, frozen_state, params)
    params = optax.apply_updates(params, updates)
    if step < plan.tau:
      p = p_star_from_optax(optimizer_state, beta2=config.beta2, eps=config.eps)
      if step == plan.tau - 2:
        previous_p = p
      if step == plan.tau - 1:
        frozen_state = freeze_optax_adamw(
            optimizer_state, beta2=config.beta2, eps=config.eps)
        switch_params, switch_state = params, frozen_state
        if collect_replay:
          replay = FullPyTreeReplayAccumulator(
              params=params, dynamic_optimizer=optimizer,
              dynamic_state=optimizer_state, frozen_state=frozen_state,
              learning_rate=config.learning_rate, beta1=config.beta1,
              weight_decay=config.weight_decay)
    elif collect_replay:
      assert replay is not None
      replay.update(clipped, perturbation)
    current = step + 1
    should_eval = (current == plan.tau or current == contract.horizon or
                   current * config.batch_size // contract.num_examples !=
                   step * config.batch_size // contract.num_examples)
    if should_eval:
      metrics = evaluate_classifier_metrics(
          params, model, test_x, test_y, batch_size=config.batch_size)
      history.append({"step": current, **metrics})
  assert previous_p is not None and switch_state is not None and switch_params is not None
  diagnostics = _p_stats(switch_state.p_star, previous_p)
  diagnostics["switch_p_weighted_objective"] = p_aware_hybrid_objective(
      plan, switch_state.p_star, beta1=config.beta1,
      learning_rate=config.learning_rate, weight_decay=config.weight_decay,
      reduction=config.reduction)
  replay_result = replay.result() if replay is not None else None
  return params, history, diagnostics, switch_params, switch_state, replay_result


def run(config, out: Path, seeds: list[int], tau_candidates: list[int]):
  train_x, train_y = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  test_x, test_y = load_cifar10(resolve_repo_path(config.data_dir), train=False)
  contract = derive_contract(config, num_examples=len(train_x))
  model = ViTTiny()
  common = dict(
      horizon=contract.horizon, bandwidth=config.bandwidth,
      min_sep=contract.min_sep, max_participations=contract.max_participations,
      learning_rate=config.learning_rate, beta1=config.beta1,
      weight_decay=config.weight_decay, epsilon=config.epsilon, delta=config.delta,
      clip_norm=config.clip_norm, normalize_by=float(config.batch_size),
      adjacency=config.adjacency, max_optimizer_steps=config.max_optimizer_steps,
      reduction=config.reduction)
  switch_plans = _fit_switch_plans(common, tau_candidates)
  switch_rows = []
  for seed in seeds:
    schedule = build_fixed_cycle_logical_schedule(
        num_examples=len(train_x), batch_size=config.batch_size,
        horizon=contract.horizon, min_sep=contract.min_sep,
        max_participations=contract.max_participations, seed=seed)
    for tau in tau_candidates:
      _log(f"training 5A seed={seed} tau={tau}")
      plan = switch_plans[tau]
      _, history, stats, _, _, _ = _train(
          config, contract, plan, seed, dynamic=False, train_x=train_x,
          train_y=train_y, test_x=test_x, test_y=test_y, model=model, schedule=schedule)
      _write_csv(out / f"metrics_switch_tau{tau}_seed{seed}.csv", history)
      switch = next(row for row in history if row["step"] == tau)
      row = _result(seed, f"tau{tau}", history)
      switch_rows.append({"seed": seed, "tau": tau, **stats,
                          "switch_test_accuracy": switch["test_accuracy"],
                          **{k: row[k] for k in ("final_test_loss", "final_test_accuracy",
                                                "best_test_loss", "best_test_accuracy")},
                          "post_switch_accuracy_gain": row["final_test_accuracy"] - switch["test_accuracy"]})
  _write_csv(out / "switch_comparison.csv", switch_rows)
  tau_star = select_tau_star(switch_rows)
  _log(f"selected tau_star={tau_star}")
  threshold = float(np.mean([r["switch_relative_change"] for r in switch_rows if r["tau"] == tau_star]))
  (out / "switch_summary.json").write_text(json.dumps({
      "smoke": False, "tau_star": tau_star, "tau_candidates": tau_candidates,
      "selection": "highest mean final_test_accuracy; lower mean final_test_loss breaks ties",
      "empirical_stability_threshold": threshold,
      "contract": asdict(contract),
      "privacy": {"epsilon": config.epsilon, "delta": config.delta,
                  "adjacency": config.adjacency,
                  "calibration_scope": "one full hybrid transcript per run"},
      "aggregate": _aggregate(switch_rows, "tau", ["final_test_accuracy", "final_test_loss",
                                                     "switch_relative_change"]),
  }, indent=2), encoding="utf-8")

  _log("fitting 5B continuous/seg97 plans...")
  plans = {
      "cont": fit_hybrid_plan(tau=tau_star, block_size=None, **common),
      "seg97": fit_hybrid_plan(tau=tau_star, block_size=97, **common),
  }
  _log("fitted 5B continuous/seg97 plans")
  comparison, replay_rows = [], []
  for seed in seeds:
    schedule = build_fixed_cycle_logical_schedule(
        num_examples=len(train_x), batch_size=config.batch_size,
        horizon=contract.horizon, min_sep=contract.min_sep,
        max_participations=contract.max_participations, seed=seed)
    for mechanism, plan in plans.items():
      for dynamic in (True, False):
        condition = ("dynamic_" if dynamic else "frozen_") + mechanism
        _log(f"training condition={condition} seed={seed}")
        _, history, _, _, _, replay = _train(
            config, contract, plan, seed, dynamic=dynamic, train_x=train_x,
            train_y=train_y, test_x=test_x, test_y=test_y, model=model,
            schedule=schedule, collect_replay=not dynamic)
        _write_csv(out / f"metrics_{condition}_seed{seed}.csv", history)
        comparison.append(_result(seed, condition, history))
        if not dynamic:
          assert replay is not None
          replay_rows.append({"seed": seed, "mechanism": mechanism,
                              "G_dynamic": replay.g_dynamic,
                              "G_frozen": replay.g_frozen,
                              "numerator_dynamic": replay.numerator_dynamic,
                              "numerator_frozen": replay.numerator_frozen,
                              "denominator": replay.denominator})
  _write_csv(out / "nonlinearity_comparison.csv", comparison)
  deltas = []
  for seed in seeds:
    values = {r["condition"]: r["final_test_accuracy"] for r in comparison if r["seed"] == seed}
    deltas.append({"seed": seed,
                   "delta_seg_dynamic": values["dynamic_seg97"] - values["dynamic_cont"],
                   "delta_seg_frozen": values["frozen_seg97"] - values["frozen_cont"]})
  (out / "nonlinearity_summary.json").write_text(json.dumps({
      "smoke": False, "tau_star": tau_star,
      "aggregate": _aggregate(comparison, "condition", ["final_test_accuracy", "final_test_loss"]),
      "segmentation_deltas": deltas,
      "calibration_statement": (
          "each mechanism independently calibrated to the same final privacy target"),
      "mechanisms": {
          name: {
              "sensitivity_squared": plan.sensitivity_squared,
              "iid_noise_std": plan.calibration.iid_noise_std,
              "epsilon": plan.calibration.epsilon,
              "delta": plan.calibration.delta,
          } for name, plan in plans.items()
      },
      "paired_within_mechanism": True,
      "shared_base_gaussian_seeds_across_mechanisms": True,
      "exact_equal_final_privacy_target": True,
  }, indent=2), encoding="utf-8")
  _write_csv(out / "replay_results.csv", replay_rows)
  (out / "replay_summary.json").write_text(json.dumps({
      "smoke": False, "tau_star": tau_star,
      "aggregate": _aggregate(replay_rows, "mechanism", ["G_dynamic", "G_frozen"]),
      "meaning": (
          "Full-model/full-PyTree online replay of dynamic and frozen-p "
          "optimizer perturbations against the exact frozen-p linear response."),
  }, indent=2), encoding="utf-8")
