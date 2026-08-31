#!/usr/bin/env python3
"""Run Experiment 8's one-trajectory, paired shadow mechanism diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.optim import decayed_prefix_sum_workload_coef
from dp_muon.privacy import (
    ParticipationSpec,
    calibrate_nonamplified_bandinv,
)
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import (
    get_or_fit_strategy_snapshot,
    load_cifar10_bandinv_dpadamw_config,
)
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule,
    cross_entropy_loss,
    run_training,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from exp2.common import derive_contract, resolve_repo_path
from exp8.core import (
    PATHS,
    bandinv_marginal_variances,
    init_exp8_train_state,
    make_exp8_train_step,
)
from exp8.diagnostics import (
    add_paired_gains,
    aggregate_window_rows,
    attach_path_degradation,
    write_window_rows,
    write_window_summary,
)
from exp8.online_shadow import Exp8WindowCollector
from exp8.plotting import plot_decomposition, plot_gain_over_steps, plot_path_gain_summary


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--output-dir", default="exp8/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, (np.integer, np.floating)):
    value = value.item()
  if isinstance(value, float) and not np.isfinite(value):
    return 0.0
  return value


def _stage_payload(stage: Mapping[str, object]) -> dict[str, object]:
  metrics = stage["metrics"]
  assert isinstance(metrics, Mapping)
  payload = attach_path_degradation(metrics)  # type: ignore[arg-type]
  payload["start_step"] = int(stage["start_step"])
  payload["end_step"] = int(stage["end_step"])
  payload["num_steps"] = int(stage["num_steps"])
  payload["decomposition"] = dict(stage["decomposition"])
  flat_paths = {}
  metric_tree = payload["metrics"]
  assert isinstance(metric_tree, Mapping)
  for path in PATHS:
    corr = metric_tree["corr"][path]
    iid = metric_tree["iid"][path]
    flat_paths[path] = {
        "C_corr": float(corr.get("C", 0.0)),
        "C_iid": float(iid.get("C", 0.0)),
        "J_corr": float(corr.get("J", 0.0)),
        "J_iid": float(iid.get("J", 0.0)),
        "D_corr": float(corr.get("D", 0.0)),
        "D_iid": float(iid.get("D", 0.0)),
        "G_C": float(corr.get("G_C", 0.0)),
        "G_J": float(corr.get("G_J", 0.0)),
    }
  payload["paths"] = flat_paths
  return payload


def _cross_seed_aggregate(per_seed: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
  output: dict[str, object] = {}
  for stage in ("early", "late", "full"):
    stage_values = [runs[stage] for runs in per_seed.values() if stage in runs]
    if not stage_values:
      output[stage] = {}
      continue
    paths = {}
    for path in PATHS:
      paths[path] = {}
      for field in ("C_corr", "C_iid", "J_corr", "J_iid", "D_corr", "D_iid", "G_C", "G_J"):
        values = np.asarray([float(stage_value["paths"][path][field]) for stage_value in stage_values])
        paths[path][f"{field}_mean"] = float(np.mean(values))
        paths[path][f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    degradation_values = {}
    for gain in ("G_C", "G_J"):
      for field in ("delta_clean", "delta_bias", "delta_nonlinear"):
        values = [float(stage_value["degradation"][gain][field]) for stage_value in stage_values]
        degradation_values[f"{gain}_{field}_mean"] = float(np.mean(values))
        degradation_values[f"{gain}_{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    output[stage] = {"num_seeds": len(stage_values), "paths": paths, "degradation": degradation_values}
  return output


def _write_outputs(
    output: Path,
    rows: list[dict[str, object]],
    per_seed: dict[str, dict[str, object]],
    *,
    metadata: dict[str, object],
) -> dict[str, object]:
  write_window_rows(output / "window_diagnostics_all.csv", rows)
  write_window_summary(output / "window_summary.csv", aggregate_window_rows(rows))
  plot_gain_over_steps(rows, output / "correlation_gain_over_steps.png", gain="G_C")
  plot_gain_over_steps(rows, output / "endpoint_gain_over_steps.png", gain="G_J")
  # Cross-seed plots use the same raw-stage metrics, not averages of window C/J.
  aggregate = _cross_seed_aggregate(per_seed)
  plot_summaries = {}
  for stage in ("early", "late", "full"):
    stage_runs = [run[stage] for run in per_seed.values() if stage in run]
    if stage_runs:
      # A plot only needs a representative stage; the JSON retains every seed.
      plot_summaries[stage] = stage_runs[0]
  plot_path_gain_summary(plot_summaries, output / "path_gain_summary.png")
  plot_decomposition(plot_summaries, output / "privacy_clean_decomposition.png")
  summary = {
      **metadata,
      "seeds": [int(seed) for seed in per_seed],
      "per_seed": per_seed,
      "cross_seed_aggregate": aggregate,
      "num_window_rows": len(rows),
      "window_size": 16,
      "stage_aggregation": "J,D,C,G are recomputed from exact raw steps for each stage; window summaries are descriptive only",
  }
  (output / "summary.json").write_text(
      json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
  )
  return summary


def _run_one(
    *,
    seed: int,
    params: Any,
    training_key: jax.Array,
    diagnostic_key: jax.Array,
    strategy: Any,
    calibration: Any,
    participation: ParticipationSpec,
    batches: Any,
    horizon: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    microbatch_size: int | None,
    output: Path,
    loss_fn: Any,
) -> tuple[list[dict[str, object]], dict[str, object]]:
  train_step, optimizer = make_exp8_train_step(
      loss_fn, strategy, calibration, participation,
      learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
      weight_decay=weight_decay, microbatch_size=microbatch_size,
  )
  state = init_exp8_train_state(
      params, strategy, training_key, optimizer, diagnostic_rng_key=diagnostic_key
  )
  collector = Exp8WindowCollector(
      params, seed=seed, learning_rate=learning_rate,
      weight_decay=weight_decay, horizon=horizon,
  )
  final, _ = run_training(
      initial_state=state,
      train_step=train_step,
      logical_batches=batches,
      horizon=horizon,
      experiment_config={"experiment": "exp8", "seed": seed},
      artifact_identifiers={"experiment": "exp8", "diagnostic": "paired-bandinv-iid"},
      eval_every=horizon,
      after_step=collector.after_step,
  )
  del final
  rows = collector.finalize()
  write_window_rows(output / f"window_diagnostics_seed{seed}.csv", rows)
  stages = {
      name: _stage_payload(values)
      for name, values in collector.stage_summaries().items()
  }
  return rows, stages


def run_real(config_path: str | Path, output: Path, seeds: list[int]) -> None:
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  train_images, train_labels = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  contract = derive_contract(config, num_examples=len(train_images))
  participation = ParticipationSpec(contract.horizon, contract.min_sep, contract.max_participations)
  snapshot, action = get_or_fit_strategy_snapshot(config, participation)
  strategy = snapshot.strategy
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  output.mkdir(parents=True, exist_ok=True)
  all_rows: list[dict[str, object]] = []
  per_seed: dict[str, dict[str, object]] = {}
  model = ViTTiny()
  for seed in seeds:
    schedule = build_fixed_cycle_logical_schedule(
        num_examples=contract.num_examples, batch_size=contract.batch_size,
        horizon=contract.horizon, min_sep=contract.min_sep,
        max_participations=contract.max_participations, seed=seed,
    )
    # Preserve the existing Exp7/baseline parameter and training streams
    # (split into exactly two keys); derive the shadow stream separately.
    base_key = jax.random.key(seed)
    parameter_key, training_key = jax.random.split(base_key)
    diagnostic_key = jax.random.fold_in(base_key, 8_000_008)
    pretrained = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key)
    rows, stages = _run_one(
        seed=seed, params=pretrained.params, training_key=training_key,
        diagnostic_key=diagnostic_key, strategy=strategy, calibration=calibration,
        participation=participation,
        batches=iter_logical_batches(train_images, train_labels, schedule),
        horizon=contract.horizon, learning_rate=config.learning_rate,
        beta1=config.beta1, beta2=config.beta2, eps=config.eps,
        weight_decay=config.weight_decay, microbatch_size=config.microbatch_size,
        output=output,
        loss_fn=lambda params, batch: cross_entropy_loss(params, batch, model),
    )
    all_rows.extend(rows)
    per_seed[str(seed)] = stages
  phi = bandinv_marginal_variances(strategy, calibration.iid_noise_std)
  metadata = {
      "experiment": "exp8",
      "smoke": False,
      "config": asdict(config),
      "contract": asdict(contract),
      "privacy_calibration": asdict(calibration),
      "bandinv_strategy": {
          "artifact": str(snapshot.path.resolve()), "sha256": snapshot.sha256,
          "action": action, "horizon": strategy.horizon,
          "bandwidth": strategy.bandwidth, "min_sep": strategy.min_sep,
          "max_participations": strategy.max_participations,
          "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
          "strategy_coef_C": np.asarray(strategy.strategy_coef).tolist(),
          "sensitivity_squared": float(strategy.sensitivity_squared),
          "objective": float(strategy.objective),
      },
      "phi_t": np.asarray(phi).tolist(),
      "phi_t_definition": "phi_t = iid_noise_std^2 * sum of squares of the first min(t+1, bandwidth) C^-1 coefficients",
      "diagnostic_control": "matched-marginal IID mechanism diagnostic only; not a formal same-guarantee DP baseline",
      "trajectory": "one real correlated DP-AdamW baseline trajectory per seed; diagnostic shadows never update params or optimizer state",
  }
  _write_outputs(output, all_rows, per_seed, metadata=metadata)


def run_smoke(output: Path, seeds: list[int]) -> None:
  horizon = 20
  learning_rate, weight_decay, beta1, beta2, eps = .02, .01, .9, .9, 1e-6
  from dp_muon.bandinvmf import fit_bandinv_strategy

  strategy = fit_bandinv_strategy(
      horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(horizon, learning_rate, weight_decay),
      max_optimizer_steps=3,
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(horizon, 1, 1)

  def loss(params, batch):
    prediction = jnp.dot(params["w"], batch["x"][0])
    return .5 * (prediction - batch["y"][0]) ** 2

  batches = [
      {"x": np.asarray([[1., step / horizon], [1., -step / horizon]], np.float32),
       "y": np.asarray([.5, -.25], np.float32)} for step in range(horizon)
  ]
  output.mkdir(parents=True, exist_ok=True)
  all_rows: list[dict[str, object]] = []
  per_seed: dict[str, dict[str, object]] = {}
  for seed in seeds:
    params = {"w": jax.random.normal(jax.random.key(seed), (2,)) * .01}
    training_key, diagnostic_key = jax.random.split(jax.random.key(seed + 10_000))
    rows, stages = _run_one(
        seed=seed, params=params, training_key=training_key,
        diagnostic_key=diagnostic_key, strategy=strategy, calibration=calibration,
        participation=participation, batches=list(batches), horizon=horizon,
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
        weight_decay=weight_decay, microbatch_size=None, output=output, loss_fn=loss,
    )
    all_rows.extend(rows)
    per_seed[str(seed)] = stages
  phi = bandinv_marginal_variances(strategy, calibration.iid_noise_std)
  _write_outputs(
      output, all_rows, per_seed,
      metadata={
          "experiment": "exp8", "smoke": True,
          "config": {"horizon": horizon, "learning_rate": learning_rate,
                     "beta1": beta1, "beta2": beta2, "eps": eps,
                     "weight_decay": weight_decay},
          "privacy_calibration": asdict(calibration),
          "bandinv_strategy": {
              "horizon": strategy.horizon, "bandwidth": strategy.bandwidth,
              "min_sep": strategy.min_sep, "max_participations": strategy.max_participations,
              "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
          },
          "phi_t": np.asarray(phi).tolist(),
          "phi_t_definition": "phi_t = iid_noise_std^2 * sum of squares of the causal C^-1 row",
          "diagnostic_control": "matched-marginal IID mechanism diagnostic only; not a formal same-guarantee DP baseline",
      },
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
