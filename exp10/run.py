#!/usr/bin/env python3
"""Run Experiment 10: two real closed-loop DP-AdamW trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
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
from exp10.core import bandinv_marginal_variances, init_exp10_train_state, make_exp10_train_step
from exp10.diagnostics import (
    Exp10Collector,
    aggregate_stage_rows,
    save_histograms,
    write_stage_metrics,
    write_step_metrics,
)
from exp10.plotting import plot_histograms


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--output-dir", default="exp10/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
  parser.add_argument("--histogram-bins", type=int, default=64)
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
  if value is None:
    return None
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, (np.integer, np.floating)):
    value = value.item()
  if isinstance(value, (np.bool_, bool)):
    return bool(value)
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def _strategy_metadata(strategy: Any, *, artifact: Path | None = None,
                       action: str | None = None, sha256: str | None = None,
                       config: Any | None = None) -> dict[str, object]:
  result: dict[str, object] = {
      "workload_type": "decayed-prefix-sum",
      "horizon": int(strategy.horizon),
      "bandwidth": int(strategy.bandwidth),
      "min_sep": int(strategy.min_sep),
      "max_participations": (
          None if strategy.max_participations is None
          else int(strategy.max_participations)
      ),
      "sensitivity_squared": float(strategy.sensitivity_squared),
      "objective": float(strategy.objective),
      "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
      "strategy_coef_C": np.asarray(strategy.strategy_coef).tolist(),
      "workload_representation": (
          "matrix" if strategy.workload_matrix is not None else "coef"
      ),
  }
  if config is not None:
    result.update({
        "learning_rate": float(config.learning_rate),
        "beta1": float(config.beta1),
        "beta2": float(config.beta2),
        "weight_decay": float(config.weight_decay),
        "reduction": config.reduction,
        "max_optimizer_steps": int(config.max_optimizer_steps),
    })
  if artifact is not None:
    result["artifact"] = str(artifact.resolve())
  if action is not None:
    result["action"] = action
  if sha256 is not None:
    result["sha256"] = sha256
  return result


def _run_one(
    *,
    seed: int,
    params: Any,
    rng_key: jax.Array,
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
    histogram_bins: int,
    loss_fn: Any,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, Any]]]:
  train_step, optimizer = make_exp10_train_step(
      loss_fn,
      strategy,
      calibration,
      participation,
      learning_rate=learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay,
      microbatch_size=microbatch_size,
  )
  state = init_exp10_train_state(params, strategy, rng_key, optimizer)
  collector = Exp10Collector(
      params,
      seed=seed,
      horizon=horizon,
      histogram_bins=histogram_bins,
  )
  run_training(
      initial_state=state,
      train_step=train_step,
      logical_batches=batches,
      horizon=horizon,
      experiment_config={"experiment": "exp10", "seed": seed},
      artifact_identifiers={"experiment": "exp10", "strategy": "bandinv-naive"},
      eval_every=horizon,
      after_step=collector.after_step,
  )
  return collector.rows, collector.stage_rows(), collector.histogram_records


def _write_outputs(
    output: Path,
    step_rows: list[dict[str, object]],
    stage_rows: list[dict[str, object]],
    histogram_records: list[dict[str, Any]],
    *,
    metadata: dict[str, object],
    histogram_bins: int,
) -> dict[str, object]:
  output.mkdir(parents=True, exist_ok=True)
  write_step_metrics(output / "step_metrics.csv", step_rows)
  write_stage_metrics(output / "stage_metrics.csv", stage_rows)
  save_histograms(
      output / "histograms.npz", histogram_records, histogram_bins=histogram_bins
  )
  if histogram_records:
    plot_histograms(output / "histograms.npz", output / "histograms.png")
  cross_seed = aggregate_stage_rows(stage_rows)
  expectation_checks = {
      "iid_negative_control_E[g2_cross_minus_g2]": {
          stage: float(cross_seed[stage]["iid"].get(
              "mean_g2_cross_minus_g2_mean", 0.0
          ))
          for stage in ("early", "late", "full")
      },
      "mf_feedback_E[2*g*xi]": {
          stage: float(cross_seed[stage]["mf"].get("mean_2gxi_mean", 0.0))
          for stage in ("early", "late", "full")
      },
      "private_v_hat_minus_V_g_cross_minus_V_xi_max_abs": {
          stage: {
              branch: float(cross_seed[stage][branch].get(
                  "private_v_decomposition_max_abs_mean", 0.0
              ))
              for branch in ("mf", "iid")
          }
          for stage in ("early", "late", "full")
      },
  }
  summary = {
      **metadata,
      "seeds": sorted({int(row["seed"]) for row in step_rows}),
      "num_step_rows": len(step_rows),
      "num_stage_rows": len(stage_rows),
      "num_histogram_checkpoints": len(histogram_records),
      "cross_seed_stage_aggregate": cross_seed,
      "expectation_checks": expectation_checks,
      "histogram_artifact": {
          "file": "histograms.npz",
          "format_version": "exp10-histograms-v1",
          "branches": ["mf", "iid"],
          "components": ["g2", "g2_cross", "xi2", "V_g", "V_g_cross", "V_xi"],
          "axes": {
              "seeds": "(K,)", "steps": "(K,)",
              "bin_edges": "(K, bins+1)",
              "counts": "(K, 2, 6, bins)",
              "relative_frequency": "(K, 2, 6, bins)",
          },
          "common_bins": "For every (seed, step), one edge vector is shared by both branches and all six components.",
          "negative_values": "g2_cross is histogrammed directly; no clipping or floor is applied.",
      },
  }
  (output / "summary.json").write_text(
      json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
  )
  return summary


def run_real(
    config_path: str | Path,
    output: Path,
    seeds: list[int],
    *,
    histogram_bins: int = 64,
) -> dict[str, object]:
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  train_images, train_labels = load_cifar10(
      resolve_repo_path(config.data_dir), train=True
  )
  contract = derive_contract(config, num_examples=len(train_images))
  participation = ParticipationSpec(
      contract.horizon, contract.min_sep, contract.max_participations
  )
  snapshot, action = get_or_fit_strategy_snapshot(config, participation)
  strategy = snapshot.strategy
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon,
      delta=config.delta,
      clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size),
      adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  output.mkdir(parents=True, exist_ok=True)
  model = ViTTiny()
  step_rows: list[dict[str, object]] = []
  stage_rows: list[dict[str, object]] = []
  histogram_records: list[dict[str, Any]] = []
  for seed in seeds:
    schedule = build_fixed_cycle_logical_schedule(
        num_examples=contract.num_examples,
        batch_size=contract.batch_size,
        horizon=contract.horizon,
        min_sep=contract.min_sep,
        max_participations=contract.max_participations,
        seed=seed,
    )
    base_key = jax.random.key(seed)
    parameter_key, rng_key = jax.random.split(base_key)
    pretrained = load_pretrained_snapshot(
        resolve_repo_path(config.pretrained), key=parameter_key
    )
    rows, stages, histograms = _run_one(
        seed=seed,
        params=pretrained.params,
        rng_key=rng_key,
        strategy=strategy,
        calibration=calibration,
        participation=participation,
        batches=iter_logical_batches(train_images, train_labels, schedule),
        horizon=contract.horizon,
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        microbatch_size=config.microbatch_size,
        histogram_bins=histogram_bins,
        loss_fn=lambda params, batch: cross_entropy_loss(params, batch, model),
    )
    step_rows.extend(rows)
    stage_rows.extend(stages)
    histogram_records.extend(histograms)
  phi = bandinv_marginal_variances(strategy, calibration.iid_noise_std)
  return _write_outputs(
      output,
      step_rows,
      stage_rows,
      histogram_records,
      metadata={
          "experiment": "exp10",
          "smoke": False,
          "config": asdict(config),
          "contract": asdict(contract),
          "privacy_calibration": asdict(calibration),
          "bandinv_strategy": _strategy_metadata(
              strategy,
              artifact=snapshot.path,
              action=action,
              sha256=snapshot.sha256,
              config=config,
          ),
          "phi_t": np.asarray(phi).tolist(),
          "phi_t_definition": "phi_t = Var(xi_mf_t) = iid_noise_std^2 * sum_{k=0}^{min(t+1, bandwidth)-1} noising_coef[k]^2",
          "trajectory": "two real closed-loop branches (mf and iid), each with its own parameters and AdamW moments",
          "noise_pairing": "one shared current standard-normal z_t; mf uses the causal BandInvMF filter and iid uses sqrt(phi_t) * z_t",
          "optimizer": "existing naive non-amplified DP-AdamW; no noise/oracle correction is applied; standard AdamW state is read only for the v_hat decomposition check",
          "diagnostics": "V_g, V_g_cross, and V_xi use beta2 exactly and are bias-corrected for reporting",
          "stage_definition": "early=1..min(97,horizon), late=98..horizon, full=1..horizon; stage ratios are recomputed from summed raw quantities",
      },
      histogram_bins=histogram_bins,
  )


def run_smoke(
    output: Path,
    seeds: list[int],
    *,
    histogram_bins: int = 32,
) -> dict[str, object]:
  """Run a tiny synthetic-data-shaped job without CIFAR or a checkpoint."""
  horizon = 20
  learning_rate, weight_decay, beta1, beta2, eps = .02, .01, .9, .9, 1e-6
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
    return .5 * (prediction - batch["y"][0]) ** 2

  batches = [
      {
          "x": np.asarray([[1.0, step / horizon], [1.0, -step / horizon]], np.float32),
          "y": np.asarray([.5, -.25], np.float32),
      }
      for step in range(horizon)
  ]
  step_rows: list[dict[str, object]] = []
  stage_rows: list[dict[str, object]] = []
  histogram_records: list[dict[str, Any]] = []
  for seed in seeds:
    params = {"w": jax.random.normal(jax.random.key(seed), (2,)) * .01}
    rng_key = jax.random.fold_in(jax.random.key(seed + 10_000), 1)
    rows, stages, histograms = _run_one(
        seed=seed,
        params=params,
        rng_key=rng_key,
        strategy=strategy,
        calibration=calibration,
        participation=participation,
        batches=list(batches),
        horizon=horizon,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        microbatch_size=None,
        histogram_bins=histogram_bins,
        loss_fn=loss,
    )
    step_rows.extend(rows)
    stage_rows.extend(stages)
    histogram_records.extend(histograms)
  phi = bandinv_marginal_variances(strategy, calibration.iid_noise_std)
  return _write_outputs(
      output,
      step_rows,
      stage_rows,
      histogram_records,
      metadata={
          "experiment": "exp10",
          "smoke": True,
          "config": {
              "horizon": horizon,
              "learning_rate": learning_rate,
              "beta1": beta1,
              "beta2": beta2,
              "eps": eps,
              "weight_decay": weight_decay,
          },
          "privacy_calibration": asdict(calibration),
          "bandinv_strategy": _strategy_metadata(
              strategy,
              config=type("SmokeConfig", (), {
                  "learning_rate": learning_rate,
                  "beta1": beta1,
                  "beta2": beta2,
                  "weight_decay": weight_decay,
                  "reduction": "mean",
                  "max_optimizer_steps": 3,
              })(),
          ),
          "phi_t": np.asarray(phi).tolist(),
          "trajectory": "two real closed-loop synthetic branches with separate AdamW moments",
          "noise_pairing": "one shared current z_t; causal MF filter versus matched-marginal IID scaling",
          "optimizer": "existing naive non-amplified DP-AdamW; no noise/oracle correction",
          "diagnostics": "beta2-consistent bias-corrected V_g, V_g_cross, V_xi and private_v_hat decomposition checks",
          "stage_definition": "early=1..min(97,horizon), late=98..horizon, full=1..horizon",
      },
      histogram_bins=histogram_bins,
  )


def main(argv=None):
  args = parse_args(argv)
  output = resolve_repo_path(args.output_dir)
  if args.smoke:
    if args.output_dir == "exp10/results":
      output = resolve_repo_path("exp10/results_smoke")
    run_smoke(output, args.seeds, histogram_bins=args.histogram_bins)
  else:
    run_real(args.config, output, args.seeds, histogram_bins=args.histogram_bins)


if __name__ == "__main__":
  main()
