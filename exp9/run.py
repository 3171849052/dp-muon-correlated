#!/usr/bin/env python3
"""Run Experiment 9's Muon nonlinear cancellation decomposition."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.bandinvmf import (
    fit_bandinv_strategy,
    load_bandinv_strategy,
    load_bandinv_strategy_metadata,
    save_bandinv_strategy,
)
from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.optim import fixed_lr_nesterov_decayed_trajectory_workload_coef
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training.cifar10_bandinv_dpmuon_experiment import (
    get_or_fit_strategy_snapshot,
    load_cifar10_bandinv_dpmuon_config,
)
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule,
    cross_entropy_loss,
    run_training,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from exp2.common import derive_contract, resolve_repo_path
from exp9.core import (
    PATHS,
    bandinv_marginal_variances,
    init_exp9_train_state,
    make_exp9_train_step,
    pre_q_marginal_variances,
)
from exp9.diagnostics import (
    aggregate_window_rows,
    attach_path_degradation,
    cross_seed_aggregate,
    write_window_rows,
    write_window_summary,
)
from exp9.online_shadow import Exp9WindowCollector
from exp9.plotting import (
    plot_bias_diagnostics,
    plot_cancellation_paths,
    plot_decomposition,
    plot_path_gain_summary,
    plot_stage_cancellation,
    plot_stage_diagnostics,
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="config/cifar10_bandinv_dpmuon_naive.yaml")
  parser.add_argument("--output-dir", default="exp9/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--bias-probes", type=int, default=8)
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
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def _stage_payload(stage: Mapping[str, object]) -> dict[str, object]:
  payload = attach_path_degradation(stage["metrics"])  # type: ignore[arg-type]
  payload["start_step"] = int(stage["start_step"])
  payload["end_step"] = int(stage["end_step"])
  payload["num_steps"] = int(stage["num_steps"])
  payload["decomposition"] = dict(stage["decomposition"])
  payload["bias"] = dict(stage["bias"])
  payload["stage_metrics"] = stage.get("stage_metrics", {})
  payload["stage_odd_response"] = stage.get("stage_odd_response", {})
  payload["secondary_stage_odd_response"] = stage.get(
      "secondary_stage_odd_response", {}
  )
  bias = payload["bias"]
  payload["P3_reliable"] = {
      "corr": bool(bias.get("P3_reliable_corr", False)),
      "iid": bool(bias.get("P3_reliable_iid", False)),
      "rule": "probe_error_to_P3_D <= 0.1 AND probe_error_to_P3_endpoint <= 0.1",
  }
  payload["delta_even_interpretation"] = {
      branch: (
          "reliable"
          if bool(bias.get(f"P3_reliable_{branch}", False))
          else "unreliable due to bias-probe Monte Carlo error"
      ) for branch in ("corr", "iid")
  }
  metric_tree = payload["metrics"]
  flat_paths = {}
  for path in PATHS:
    corr = metric_tree["corr"][path]  # type: ignore[index]
    iid = metric_tree["iid"][path]  # type: ignore[index]
    flat_paths[path] = {
        "C_corr": corr.get("C"), "C_iid": iid.get("C"),
        "J_corr": corr.get("J"), "J_iid": iid.get("J"),
        "D_corr": corr.get("D"), "D_iid": iid.get("D"),
        "G_C": corr.get("G_C"), "G_J": corr.get("G_J"),
    }
  payload["paths"] = flat_paths
  return payload


def _strategy_metadata(strategy: Any, *, workload_type: str, momentum: float,
                       learning_rate: float, weight_decay: float, artifact: Path | None = None,
                       action: str | None = None, max_optimizer_steps: int | None = None,
                       reduction: str | None = None) -> dict[str, object]:
  result: dict[str, object] = {
      "workload_type": workload_type, "horizon": int(strategy.horizon),
      "bandwidth": int(strategy.bandwidth), "min_sep": int(strategy.min_sep),
      "max_participations": int(strategy.max_participations),
      "momentum": float(momentum), "learning_rate": float(learning_rate),
      "weight_decay": float(weight_decay),
      "sensitivity_squared": float(strategy.sensitivity_squared),
      "objective": float(strategy.objective),
      "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
      "strategy_coef_C": np.asarray(strategy.strategy_coef).tolist(),
      "workload_representation": "matrix" if strategy.workload_matrix is not None else "coef",
  }
  if artifact is not None:
    result["artifact"] = str(artifact.resolve())
    result["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
  if action is not None:
    result["action"] = action
  if max_optimizer_steps is not None:
    result["max_optimizer_steps"] = int(max_optimizer_steps)
  if reduction is not None:
    result["reduction"] = reduction
  return result


def _load_or_fit_diagnostic_strategy(config: Any, contract: Any, output: Path):
  """Fit/load a separately calibrated strategy for the linear Muon frontend."""
  artifact = output / "strategies" / "exp9_diagnostic_nesterov_decayed.npz"
  existed = artifact.is_file()
  if existed and not config.force_refit:
    try:
      strategy = load_bandinv_strategy(artifact)
      metadata = load_bandinv_strategy_metadata(artifact)
      compatible = (
          strategy.horizon == contract.horizon and strategy.bandwidth == config.bandwidth
          and strategy.min_sep == contract.min_sep
          and strategy.max_participations == contract.max_participations
          and metadata.workload_type == "nesterov-decayed-trajectory"
          and metadata.reduction == config.reduction
          and metadata.max_optimizer_steps == config.max_optimizer_steps
          and metadata.momentum is not None
          and np.isclose(metadata.momentum, config.momentum)
          and metadata.learning_rate is not None
          and np.isclose(metadata.learning_rate, config.muon_learning_rate)
          and metadata.weight_decay is not None
          and np.isclose(metadata.weight_decay, config.muon_weight_decay)
      )
      if compatible:
        return strategy, artifact, "reuse"
    except (OSError, ValueError):
      pass
  workload = fixed_lr_nesterov_decayed_trajectory_workload_coef(
      contract.horizon, config.momentum, config.muon_learning_rate, config.muon_weight_decay
  )
  strategy = fit_bandinv_strategy(
      contract.horizon, config.bandwidth, contract.min_sep,
      max_participations=contract.max_participations, workload_coef=workload,
      max_optimizer_steps=config.max_optimizer_steps, reduction=config.reduction,
  )
  save_bandinv_strategy(
      artifact, strategy, reduction=config.reduction,
      workload_type="nesterov-decayed-trajectory", momentum=config.momentum,
      learning_rate=config.muon_learning_rate, weight_decay=config.muon_weight_decay,
      max_optimizer_steps=config.max_optimizer_steps,
  )
  return strategy, artifact, "fit"


def _write_outputs(output: Path, rows: list[dict[str, object]], per_seed: dict[str, dict[str, object]], *, metadata: dict[str, object]):
  write_window_rows(output / "window_diagnostics_all.csv", rows)
  write_window_summary(output / "window_summary.csv", aggregate_window_rows(rows))
  plot_cancellation_paths(rows, output / "cancellation_gain_G_C.png", gain="G_C")
  plot_cancellation_paths(rows, output / "cancellation_gain_G_J.png", gain="G_J")
  aggregate = cross_seed_aggregate(per_seed)
  with (output / "cross_seed_summary.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=["stage", "scope", "metric", "mean", "std"],
        lineterminator="\n",
    )
    writer.writeheader()
    for stage in ("early", "late", "full"):
      stage_value = aggregate.get(stage, {})
      for path, values in stage_value.get("paths", {}).items():
        for field, value in values.items():
          if not field.endswith("_mean"):
            continue
          metric = field.removesuffix("_mean")
          writer.writerow({"stage": stage, "scope": path,
                           "metric": metric, "mean": value,
                           "std": values.get(metric + "_std", 0.0)})
      for group in ("degradation", "bias_flat"):
        for field, value in stage_value.get(group, {}).items():
          if not field.endswith("_mean"):
            continue
          metric = field.removesuffix("_mean")
          writer.writerow({"stage": stage, "scope": group,
                           "metric": metric, "mean": value,
                           "std": stage_value[group].get(metric + "_std", 0.0)})
      for branch, stage_metrics in stage_value.get("stage_metrics", {}).items():
        for stage_name, values in stage_metrics.items():
          for metric_name, aggregate_value in values.items():
            writer.writerow({
                "stage": stage, "scope": f"stage_{branch}_{stage_name}",
                "metric": metric_name, "mean": aggregate_value.get("mean"),
                "std": aggregate_value.get("std"),
            })
  plot_path_gain_summary(aggregate, output / "p0_p3_cancellation.png")
  plot_decomposition(aggregate, output / "path_gap_decomposition.png")
  plot_bias_diagnostics(aggregate, output / "bias_diagnostics.png")
  plot_stage_cancellation(aggregate, output / "q_stage_cancellation.png")
  plot_stage_diagnostics(aggregate, output / "q_stage_odd_response.png")
  summary = {
      **metadata, "seeds": [int(seed) for seed in per_seed],
      "per_seed": per_seed, "cross_seed_aggregate": aggregate,
      "num_window_rows": len(rows), "window_size": 16,
      "primary_paths": {
          "P0": "s_W * R_t,W, where s_W=consistent_rms*sqrt(max(shape_W))",
          "P1": "J_F(X_t)[R_t] from a real JVP",
          "P2": "(F(X_t+R_t)-F(X_t-R_t))/2",
          "P3": "Y_t - Bhat_t, with Y_t=F(X_t+R_t) and Bhat=(BA+BB)/2",
      },
      "stage_aggregation": "J,D,C,G and bias endpoints are recomputed from raw steps for each exact stage; window ratios are descriptive only",
      "bias_label": "output_bias and raw_private_clean_gap are diagnostics, never cancellation metrics",
      "P3_reliability_rule": "probe_error_to_P3_D <= 0.1 AND probe_error_to_P3_endpoint <= 0.1; otherwise Delta_even interpretation is unreliable due to bias-probe Monte Carlo error",
  }
  (output / "summary.json").write_text(
      json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
  )
  return summary


def _run_one(*, seed: int, params: Any, training_key: jax.Array, diagnostic_key: jax.Array,
             bias_key: jax.Array, training_strategy: Any, diagnostic_strategy: Any,
             training_calibration: Any, diagnostic_calibration: Any,
             participation: ParticipationSpec, batches: Any, horizon: int,
             config: Any, output: Path, loss_fn: Any, bias_probes: int):
  train_step, optimizer = make_exp9_train_step(
      loss_fn, training_strategy, training_calibration, participation,
      diagnostic_strategy=diagnostic_strategy, diagnostic_calibration=diagnostic_calibration,
      muon_learning_rate=config.muon_learning_rate, muon_weight_decay=config.muon_weight_decay,
      momentum=config.momentum, ns_steps=config.ns_steps, consistent_rms=config.consistent_rms,
      adamw_learning_rate=config.adamw_learning_rate, adamw_beta1=config.adamw_beta1,
      adamw_beta2=config.adamw_beta2, adamw_eps=config.adamw_eps,
      adamw_weight_decay=config.adamw_weight_decay, microbatch_size=config.microbatch_size,
      probes=bias_probes, secondary_use_bf16_ns=config.use_bf16_ns,
  )
  state = init_exp9_train_state(
      params, training_strategy, training_key, optimizer, diagnostic_key,
      bias_probe_rng_key=bias_key, diagnostic_strategy=diagnostic_strategy,
  )
  collector = Exp9WindowCollector(
      params, seed=seed, learning_rate=config.muon_learning_rate,
      weight_decay=config.muon_weight_decay, horizon=horizon,
  )
  run_training(
      initial_state=state, train_step=train_step, logical_batches=batches,
      horizon=horizon, experiment_config={"experiment": "exp9", "seed": seed},
      artifact_identifiers={"experiment": "exp9", "diagnostic": "paired-nonlinear-muon"},
      eval_every=horizon, after_step=collector.after_step,
  )
  rows = collector.finalize()
  write_window_rows(output / f"window_diagnostics_seed{seed}.csv", rows)
  return rows, {name: _stage_payload(value) for name, value in collector.stage_summaries().items()}


def run_real(config_path: str | Path, output: Path, seeds: list[int], bias_probes: int = 8) -> None:
  config = load_cifar10_bandinv_dpmuon_config(resolve_repo_path(config_path))
  train_images, train_labels = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  contract = derive_contract(config, num_examples=len(train_images))
  participation = ParticipationSpec(contract.horizon, contract.min_sep, contract.max_participations)
  training_snapshot, training_action = get_or_fit_strategy_snapshot(config, participation)
  training_strategy = training_snapshot.strategy
  training_calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(training_strategy.sensitivity_squared),
  )
  output.mkdir(parents=True, exist_ok=True)
  diagnostic_strategy, diagnostic_artifact, diagnostic_action = _load_or_fit_diagnostic_strategy(
      config, contract, output
  )
  diagnostic_calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(diagnostic_strategy.sensitivity_squared),
  )
  model = ViTTiny()
  all_rows: list[dict[str, object]] = []
  per_seed: dict[str, dict[str, object]] = {}
  for seed in seeds:
    schedule = build_fixed_cycle_logical_schedule(
        num_examples=contract.num_examples, batch_size=contract.batch_size,
        horizon=contract.horizon, min_sep=contract.min_sep,
        max_participations=contract.max_participations, seed=seed,
    )
    base_key = jax.random.key(seed)
    parameter_key, training_key = jax.random.split(base_key)
    diagnostic_key = jax.random.fold_in(base_key, 9_000_009)
    bias_key = jax.random.fold_in(base_key, 9_000_010)
    pretrained = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key)
    rows, stages = _run_one(
        seed=seed, params=pretrained.params, training_key=training_key,
        diagnostic_key=diagnostic_key, bias_key=bias_key,
        training_strategy=training_strategy, diagnostic_strategy=diagnostic_strategy,
        training_calibration=training_calibration, diagnostic_calibration=diagnostic_calibration,
        participation=participation, batches=iter_logical_batches(train_images, train_labels, schedule),
        horizon=contract.horizon, config=config, output=output,
        loss_fn=lambda params, batch: cross_entropy_loss(params, batch, model),
        bias_probes=bias_probes,
    )
    all_rows.extend(rows)
    per_seed[str(seed)] = stages
  variance = pre_q_marginal_variances(
      diagnostic_strategy, diagnostic_calibration.iid_noise_std, config.momentum
  )
  metadata = {
      "experiment": "exp9", "smoke": False, "config": asdict(config),
      "contract": asdict(contract),
      "training_privacy_calibration": asdict(training_calibration),
      "diagnostic_privacy_calibration": asdict(diagnostic_calibration),
      "training_bandinv_strategy": _strategy_metadata(
          training_strategy, workload_type="nesterov-decayed-trajectory",
          momentum=config.momentum, learning_rate=config.muon_learning_rate,
          weight_decay=config.muon_weight_decay, artifact=training_snapshot.path,
          action=training_action, max_optimizer_steps=config.max_optimizer_steps,
          reduction=config.reduction,
      ),
      "diagnostic_bandinv_strategy": _strategy_metadata(
          diagnostic_strategy, workload_type="nesterov-decayed-trajectory",
          momentum=config.momentum, learning_rate=config.muon_learning_rate,
          weight_decay=config.muon_weight_decay, artifact=diagnostic_artifact,
          action=diagnostic_action, max_optimizer_steps=config.max_optimizer_steps,
          reduction=config.reduction,
      ),
      "diagnostic_control": "matched-IID raw gradient-noise marginal control; not a formal same-DP baseline",
      "trajectory": "one real correlated DP-Muon trajectory per seed; diagnostic and bias-probe branches never update params",
      "primary_q": "float32 Muon Q with use_bf16_ns=False; production BF16 is secondary only",
      "bias_probes": int(bias_probes), "raw_marginal_variances": np.asarray(variance["raw_corr"]).tolist(),
      "pre_q_marginal_variances_corr": np.asarray(variance["pre_q_corr"]).tolist(),
      "pre_q_marginal_variances_iid": np.asarray(variance["pre_q_iid"]).tolist(),
  }
  _write_outputs(output, all_rows, per_seed, metadata=metadata)


def run_smoke(output: Path, seeds: list[int], bias_probes: int = 8) -> None:
  """Small deterministic Muon-block smoke run; never touches CIFAR assets."""
  horizon = 20
  class SmokeConfig:
    muon_learning_rate = .02
    muon_weight_decay = .01
    momentum = .9
    ns_steps = 3
    consistent_rms = .2
    adamw_learning_rate = .01
    adamw_beta1 = .9
    adamw_beta2 = .9
    adamw_eps = 1e-6
    adamw_weight_decay = .01
    microbatch_size = None
    use_bf16_ns = True
  config = SmokeConfig()
  from dp_muon.bandinvmf import fit_bandinv_strategy
  from dp_muon.optim import decayed_prefix_sum_workload_coef
  training_strategy = fit_bandinv_strategy(
      horizon, 2, 1, max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(horizon, config.muon_learning_rate, config.muon_weight_decay),
      max_optimizer_steps=3,
  )
  diagnostic_strategy = fit_bandinv_strategy(
      horizon, 2, 1, max_participations=1,
      workload_coef=fixed_lr_nesterov_decayed_trajectory_workload_coef(
          horizon, config.momentum, config.muon_learning_rate, config.muon_weight_decay
      ), max_optimizer_steps=3,
  )
  training_calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(training_strategy.sensitivity_squared),
  )
  diagnostic_calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(diagnostic_strategy.sensitivity_squared),
  )
  participation = ParticipationSpec(horizon, 1, 1)
  def loss(params, batch):
    matrix = params["blocks"][0]["attention"]["query"]["kernel"]
    return .5 * (jnp.sum(matrix * batch["x"][0]) - batch["y"][0]) ** 2
  batches = [
      {"x": np.asarray([[1., step / horizon], [1., -step / horizon]], np.float32),
       "y": np.asarray([.5, -.25], np.float32)} for step in range(horizon)
  ]
  output.mkdir(parents=True, exist_ok=True)
  all_rows: list[dict[str, object]] = []
  per_seed: dict[str, dict[str, object]] = {}
  for seed in seeds:
    matrix = jax.random.normal(jax.random.key(seed), (2, 2), dtype=jnp.float32) * .1
    params = {"blocks": ({"attention": {"query": {"kernel": matrix}}},)}
    training_key, diagnostic_key, bias_key = jax.random.split(jax.random.key(seed + 10_000), 3)
    rows, stages = _run_one(
        seed=seed, params=params, training_key=training_key, diagnostic_key=diagnostic_key,
        bias_key=bias_key, training_strategy=training_strategy,
        diagnostic_strategy=diagnostic_strategy, training_calibration=training_calibration,
        diagnostic_calibration=diagnostic_calibration, participation=participation,
        batches=list(batches), horizon=horizon, config=config, output=output,
        loss_fn=loss, bias_probes=bias_probes,
    )
    all_rows.extend(rows)
    per_seed[str(seed)] = stages
  _write_outputs(output, all_rows, per_seed, metadata={
      "experiment": "exp9", "smoke": True,
      "bias_probes": int(bias_probes),
      "config": {"horizon": horizon, "bias_probes": bias_probes},
      "training_privacy_calibration": asdict(training_calibration),
      "diagnostic_privacy_calibration": asdict(diagnostic_calibration),
      "training_bandinv_strategy": _strategy_metadata(
          training_strategy, workload_type="smoke-prefix-control", momentum=config.momentum,
          learning_rate=config.muon_learning_rate, weight_decay=config.muon_weight_decay,
          max_optimizer_steps=3, reduction="mean",
      ),
      "diagnostic_bandinv_strategy": _strategy_metadata(
          diagnostic_strategy, workload_type="nesterov-decayed-trajectory", momentum=config.momentum,
          learning_rate=config.muon_learning_rate, weight_decay=config.muon_weight_decay,
          max_optimizer_steps=3, reduction="mean",
      ),
      "diagnostic_control": "matched-IID raw gradient-noise marginal control; not a formal same-DP baseline",
      "primary_q": "float32 Muon Q with use_bf16_ns=False",
  })


def main(argv=None):
  args = parse_args(argv)
  output = resolve_repo_path(args.output_dir)
  if args.smoke:
    run_smoke(output, args.seeds, args.bias_probes)
  else:
    run_real(args.config, output, args.seeds, args.bias_probes)


if __name__ == "__main__":
  main()
