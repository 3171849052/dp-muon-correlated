#!/usr/bin/env python3
"""Run Experiment 8's one-trajectory, paired shadow mechanism diagnostic."""

from __future__ import annotations

import argparse
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

from dp_muon.data import iter_logical_batches, load_cifar10
from dp_muon.models import ViTTiny
from dp_muon.optim import (
    adam_first_moment_workload_matrix,
    decayed_prefix_sum_workload_coef,
)
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
from exp2.strategies import ADAM_M_AWARE, StrategySpec, load_or_fit_strategy
from exp8.core import (
    PATHS,
    bandinv_marginal_variances,
    init_exp8_train_state,
    make_exp8_train_step,
)
from exp8.diagnostics import (
    aggregate_window_rows,
    attach_path_degradation,
    cross_seed_aggregate,
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
  # Kept as a local compatibility wrapper for callers that used the previous
  # private helper; the implementation now lives in diagnostics.py for tests
  # and plotting to share exactly the same aggregate.
  return cross_seed_aggregate(per_seed)


def _strategy_metadata(
    strategy: Any,
    *,
    workload_type: str,
    beta1: float,
    learning_rate: float,
    weight_decay: float,
    artifact: Path | None = None,
    action: str | None = None,
    artifact_sha256: str | None = None,
    reduction: str | None = None,
    max_optimizer_steps: int | None = None,
) -> dict[str, object]:
  """Serialize one strategy without conflating it with the other strategy."""
  metadata: dict[str, object] = {
      "workload_type": workload_type,
      "horizon": int(strategy.horizon),
      "bandwidth": int(strategy.bandwidth),
      "min_sep": int(strategy.min_sep),
      "max_participations": (
          None if strategy.max_participations is None
          else int(strategy.max_participations)
      ),
      "beta1": float(beta1),
      "learning_rate": float(learning_rate),
      "weight_decay": float(weight_decay),
      "sensitivity_squared": float(strategy.sensitivity_squared),
      "objective": float(strategy.objective),
      "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
      "strategy_coef_C": np.asarray(strategy.strategy_coef).tolist(),
      "workload_representation": (
          "matrix" if strategy.workload_matrix is not None else "coef"
      ),
  }
  if reduction is not None:
    metadata["reduction"] = reduction
  if max_optimizer_steps is not None:
    metadata["max_optimizer_steps"] = int(max_optimizer_steps)
  if strategy.workload_matrix is not None:
    matrix = np.asarray(strategy.workload_matrix)
    metadata["workload_matrix_shape"] = list(matrix.shape)
    metadata["workload_matrix_sha256"] = hashlib.sha256(
        np.ascontiguousarray(matrix).tobytes()
    ).hexdigest()
  if artifact is not None:
    metadata["artifact"] = str(artifact.resolve())
  if artifact_sha256 is not None:
    metadata["sha256"] = artifact_sha256
  if action is not None:
    metadata["action"] = action
  return metadata


def _load_or_fit_diagnostic_strategy(
    config: Any, contract: Any, output: Path
) -> tuple[Any, Path, str, str]:
  """Load/fit Exp8's momentum-aware diagnostic artifact in Exp8 output."""
  spec = StrategySpec(
      name=ADAM_M_AWARE,
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
  artifact = output / "strategies" / "exp8_diagnostic_adam_m_aware.npz"
  existed = artifact.is_file()
  diagnostic_strategy = load_or_fit_strategy(
      artifact, spec, force_refit=config.force_refit
  )
  action = "fit" if config.force_refit or not existed else "reuse"
  artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
  return diagnostic_strategy, artifact, action, artifact_sha256


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
  # Cross-seed plots use the same exact-stage aggregate, not a representative seed.
  aggregate = _cross_seed_aggregate(per_seed)
  plot_path_gain_summary(aggregate, output / "path_gain_summary.png")
  plot_decomposition(aggregate, output / "privacy_clean_decomposition.png")
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
    training_strategy: Any,
    diagnostic_strategy: Any,
    training_calibration: Any,
    diagnostic_calibration: Any,
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
      loss_fn, training_strategy, training_calibration, participation,
      diagnostic_strategy=diagnostic_strategy,
      diagnostic_calibration=diagnostic_calibration,
      learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
      weight_decay=weight_decay, microbatch_size=microbatch_size,
  )
  state = init_exp8_train_state(
      params, training_strategy, training_key, optimizer, diagnostic_key,
      diagnostic_strategy=diagnostic_strategy,
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
  training_strategy = snapshot.strategy
  training_calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(training_strategy.sensitivity_squared),
  )
  output.mkdir(parents=True, exist_ok=True)
  diagnostic_strategy, diagnostic_artifact, diagnostic_action, diagnostic_sha256 = (
      _load_or_fit_diagnostic_strategy(config, contract, output)
  )
  diagnostic_calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(diagnostic_strategy.sensitivity_squared),
  )
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
        diagnostic_key=diagnostic_key, training_strategy=training_strategy,
        diagnostic_strategy=diagnostic_strategy,
        training_calibration=training_calibration,
        diagnostic_calibration=diagnostic_calibration,
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
  phi = bandinv_marginal_variances(
      diagnostic_strategy, diagnostic_calibration.iid_noise_std
  )
  training_metadata = _strategy_metadata(
      training_strategy,
      workload_type="decayed-prefix-sum",
      beta1=config.beta1, learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      artifact=snapshot.path, action=action, artifact_sha256=snapshot.sha256,
      reduction=config.reduction, max_optimizer_steps=config.max_optimizer_steps,
  )
  diagnostic_metadata = _strategy_metadata(
      diagnostic_strategy,
      workload_type="adam-first-moment-aware",
      beta1=config.beta1, learning_rate=config.learning_rate,
      weight_decay=config.weight_decay,
      artifact=diagnostic_artifact, action=diagnostic_action,
      artifact_sha256=diagnostic_sha256,
      reduction=config.reduction, max_optimizer_steps=config.max_optimizer_steps,
  )
  metadata = {
      "experiment": "exp8",
      "smoke": False,
      "config": asdict(config),
      "contract": asdict(contract),
      "training_privacy_calibration": asdict(training_calibration),
      "diagnostic_privacy_calibration": asdict(diagnostic_calibration),
      "training_bandinv_strategy": training_metadata,
      "diagnostic_bandinv_strategy": diagnostic_metadata,
      # Compatibility aliases retain their historical meaning: they refer to
      # the real training mechanism, never to the diagnostic mechanism.
      "privacy_calibration": asdict(training_calibration),
      "bandinv_strategy": training_metadata,
      "phi_t": np.asarray(phi).tolist(),
      "phi_t_definition": "phi_t = diagnostic_calibration.iid_noise_std^2 * sum of squares of the first min(t+1, diagnostic_bandinv_strategy.bandwidth) coefficients from the diagnostic momentum-aware C^-1",
      "diagnostic_control": "matched-marginal IID mechanism diagnostic only; not a formal same-guarantee DP baseline",
      "trajectory": "one real correlated DP-AdamW baseline trajectory per seed; diagnostic shadows never update params or optimizer state",
  }
  _write_outputs(output, all_rows, per_seed, metadata=metadata)


def run_smoke(output: Path, seeds: list[int]) -> None:
  horizon = 20
  learning_rate, weight_decay, beta1, beta2, eps = .02, .01, .9, .9, 1e-6
  from dp_muon.bandinvmf import fit_bandinv_strategy

  training_strategy = fit_bandinv_strategy(
      horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(horizon, learning_rate, weight_decay),
      max_optimizer_steps=3,
  )
  diagnostic_strategy = fit_bandinv_strategy(
      horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_matrix=adam_first_moment_workload_matrix(
          horizon, beta1, learning_rate, weight_decay
      ),
      max_optimizer_steps=3,
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
        diagnostic_key=diagnostic_key, training_strategy=training_strategy,
        diagnostic_strategy=diagnostic_strategy,
        training_calibration=training_calibration,
        diagnostic_calibration=diagnostic_calibration,
        participation=participation, batches=list(batches), horizon=horizon,
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
        weight_decay=weight_decay, microbatch_size=None, output=output, loss_fn=loss,
    )
    all_rows.extend(rows)
    per_seed[str(seed)] = stages
  phi = bandinv_marginal_variances(
      diagnostic_strategy, diagnostic_calibration.iid_noise_std
  )
  _write_outputs(
      output, all_rows, per_seed,
      metadata={
          "experiment": "exp8", "smoke": True,
          "config": {"horizon": horizon, "learning_rate": learning_rate,
                     "beta1": beta1, "beta2": beta2, "eps": eps,
                     "weight_decay": weight_decay},
          "training_privacy_calibration": asdict(training_calibration),
          "diagnostic_privacy_calibration": asdict(diagnostic_calibration),
          "training_bandinv_strategy": _strategy_metadata(
              training_strategy, workload_type="decayed-prefix-sum", beta1=beta1,
              learning_rate=learning_rate, weight_decay=weight_decay,
              reduction="mean", max_optimizer_steps=3,
          ),
          "diagnostic_bandinv_strategy": _strategy_metadata(
              diagnostic_strategy, workload_type="adam-first-moment-aware", beta1=beta1,
              learning_rate=learning_rate, weight_decay=weight_decay,
              reduction="mean", max_optimizer_steps=3,
          ),
          "privacy_calibration": asdict(training_calibration),
          "bandinv_strategy": _strategy_metadata(
              training_strategy, workload_type="decayed-prefix-sum", beta1=beta1,
              learning_rate=learning_rate, weight_decay=weight_decay,
              reduction="mean", max_optimizer_steps=3,
          ),
          "phi_t": np.asarray(phi).tolist(),
          "phi_t_definition": "phi_t = diagnostic_calibration.iid_noise_std^2 * sum of squares of the causal diagnostic momentum-aware C^-1 row",
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
