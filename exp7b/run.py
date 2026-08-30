#!/usr/bin/env python3
"""Run paired correlated DP-AdamW / DP-AdamBC-gamma Experiment 7b."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Callable

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
from dp_muon.training.cifar10_bandinv_dpadamw_experiment import load_cifar10_bandinv_dpadamw_config
from dp_muon.training.cifar10_driver import (
    build_fixed_cycle_logical_schedule, cross_entropy_loss,
    evaluate_classifier_metrics, run_training,
)
from dp_muon.training.pretrained_snapshot import load_pretrained_snapshot
from exp2.common import derive_contract, resolve_repo_path
from exp2.strategies import DECAYED_PREFIX
from exp3.run import _strategy as exp3_strategy
from exp7.core import bandinv_marginal_variances, init_exp7_train_state
from exp7.plotting import plot_paired_gaps
from exp7b.core import (
    DEFAULT_GAMMA_PRIME_RATIO, gamma_prime_from_ratio, make_exp7b_train_step,
)
from exp7b.diagnostics import (
    aggregate_window_rows, two_stage_summary, write_window_rows, write_window_summary,
)
from exp7b.online_shadow import Exp7bWindowCollector
from exp7b.plotting import plot_bc_preconditioner, plot_cancellation, plot_update_norms


WINDOW_SIZE = 16


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="config/cifar10_bandinv_dpadamw_naive.yaml")
  parser.add_argument("--output-dir", default="exp7b/results")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--gamma-prime-ratio", type=float, default=DEFAULT_GAMMA_PRIME_RATIO)
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args(argv)


def build_schedule_for_seed(contract, seed):
  return build_fixed_cycle_logical_schedule(
      num_examples=contract.num_examples, batch_size=contract.batch_size,
      horizon=contract.horizon, min_sep=contract.min_sep,
      max_participations=contract.max_participations, seed=seed,
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


def _run_trajectory(
    *, algorithm: str, seed: int, params, noise_key, strategy, calibration,
    participation, batches, horizon: int, learning_rate: float, beta1: float,
    beta2: float, eps: float, weight_decay: float, microbatch_size: int | None,
    gamma_prime: float, output: Path, experiment_config: dict[str, Any],
    artifact_identifiers: dict[str, str], loss_fn: Callable[[Any, Any], Any],
    evaluate: Callable[[Any], dict[str, float]] | None, eval_every: int = 1,
    num_train_examples: int | None = None, logical_batch_size: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
  step = make_exp7b_train_step(
      loss_fn, strategy, calibration, participation, algorithm=algorithm,
      learning_rate=learning_rate, gamma_prime=gamma_prime, beta1=beta1,
      beta2=beta2, eps=eps, weight_decay=weight_decay,
      microbatch_size=microbatch_size,
  )
  state = init_exp7_train_state(params, strategy, noise_key)
  collector = Exp7bWindowCollector(
      params, seed=seed, algorithm=algorithm, beta1=beta1, beta2=beta2,
      learning_rate=learning_rate, weight_decay=weight_decay, eps=eps,
      gamma_prime=gamma_prime, window_size=WINDOW_SIZE,
  )
  final, history = run_training(
      initial_state=state, train_step=step, logical_batches=batches, horizon=horizon,
      experiment_config=experiment_config, artifact_identifiers=artifact_identifiers,
      eval_every=eval_every, evaluate=evaluate,
      num_train_examples=num_train_examples, logical_batch_size=logical_batch_size,
      privacy_accountant=(lambda current: epsilon_spent_for_bandinv_prefix(
          prefix_steps=current, noising_coef=strategy.noising_coef,
          horizon=strategy.horizon, min_sep=strategy.min_sep,
          max_participations=strategy.max_participations, calibration=calibration,
          full_sensitivity_squared=float(strategy.sensitivity_squared),
      )) if num_train_examples is not None else None,
      after_step=collector.after_step,
  )
  rows = collector.finalize()
  write_window_rows(output / f"window_diagnostics_{algorithm}_seed{seed}.csv", rows)
  last = history[-1] if history else {}
  return {
      "seed": seed, "algorithm": algorithm, "final_step": int(final.step),
      "final_test_loss": last.get("test_loss"),
      "final_test_accuracy": last.get("test_accuracy"),
      "num_windows": len(rows),
  }, rows


class _Evaluator:
  def __init__(self, loss_fn, evaluate_fn):
    self.loss_fn, self.evaluate_fn = loss_fn, evaluate_fn

  def __call__(self, state):
    return self.evaluate_fn(state)


def _write_outputs(
    output: Path, rows: list[dict[str, object]], runs: list[dict[str, Any]],
    *, horizon: int, metadata: dict[str, Any], gamma_prime_ratio: float,
    phi_inf: float, gamma_prime: float,
) -> dict[str, Any]:
  aggregated = aggregate_window_rows(rows)
  write_window_summary(output / "window_summary.csv", aggregated)
  plot_cancellation(aggregated, output / "shadow_cancellation.png")
  plot_paired_gaps(aggregated, output / "paired_real_vs_clean_gap.png")
  plot_bc_preconditioner(aggregated, output / "bc_preconditioner_floor_activation.png")
  plot_update_norms(aggregated, output / "update_norm_diagnostics.png")
  by_algorithm = {
      algorithm: two_stage_summary(
          [row for row in rows if row["algorithm"] == algorithm], total_steps=horizon
      ) for algorithm in ("baseline", "bc")
  }
  per_seed: dict[str, Any] = {}
  for run in runs:
    seed = str(run["seed"])
    per_seed.setdefault(seed, {})[run["algorithm"]] = {
        "final_test_loss": run["final_test_loss"],
        "final_test_accuracy": run["final_test_accuracy"],
        "final_step": run["final_step"],
    }
  for seed in per_seed:
    for algorithm in ("baseline", "bc"):
      per_seed[seed][algorithm].update(two_stage_summary(
          [row for row in rows
           if str(row["seed"]) == seed and row["algorithm"] == algorithm],
          total_steps=horizon,
      ))
  summary = {
      **metadata,
      "gamma_prime_ratio": gamma_prime_ratio,
      "phi_infty": phi_inf,
      "gamma_prime": gamma_prime,
      "implied_p_max": 1.0 / np.sqrt(gamma_prime),
      "gamma_prime_privacy_note": (
          "deterministic post-processing of the existing DP output; no additional privacy cost"
      ),
      "window_size": WINDOW_SIZE,
      "per_seed": per_seed,
      "baseline_shadow_decomposition": by_algorithm["baseline"],
      "paired_training_cancellation": by_algorithm,
      "baseline_real_vs_clean_gap": {
          stage: values.get("gap") for stage, values in by_algorithm["baseline"].items()
      },
      "bc_real_vs_clean_gap": {
          stage: values.get("gap") for stage, values in by_algorithm["bc"].items()
      },
      "num_window_rows": len(rows),
  }
  (output / "summary.json").write_text(
      json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
  )
  return summary


def run_real(
    config_path: str | Path, output: Path, seeds: list[int], *, gamma_prime_ratio: float
) -> None:
  config = load_cifar10_bandinv_dpadamw_config(resolve_repo_path(config_path))
  train_images, train_labels = load_cifar10(resolve_repo_path(config.data_dir), train=True)
  test_images, test_labels = load_cifar10(resolve_repo_path(config.data_dir), train=False)
  contract = derive_contract(config, num_examples=len(train_images))
  strategy = exp3_strategy(config, contract, DECAYED_PREFIX)
  calibration = calibrate_nonamplified_bandinv(
      epsilon=config.epsilon, delta=config.delta, clip_norm=config.clip_norm,
      normalize_by=float(config.batch_size), adjacency=config.adjacency,
      sensitivity_squared=float(strategy.sensitivity_squared),
  )
  phi_inf_array, gamma_array = gamma_prime_from_ratio(
      strategy, calibration.iid_noise_std, gamma_prime_ratio
  )
  phi_inf, gamma_prime = float(phi_inf_array), float(gamma_array)
  participation = ParticipationSpec(
      contract.horizon, contract.min_sep, contract.max_participations
  )
  output.mkdir(parents=True, exist_ok=True)
  rows, runs = [], []
  for seed in seeds:
    schedule = build_schedule_for_seed(contract, seed)
    parameter_key, noise_key = jax.random.split(jax.random.key(seed))
    snapshot = load_pretrained_snapshot(resolve_repo_path(config.pretrained), key=parameter_key)
    model = ViTTiny()
    evaluator = _Evaluator(
        lambda p, b: cross_entropy_loss(p, b, model),
        lambda state: {
            "train_loss": evaluate_classifier_metrics(
                state.params, model, train_images, train_labels, batch_size=config.batch_size
            )["test_loss"],
            **evaluate_classifier_metrics(
                state.params, model, test_images, test_labels, batch_size=config.batch_size
            ),
        },
    )
    for algorithm in ("baseline", "bc"):
      result, trajectory_rows = _run_trajectory(
          algorithm=algorithm, seed=seed, params=snapshot.params, noise_key=noise_key,
          strategy=strategy, calibration=calibration, participation=participation,
          batches=iter_logical_batches(train_images, train_labels, schedule),
          horizon=contract.horizon, learning_rate=config.learning_rate,
          beta1=config.beta1, beta2=config.beta2, eps=config.eps,
          weight_decay=config.weight_decay, microbatch_size=config.microbatch_size,
          gamma_prime=gamma_prime, output=output,
          experiment_config={**asdict(config), "exp7b_algorithm": algorithm,
                             "gamma_prime_ratio": gamma_prime_ratio},
          artifact_identifiers={"experiment": "exp7b", "algorithm": algorithm,
                                "strategy": DECAYED_PREFIX,
                                "pretrained_sha256": snapshot.sha256},
          loss_fn=evaluator.loss_fn, evaluate=evaluator, eval_every=config.eval_every,
          num_train_examples=contract.num_examples, logical_batch_size=contract.batch_size,
      )
      runs.append(result)
      rows.extend(trajectory_rows)
  _write_outputs(
      output, rows, runs, horizon=contract.horizon,
      gamma_prime_ratio=gamma_prime_ratio, phi_inf=phi_inf, gamma_prime=gamma_prime,
      metadata={"smoke": False, "contract": asdict(contract), "config": asdict(config),
                "calibration": asdict(calibration),
                "bandinv_noise": {
                    "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
                    "phi_t": np.asarray(bandinv_marginal_variances(
                        strategy, calibration.iid_noise_std
                    )).tolist(),
                    "definition": "phi_t = iid_noise_std^2 * ||row_t(C^-1)||_2^2",
                }},
  )


def run_smoke(
    output: Path, seeds: list[int], *,
    gamma_prime_ratio: float = DEFAULT_GAMMA_PRIME_RATIO,
) -> None:
  horizon = 20
  learning_rate, weight_decay, beta1, beta2, eps = .02, .01, .9, .9, 1e-6
  strategy = fit_bandinv_strategy(
      horizon, bandwidth=2, min_sep=1, max_participations=1,
      workload_coef=decayed_prefix_sum_workload_coef(
          horizon, learning_rate, weight_decay
      ),
      max_optimizer_steps=3,
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=3.0, delta=1e-5, clip_norm=1.0, normalize_by=2.0,
      adjacency="add_remove", sensitivity_squared=float(strategy.sensitivity_squared),
  )
  phi_inf_array, gamma_array = gamma_prime_from_ratio(
      strategy, calibration.iid_noise_std, gamma_prime_ratio
  )
  phi_inf, gamma_prime = float(phi_inf_array), float(gamma_array)
  participation = ParticipationSpec(horizon, 1, 1)

  def loss(params, batch):
    prediction = jnp.dot(params["w"], batch["x"][0])
    return .5 * (prediction - batch["y"][0]) ** 2

  batches = [
      {"x": np.asarray([[1., step / horizon], [1., -step / horizon]], np.float32),
       "y": np.asarray([.5, -.25], np.float32)} for step in range(horizon)
  ]
  output.mkdir(parents=True, exist_ok=True)
  rows, runs = [], []
  for seed in seeds:
    params = {"w": jax.random.normal(jax.random.key(seed), (2,)) * .01}
    noise_key = jax.random.key(seed + 10_000)
    evaluator = _Evaluator(
        loss,
        lambda state: {
            "test_loss": float(jnp.sum(state.params["w"] ** 2)),
            "test_accuracy": float(jnp.mean(jnp.isfinite(state.params["w"]))),
        },
    )
    for algorithm in ("baseline", "bc"):
      result, trajectory_rows = _run_trajectory(
          algorithm=algorithm, seed=seed, params=params, noise_key=noise_key,
          strategy=strategy, calibration=calibration, participation=participation,
          batches=list(batches), horizon=horizon, learning_rate=learning_rate,
          beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
          microbatch_size=None, gamma_prime=gamma_prime, output=output,
          experiment_config={"experiment": "exp7b-smoke", "algorithm": algorithm,
                             "gamma_prime_ratio": gamma_prime_ratio},
          artifact_identifiers={"experiment": "exp7b-smoke", "algorithm": algorithm},
          loss_fn=evaluator.loss_fn, evaluate=evaluator,
      )
      runs.append(result)
      rows.extend(trajectory_rows)
  _write_outputs(
      output, rows, runs, horizon=horizon,
      gamma_prime_ratio=gamma_prime_ratio, phi_inf=phi_inf, gamma_prime=gamma_prime,
      metadata={"smoke": True, "calibration": asdict(calibration),
                "bandinv_noise": {
                    "noising_coef_C_inverse": np.asarray(strategy.noising_coef).tolist(),
                    "phi_t": np.asarray(bandinv_marginal_variances(
                        strategy, calibration.iid_noise_std
                    )).tolist(),
                    "definition": "phi_t = iid_noise_std^2 * ||row_t(C^-1)||_2^2",
                }},
  )


def main(argv=None):
  args = parse_args(argv)
  output = resolve_repo_path(args.output_dir)
  if args.smoke:
    run_smoke(output, args.seeds, gamma_prime_ratio=args.gamma_prime_ratio)
  else:
    run_real(
        args.config, output, args.seeds, gamma_prime_ratio=args.gamma_prime_ratio
    )


if __name__ == "__main__":
  main()
