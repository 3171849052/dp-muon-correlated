#!/usr/bin/env python3
"""Paired full-AdamW replay for the two Experiment 2 BandInvMF strategies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from dp_muon.bandinvmf import BandInvMFStrategy

from exp2.strategies import ADAM_M_AWARE, DECAYED_PREFIX, StrategySpec, load_or_fit_strategy


def _resolve(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> Any:
  value = np.asarray(archive[name])
  if value.shape != ():
    raise ValueError(f"trajectory field {name!r} must be a scalar")
  return value.item()


def load_trajectory(path: str | Path) -> dict[str, Any]:
  """Loads the public archive emitted by :mod:`exp2.collect_trajectory`."""
  required = {"g", "parameter_name", "start_step", "learning_rate", "beta1", "beta2", "eps", "weight_decay"}
  with np.load(path, allow_pickle=False) as archive:
    missing = required.difference(archive.files)
    if missing:
      raise ValueError(f"trajectory is missing fields: {sorted(missing)}")
    g = np.asarray(archive["g"], dtype=np.float32)
    if g.ndim != 3 or g.shape[0] < 1 or min(g.shape[1:]) < 1 or not np.all(np.isfinite(g)):
      raise ValueError("trajectory g must be finite and have shape (T,m,n)")
    result = {"g": g, "parameter_name": str(_scalar(archive, "parameter_name"))}
    for name in ("start_step", "learning_rate", "beta1", "beta2", "eps", "weight_decay"):
      result[name] = int(_scalar(archive, name)) if name == "start_step" else float(_scalar(archive, name))
  if result["start_step"] != 0:
    raise ValueError("Exp2 currently requires trajectory start_step == 0")
  if result["learning_rate"] <= 0 or result["eps"] <= 0 or result["weight_decay"] < 0:
    raise ValueError("trajectory AdamW scalar metadata is invalid")
  if not 0 <= result["beta1"] < 1 or not 0 <= result["beta2"] < 1:
    raise ValueError("trajectory beta1 and beta2 must be in [0, 1)")
  return result


def _lower_toeplitz(coef: np.ndarray, horizon: int) -> np.ndarray:
  coef = np.asarray(coef, dtype=np.float64)
  if coef.ndim != 1 or not 1 <= len(coef) <= horizon:
    raise ValueError("noising_coef must be one-dimensional with length in [1, horizon]")
  padded = np.pad(coef, (0, horizon - len(coef)))
  offsets = np.arange(horizon)[:, None] - np.arange(horizon)[None, :]
  return np.where(offsets >= 0, padded[np.maximum(offsets, 0)], 0.0)


def adamw_q(gradients: np.ndarray, *, beta1: float, beta2: float, eps: float) -> np.ndarray:
  """Runs Adam's bias-corrected nonlinear direction recurrence on a batch."""
  gradients = np.asarray(gradients, dtype=np.float64)
  if gradients.ndim != 4:
    raise ValueError("gradients must have shape (samples,T,m,n)")
  samples, horizon, rows, columns = gradients.shape
  del rows, columns
  moment = np.zeros_like(gradients[:, 0])
  variance = np.zeros_like(moment)
  result = np.empty_like(gradients)
  for step in range(horizon):
    gradient = gradients[:, step]
    moment = beta1 * moment + (1.0 - beta1) * gradient
    variance = beta2 * variance + (1.0 - beta2) * gradient * gradient
    corrected_m = moment / (1.0 - beta1 ** (step + 1))
    corrected_v = variance / (1.0 - beta2 ** (step + 1))
    result[:, step] = corrected_m / (np.sqrt(corrected_v) + eps)
  return result


def adamw_perturbations(
    clean_gradients: np.ndarray, noise: np.ndarray, *, learning_rate: float,
    beta1: float, beta2: float, eps: float, weight_decay: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns AdamW ``Delta q`` and parameter deltas for a noisy transcript."""
  clean_gradients = np.asarray(clean_gradients, dtype=np.float64)
  noise = np.asarray(noise, dtype=np.float64)
  if noise.ndim != 4 or noise.shape[1:] != clean_gradients.shape:
    raise ValueError("noise must have shape (samples,T,m,n) matching clean gradients")
  clean_q = adamw_q(clean_gradients[None, ...], beta1=beta1, beta2=beta2, eps=eps)
  noisy_q = adamw_q(clean_gradients[None, ...] + noise, beta1=beta1, beta2=beta2, eps=eps)
  delta_q = noisy_q - clean_q
  rho = 1.0 - learning_rate * weight_decay
  delta_theta = np.empty_like(delta_q)
  previous = np.zeros_like(delta_q[:, 0])
  for step in range(delta_q.shape[1]):
    previous = rho * previous - learning_rate * delta_q[:, step]
    delta_theta[:, step] = previous
  return delta_q, delta_theta


def cancellation_statistics(delta_q: np.ndarray, delta_theta: np.ndarray, learning_rate: float) -> dict[str, Any]:
  """Computes ``J_k,D_k,R_k`` with expectation before all divisions."""
  x = -learning_rate * np.asarray(delta_q, dtype=np.float64)
  delta_theta = np.asarray(delta_theta, dtype=np.float64)
  j = np.mean(np.sum(delta_theta * delta_theta, axis=(2, 3)), axis=0)
  d = np.cumsum(np.mean(np.sum(x * x, axis=(2, 3)), axis=0))
  r = np.divide(j, d, out=np.zeros_like(j), where=d != 0)
  total_d = float(np.sum(d))
  return {"J_k": j, "D_k": d, "R_k": r, "J": float(np.sum(j)), "D": total_d,
          "R": float(np.sum(j) / total_d) if total_d else 0.0}


def linear_m_reference(noise: np.ndarray, trajectory: dict[str, Any]) -> dict[str, Any]:
  """Linear reference for ``A_m`` using the Adam first-moment recurrence only."""
  beta1, eta, weight_decay = trajectory["beta1"], trajectory["learning_rate"], trajectory["weight_decay"]
  moment = np.zeros_like(noise[:, 0], dtype=np.float64)
  delta_q = np.empty_like(noise, dtype=np.float64)
  for step in range(noise.shape[1]):
    moment = beta1 * moment + (1.0 - beta1) * noise[:, step]
    delta_q[:, step] = moment / (1.0 - beta1 ** (step + 1))
  rho = 1.0 - eta * weight_decay
  delta_theta = np.empty_like(delta_q)
  previous = np.zeros_like(delta_q[:, 0])
  for step in range(noise.shape[1]):
    previous = rho * previous - eta * delta_q[:, step]
    delta_theta[:, step] = previous
  return cancellation_statistics(delta_q, delta_theta, eta)


def _relative_noise_ratios(clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(clean, axis=(1, 2))
  if np.any(norm == 0):
    raise ValueError("cannot calibrate relative noise against a zero-norm clean gradient")
  return np.linalg.norm(noise, axis=(2, 3)) / norm[None, :]


def _strategy_noise(latent: np.ndarray, strategy: BandInvMFStrategy) -> np.ndarray:
  operator = _lower_toeplitz(np.asarray(strategy.noising_coef), latent.shape[1])
  return np.einsum("ts,bsij->btij", operator, latent, optimize=True).astype(np.float64)


def _global_scalars(
    trajectory: dict[str, Any], strategies: dict[str, BandInvMFStrategy], *, samples: int,
    seed: int, targets: list[float], sample_batch_size: int,
) -> tuple[dict[str, dict[float, float]], dict[str, dict[float, float]]]:
  """Calibrates one deterministic scalar per (strategy, target), using paired latents."""
  rng = np.random.default_rng(seed)
  ratios = {name: [] for name in strategies}
  for offset in range(0, samples, sample_batch_size):
    batch = min(sample_batch_size, samples - offset)
    latent = rng.standard_normal((batch, *trajectory["g"].shape), dtype=np.float32)
    for name, strategy in strategies.items():
      ratios[name].append(_relative_noise_ratios(trajectory["g"], _strategy_noise(latent, strategy)).reshape(-1))
  reference = {name: float(np.median(np.concatenate(parts))) for name, parts in ratios.items()}
  if any(value <= 0 or not np.isfinite(value) for value in reference.values()):
    raise ValueError("raw noise has an invalid overall median relative norm")
  return (
      {name: {target: float(target / reference[name]) for target in targets} for name in strategies},
      {name: {target: float(target) for target in targets} for name in strategies},
  )


def run_replay(
    trajectory: dict[str, Any], *, strategies: dict[str, BandInvMFStrategy], samples: int,
    seed: int, target_relative_noise: list[float], sample_batch_size: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  """Runs paired nonlinear Monte Carlo replay for both Exp2 strategies."""
  if set(strategies) != {DECAYED_PREFIX, ADAM_M_AWARE}:
    raise ValueError("Exp2 requires exactly decayed-prefix and adam-m-aware strategies")
  if samples < 1 or sample_batch_size < 1 or not target_relative_noise or any(x <= 0 for x in target_relative_noise):
    raise ValueError("samples, sample_batch_size, and target_relative_noise must be positive")
  horizon = trajectory["g"].shape[0]
  for name, strategy in strategies.items():
    if strategy.horizon != horizon:
      raise ValueError(f"{name} strategy horizon must equal frozen trajectory length")
    if name == DECAYED_PREFIX and strategy.workload_coef is None:
      raise ValueError("decayed-prefix strategy must use a Toeplitz workload coefficient")
    if name == ADAM_M_AWARE and strategy.workload_matrix is None:
      raise ValueError("adam-m-aware strategy must use a general workload matrix")
  scalars, target_values = _global_scalars(
      trajectory, strategies, samples=samples, seed=seed, targets=target_relative_noise,
      sample_batch_size=sample_batch_size,
  )
  totals = {name: {target: {"j": np.zeros(horizon), "d": np.zeros(horizon)}
                    for target in target_relative_noise} for name in strategies}
  linear_totals = {target: {"j": np.zeros(horizon), "d": np.zeros(horizon)} for target in target_relative_noise}
  actual_ratios = {name: {target: [] for target in target_relative_noise} for name in strategies}
  rng = np.random.default_rng(seed)
  for offset in range(0, samples, sample_batch_size):
    batch = min(sample_batch_size, samples - offset)
    # This single draw is deliberately shared across both covariance strategies.
    latent = rng.standard_normal((batch, *trajectory["g"].shape), dtype=np.float32)
    for name, strategy in strategies.items():
      raw_noise = _strategy_noise(latent, strategy)
      for target in target_relative_noise:
        noise = scalars[name][target] * raw_noise
        actual_ratios[name][target].append(_relative_noise_ratios(trajectory["g"], noise).reshape(-1))
        delta_q, delta_theta = adamw_perturbations(noise=noise, clean_gradients=trajectory["g"], **{
            key: trajectory[key] for key in ("learning_rate", "beta1", "beta2", "eps", "weight_decay")
        })
        stats = cancellation_statistics(delta_q, delta_theta, trajectory["learning_rate"])
        totals[name][target]["j"] += stats["J_k"]
        totals[name][target]["d"] += np.diff(np.r_[0.0, stats["D_k"]])
        if name == ADAM_M_AWARE:
          reference = linear_m_reference(noise, trajectory)
          linear_totals[target]["j"] += reference["J_k"]
          linear_totals[target]["d"] += np.diff(np.r_[0.0, reference["D_k"]])
  rows: list[dict[str, Any]] = []
  prefixes: list[dict[str, Any]] = []
  summary: dict[str, Any] = {"samples": samples, "seed": seed, "paired_latent_draws": True,
      "trajectory": {key: trajectory[key] for key in ("parameter_name", "start_step", "learning_rate", "beta1", "beta2", "eps", "weight_decay")},
      "targets": {}}
  for target in target_relative_noise:
    per_strategy: dict[str, dict[str, Any]] = {}
    for name in (DECAYED_PREFIX, ADAM_M_AWARE):
      j = totals[name][target]["j"] / samples
      d = np.cumsum(totals[name][target]["d"] / samples)
      r_k = np.divide(j, d, out=np.zeros_like(j), where=d != 0)
      total_d = float(np.sum(d))
      stats = {"J": float(np.sum(j)), "D": total_d, "R": float(np.sum(j) / total_d) if total_d else 0.0}
      per_strategy[name] = stats
      for prefix, (j_k, d_k, r_value) in enumerate(zip(j, d, r_k, strict=True), start=1):
        prefixes.append({"target_r": target, "strategy": name, "prefix": prefix,
                         "J_k": float(j_k), "D_k": float(d_k), "R_k": float(r_value)})
    delta_r = per_strategy[ADAM_M_AWARE]["R"] - per_strategy[DECAYED_PREFIX]["R"]
    for name in (DECAYED_PREFIX, ADAM_M_AWARE):
      row = {"target_r": target, "strategy": name, **per_strategy[name],
             "delta_R": delta_r if name == ADAM_M_AWARE else None}
      rows.append(row)
    linear_j = linear_totals[target]["j"] / samples
    linear_d = np.cumsum(linear_totals[target]["d"] / samples)
    linear_r = float(np.sum(linear_j) / np.sum(linear_d))
    summary["targets"][str(target)] = {
        "delta_R": delta_r,
        "strategies": {
            name: {**per_strategy[name], "global_scalar": scalars[name][target],
                   "actual_median_relative_noise_ratio": float(np.median(np.concatenate(actual_ratios[name][target]))) }
            for name in per_strategy
        },
        "adam_m_aware_linear_reference": {"R_linear": linear_r,
          "adamw_minus_linear_R": per_strategy[ADAM_M_AWARE]["R"] - linear_r},
      }
  return rows, prefixes, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=ROOT / "exp2/config.yaml")
  args = parser.parse_args()
  document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
  if not isinstance(document, dict):
    raise ValueError("experiment config must be a mapping")
  trajectory = load_trajectory(_resolve(document["trajectory"]))
  fitted = document.get("strategy", {})
  spec_base = dict(horizon=trajectory["g"].shape[0], learning_rate=trajectory["learning_rate"],
                   beta1=trajectory["beta1"], weight_decay=trajectory["weight_decay"],
                   bandwidth=int(fitted.get("bandwidth", 4)), min_sep=int(fitted.get("min_sep", 1)),
                   max_participations=fitted.get("max_participations"), reduction=str(fitted.get("reduction", "mean")),
                   max_optimizer_steps=int(fitted.get("max_optimizer_steps", 1000)))
  strategy_dir = _resolve(document.get("strategy_dir", "exp2/strategies"))
  strategies = {name: load_or_fit_strategy(strategy_dir / f"{name}.npz", StrategySpec(name=name, **spec_base))
                for name in (DECAYED_PREFIX, ADAM_M_AWARE)}
  rows, prefixes, summary = run_replay(
      trajectory, strategies=strategies, samples=int(document.get("samples", 1000)), seed=int(document.get("seed", 0)),
      target_relative_noise=[float(x) for x in document.get("target_relative_noise", [0.01, 0.1, 1.0])],
      sample_batch_size=int(document.get("sample_batch_size", 4)),
  )
  output_dir = _resolve(document.get("output_dir", "exp2/results"))
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / "results.csv", rows, ["target_r", "strategy", "J", "D", "R", "delta_R"])
  _write_csv(output_dir / "prefix_results.csv", prefixes, ["target_r", "strategy", "prefix", "J_k", "D_k", "R_k"])
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"wrote {output_dir / 'results.csv'}")
  print(f"wrote {output_dir / 'prefix_results.csv'}")
  print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
  main()
