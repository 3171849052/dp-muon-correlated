#!/usr/bin/env python3
"""Replay frozen Muon pre-Q updates through correlated BandInvMF noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dp_muon.analysis import make_causal_noise_operator, relative_noise_ratios
from dp_muon.bandinvmf import (
    BandInvMFArtifactMetadata,
    BandInvMFStrategy,
    load_bandinv_strategy,
    load_bandinv_strategy_metadata,
)
from dp_muon.optim import MUON_Q_STAGES, muon_q_stages


def _resolve(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> Any:
  value = np.asarray(archive[name])
  if value.shape != ():
    raise ValueError(f"trajectory field {name!r} must be a scalar")
  return value.item()


def load_trajectory(path: str | Path) -> dict[str, Any]:
  """Loads the public frozen-trajectory format produced by collect_trajectory."""
  required = {
      "u", "learning_rates", "parameter_name", "start_step", "momentum",
      "ns_steps", "consistent_rms",
  }
  with np.load(path, allow_pickle=False) as archive:
    missing = required.difference(archive.files)
    if missing:
      raise ValueError(f"trajectory is missing fields: {sorted(missing)}")
    u = np.asarray(archive["u"], dtype=np.float32)
    learning_rates = np.asarray(archive["learning_rates"], dtype=np.float64)
    if u.ndim != 3 or u.shape[0] < 1 or min(u.shape[1:]) < 1:
      raise ValueError("trajectory u must have shape (T,m,n) with non-empty matrices")
    if learning_rates.shape != (u.shape[0],) or not np.all(np.isfinite(learning_rates)):
      raise ValueError("trajectory learning_rates must be finite with shape (T,)")
    if np.any(learning_rates <= 0):
      raise ValueError("trajectory learning_rates must be positive")
    result = {
        "u": u,
        "learning_rates": learning_rates,
        "parameter_name": str(_scalar(archive, "parameter_name")),
        "start_step": int(_scalar(archive, "start_step")),
        "momentum": float(_scalar(archive, "momentum")),
        "ns_steps": int(_scalar(archive, "ns_steps")),
        "consistent_rms": float(_scalar(archive, "consistent_rms")),
        "use_bf16_ns": bool(_scalar(archive, "use_bf16_ns")) if "use_bf16_ns" in archive.files else True,
    }
  if not 0 <= result["momentum"] < 1 or result["ns_steps"] < 1 or result["consistent_rms"] <= 0:
    raise ValueError("trajectory Muon metadata is invalid")
  if result["start_step"] != 0:
    raise ValueError(
        "Exp1 currently requires trajectory start_step == 0; mid/late replay "
        "needs history-aware momentum/noise burn-in"
    )
  return result


def _trajectory_learning_rate(trajectory: dict[str, Any]) -> float:
  learning_rates = np.asarray(trajectory["learning_rates"], dtype=np.float64)
  learning_rate = float(learning_rates[0])
  if not np.allclose(learning_rates, learning_rate, rtol=1e-7, atol=1e-12):
    raise ValueError(
        "Exp1's nesterov-trajectory strategy validation currently requires "
        "a constant frozen trajectory learning rate"
    )
  return learning_rate


def validate_exp1_strategy(
    strategy: BandInvMFStrategy,
    metadata: BandInvMFArtifactMetadata,
    trajectory: dict[str, Any],
) -> None:
  """Rejects strategies that do not optimize ``A=eta P H_beta^Nes``."""
  if strategy.horizon != trajectory["u"].shape[0]:
    raise ValueError("BandInvMF strategy horizon must equal frozen trajectory length")
  if metadata.workload_type != "nesterov-trajectory":
    raise ValueError(
        "Exp1 requires strategy workload_type='nesterov-trajectory', not "
        f"{metadata.workload_type!r}"
    )
  if metadata.momentum is None:
    raise ValueError("Exp1 strategy metadata is missing momentum")
  if not np.isclose(metadata.momentum, trajectory["momentum"], rtol=1e-7, atol=1e-12):
    raise ValueError(
        "Exp1 strategy momentum does not match trajectory momentum "
        f"({metadata.momentum} != {trajectory['momentum']})"
    )
  trajectory_learning_rate = _trajectory_learning_rate(trajectory)
  if metadata.learning_rate is None:
    raise ValueError("Exp1 strategy metadata is missing learning_rate")
  if not np.isclose(metadata.learning_rate, trajectory_learning_rate, rtol=1e-7, atol=1e-12):
    raise ValueError(
        "Exp1 strategy learning_rate does not match frozen trajectory "
        f"learning rate ({metadata.learning_rate} != {trajectory_learning_rate})"
    )


def _q_clean(u: np.ndarray, trajectory: dict[str, Any]) -> dict[str, np.ndarray]:
  q = lambda matrix: muon_q_stages(
      matrix,
      ns_steps=trajectory["ns_steps"],
      consistent_rms=trajectory["consistent_rms"],
      use_bf16_ns=trajectory["use_bf16_ns"],
  )
  result = jax.vmap(q)(jnp.asarray(u))
  return {stage: np.asarray(result[stage]) for stage in MUON_Q_STAGES}


def _q_deltas(
    u: np.ndarray,
    noise: np.ndarray,
    clean_q: dict[str, np.ndarray],
    trajectory: dict[str, Any],
) -> dict[str, np.ndarray]:
  """Computes all Q-stage deltas with exactly the same U and E inputs."""
  q = lambda matrix: muon_q_stages(
      matrix,
      ns_steps=trajectory["ns_steps"],
      consistent_rms=trajectory["consistent_rms"],
      use_bf16_ns=trajectory["use_bf16_ns"],
  )
  perturbed = jax.vmap(jax.vmap(q))(jnp.asarray(u)[None, ...] + jnp.asarray(noise))
  deltas = {stage: np.asarray(perturbed[stage]) - clean_q[stage][None, ...] for stage in MUON_Q_STAGES}
  # This is intentionally assigned rather than obtained by subtraction: Q_0 is
  # mathematically linear and must satisfy Delta=E exactly.
  deltas["linear"] = np.asarray(noise)
  return deltas


def _accumulate_prefix(
    delta: np.ndarray,
    learning_rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  x = np.asarray(delta, dtype=np.float64) * learning_rates[None, :, None, None]
  prefix = np.cumsum(x, axis=1)
  return (
      np.sum(np.sum(prefix * prefix, axis=(2, 3)), axis=0),
      np.sum(np.sum(x * x, axis=(2, 3)), axis=0),
  )


def run_replay(
    trajectory: dict[str, Any],
    *,
    noising_coef: np.ndarray,
    samples: int,
    seed: int,
    target_median_r: list[float],
    sample_batch_size: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  """Runs the replay and returns rows for results, prefixes, and JSON summary."""
  if samples < 1 or sample_batch_size < 1:
    raise ValueError("samples and sample_batch_size must be positive")
  if not target_median_r or any(value <= 0 for value in target_median_r):
    raise ValueError("target_median_r must contain positive values")
  if trajectory["start_step"] != 0:
    raise ValueError(
        "Exp1 currently requires trajectory start_step == 0; mid/late replay "
        "needs history-aware momentum/noise burn-in"
    )
  u, learning_rates = trajectory["u"], trajectory["learning_rates"]
  horizon = u.shape[0]
  operator = make_causal_noise_operator(noising_coef, horizon, trajectory["momentum"])
  clean_q = _q_clean(u, trajectory)
  # The pilot uses exactly the same raw Gaussian draws as the formal replay,
  # but retains only O(samples*T) relative norms rather than full matrices.
  calibration_rng = np.random.default_rng(seed)
  calibration_ratios: list[np.ndarray] = []
  for offset in range(0, samples, sample_batch_size):
    batch = min(sample_batch_size, samples - offset)
    latent = calibration_rng.standard_normal((batch, *u.shape), dtype=np.float32)
    raw_noise = np.einsum("ts,bsij->btij", operator.total, latent, optimize=True).astype(np.float32)
    calibration_ratios.append(relative_noise_ratios(u, raw_noise).reshape(-1))
  reference_median_r = float(np.median(np.concatenate(calibration_ratios)))
  if reference_median_r == 0 or not np.isfinite(reference_median_r):
    raise ValueError("raw noise has an invalid overall median relative norm")
  global_scalars = {target: float(target / reference_median_r) for target in target_median_r}

  rng = np.random.default_rng(seed)
  totals = {
      target: {stage: [np.zeros(horizon), np.zeros(horizon)] for stage in MUON_Q_STAGES}
      for target in target_median_r
  }
  actual_ratios: dict[float, list[np.ndarray]] = {target: [] for target in target_median_r}
  for offset in range(0, samples, sample_batch_size):
    batch = min(sample_batch_size, samples - offset)
    latent = rng.standard_normal((batch, *u.shape), dtype=np.float32)
    raw_noise = np.einsum("ts,bsij->btij", operator.total, latent, optimize=True).astype(np.float32)
    for target in target_median_r:
      # One deterministic scalar is shared by every formal Monte-Carlo sample
      # at this target.  No sample-dependent or step-dependent rescaling is
      # permitted because either would change the Gaussian transcript law.
      noise = (global_scalars[target] * raw_noise).astype(np.float32)
      actual_ratios[target].append(relative_noise_ratios(u, noise).reshape(-1))
      deltas = _q_deltas(u, noise, clean_q, trajectory)
      for stage, delta in deltas.items():
        j_sum, d_sum = _accumulate_prefix(delta, learning_rates)
        totals[target][stage][0] += j_sum
        totals[target][stage][1] += d_sum

  result_rows: list[dict[str, Any]] = []
  prefix_rows: list[dict[str, Any]] = []
  summary: dict[str, Any] = {
      "samples": samples,
      "seed": seed,
      "trajectory": {
          key: trajectory[key] for key in (
              "parameter_name", "start_step", "momentum", "ns_steps", "consistent_rms", "use_bf16_ns"
          )
      },
      "operator": "H_beta^Nes C^{-1}",
      "targets": {},
  }
  for target in target_median_r:
    previous: float | None = None
    target_summary: dict[str, Any] = {
        "global_scalar": global_scalars[target],
        "actual_median_relative_noise_ratio": float(
            np.median(np.concatenate(actual_ratios[target]))
        ),
        "stages": {},
    }
    for stage in MUON_Q_STAGES:
      j = totals[target][stage][0] / samples
      d_increment = totals[target][stage][1] / samples
      d = np.cumsum(d_increment)
      r_prefix = j / d
      aggregate = float(np.sum(j) / np.sum(d))
      delta_r = None if previous is None else aggregate - previous
      result_rows.append({
          "target_r": target, "stage": stage, "J": float(np.sum(j)),
          "D": float(np.sum(d)), "R": aggregate, "delta_R": delta_r,
      })
      target_summary["stages"][stage] = {"J": float(np.sum(j)), "D": float(np.sum(d)), "R": aggregate, "delta_R": delta_r}
      for prefix, (j_k, d_k, r_k) in enumerate(zip(j, d, r_prefix, strict=True), start=1):
        prefix_rows.append({"target_r": target, "stage": stage, "prefix": prefix, "J_k": float(j_k), "D_k": float(d_k), "R_k": float(r_k)})
      previous = aggregate
    summary["targets"][str(target)] = target_summary
  return result_rows, prefix_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=ROOT / "exp1/config.yaml")
  args = parser.parse_args()
  document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
  if not isinstance(document, dict):
    raise ValueError("experiment config must be a mapping")
  trajectory = load_trajectory(_resolve(document["trajectory"]))
  strategy_path = _resolve(document["strategy"])
  strategy = load_bandinv_strategy(strategy_path)
  try:
    metadata = load_bandinv_strategy_metadata(strategy_path)
  except ValueError as error:
    raise ValueError(
        "Exp1 requires BandInvMF artifact metadata for nesterov-trajectory, "
        f"momentum, and learning_rate validation: {error}"
    ) from error
  validate_exp1_strategy(strategy, metadata, trajectory)
  rows, prefixes, summary = run_replay(
      trajectory,
      noising_coef=np.asarray(strategy.noising_coef),
      samples=int(document.get("samples", 1000)),
      seed=int(document.get("seed", 0)),
      target_median_r=[float(value) for value in document.get("target_median_r", [0.01, 0.1, 1.0])],
      sample_batch_size=int(document.get("sample_batch_size", 4)),
  )
  output_dir = _resolve(document.get("output_dir", "exp1/results"))
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / "results.csv", rows, ["target_r", "stage", "J", "D", "R", "delta_R"])
  _write_csv(output_dir / "prefix_results.csv", prefixes, ["target_r", "stage", "prefix", "J_k", "D_k", "R_k"])
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"wrote {output_dir / 'results.csv'}")
  print(f"wrote {output_dir / 'prefix_results.csv'}")
  print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
  main()
