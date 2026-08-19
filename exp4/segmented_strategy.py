"""Block BandInvMF construction without resetting AdamW state."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
from dp_muon.bandinvmf import BandInvMFStrategy, fit_bandinv_strategy, init_bandinv_noise_state, BandInvMFNoiseState
from dp_muon.privacy import calibrate_nonamplified_bandinv, PrivacyCalibration

@dataclass(frozen=True)
class SegmentedPlan:
  condition: str
  block_lengths: tuple[int, ...]
  strategies: tuple[BandInvMFStrategy, ...]
  calibration: PrivacyCalibration

def block_lengths(horizon: int, block_size: int) -> tuple[int, ...]:
  if horizon < 1 or block_size < 1: raise ValueError("horizon and block_size must be positive")
  return tuple([block_size] * (horizon // block_size) + ([horizon % block_size] if horizon % block_size else []))

def fit_segmented_plan(*, horizon: int, block_size: int, bandwidth: int, min_sep: int,
    max_participations: int | None, max_optimizer_steps: int, reduction: str,
    learning_rate: float, weight_decay: float, epsilon: float, delta: float,
    clip_norm: float, normalize_by: float, adjacency: str) -> SegmentedPlan:
  lengths = block_lengths(horizon, block_size)
  strategies = tuple(fit_bandinv_strategy(n, min(bandwidth, n), min_sep=min(min_sep, n),
      max_participations=max_participations, max_optimizer_steps=max_optimizer_steps,
      reduction=reduction, workload_coef=jnp.ones((n,))) for n in lengths)
  # Block-diagonal transcript sensitivity is the sum of squared block sensitivities.
  total_sensitivity_squared = float(sum(float(s.sensitivity_squared) for s in strategies))
  calibration = calibrate_nonamplified_bandinv(epsilon=epsilon, delta=delta,
      clip_norm=clip_norm, normalize_by=normalize_by, adjacency=adjacency,
      sensitivity_squared=total_sensitivity_squared)
  condition = "seg" + str(block_size)
  return SegmentedPlan(condition, lengths, strategies, calibration)

def reset_noise_state(params: Any, state: BandInvMFNoiseState, *, bandwidth: int, global_step: int) -> BandInvMFNoiseState:
  """Reset only FIR noise memory; preserve transcript step metadata."""
  fresh = init_bandinv_noise_state(params, bandwidth)
  return BandInvMFNoiseState(fresh.buffer, fresh.cursor, jnp.asarray(global_step, dtype=fresh.step.dtype), bandwidth)

__all__ = ["SegmentedPlan", "block_lengths", "fit_segmented_plan", "reset_noise_state"]
