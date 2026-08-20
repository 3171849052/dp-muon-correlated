"""Paired dynamic/frozen AdamW replay against the exact frozen-p workload."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from exp5.workload import frozen_p_time_workload


def filter_latent_draws(latent: np.ndarray, noising_blocks: tuple[np.ndarray, ...]) -> np.ndarray:
  """Apply block-local square ``D`` matrices to shared latent draws."""
  latent = np.asarray(latent, dtype=np.float64)
  output = np.empty_like(latent)
  offset = 0
  for matrix in noising_blocks:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
      raise ValueError("each noising block must be square")
    length = matrix.shape[0]
    output[offset:offset + length] = np.einsum(
        "ts,s...->t...", matrix, latent[offset:offset + length])
    offset += length
  if offset != len(latent):
    raise ValueError("noising blocks do not cover the replay horizon")
  return output


def _adam_trajectory(gradients, *, params, mu, nu, count, beta1, beta2, eps,
                     learning_rate, weight_decay, p_star=None):
  gradients = np.asarray(gradients, dtype=np.float64)
  theta = np.asarray(params, dtype=np.float64).copy()
  m = np.asarray(mu, dtype=np.float64).copy()
  v = np.asarray(nu, dtype=np.float64).copy()
  rows = np.empty_like(gradients)
  for local, gradient in enumerate(gradients):
    global_t = count + local + 1
    m = beta1 * m + (1.0 - beta1) * gradient
    if p_star is None:
      v = beta2 * v + (1.0 - beta2) * gradient * gradient
      p = 1.0 / (np.sqrt(v / (1.0 - beta2 ** global_t)) + eps)
    else:
      p = p_star
    theta = (1.0 - learning_rate * weight_decay) * theta \
        - learning_rate * p * m / (1.0 - beta1 ** global_t)
    rows[local] = theta
  return rows


def gap(actual: np.ndarray, linear: np.ndarray) -> float:
  denominator = float(np.sum(np.asarray(linear) ** 2))
  return float(np.sum((np.asarray(actual) - np.asarray(linear)) ** 2) /
               max(denominator, np.finfo(np.float64).tiny))


@dataclass(frozen=True)
class ReplayResult:
  g_dynamic: float
  g_frozen: float


def paired_replay(clean_gradients: np.ndarray, noise: np.ndarray, *, params: np.ndarray,
                  mu: np.ndarray, nu: np.ndarray, count: int, beta1: float,
                  beta2: float, eps: float, learning_rate: float,
                  weight_decay: float) -> ReplayResult:
  """Replay identical noise through dynamic, frozen, and exact linear paths."""
  correction = 1.0 - beta2 ** count
  p_star = 1.0 / (np.sqrt(np.asarray(nu) / correction) + eps)
  common = dict(params=params, mu=mu, nu=nu, count=count, beta1=beta1,
                beta2=beta2, eps=eps, learning_rate=learning_rate,
                weight_decay=weight_decay)
  clean_dynamic = _adam_trajectory(clean_gradients, **common)
  noisy_dynamic = _adam_trajectory(clean_gradients + noise, **common)
  clean_frozen = _adam_trajectory(clean_gradients, p_star=p_star, **common)
  noisy_frozen = _adam_trajectory(clean_gradients + noise, p_star=p_star, **common)
  dynamic_delta = noisy_dynamic - clean_dynamic
  frozen_delta = noisy_frozen - clean_frozen
  time = frozen_p_time_workload(
      len(clean_gradients), tau=count, beta1=beta1,
      learning_rate=learning_rate, weight_decay=weight_decay)
  linear_delta = np.einsum("ts,s...,...->t...", time, noise, p_star)
  return ReplayResult(gap(dynamic_delta, linear_delta), gap(frozen_delta, linear_delta))


__all__ = ["ReplayResult", "filter_latent_draws", "gap", "paired_replay"]
