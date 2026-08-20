"""Paired dynamic/frozen AdamW replay against the exact frozen-p workload."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from exp5.hybrid_optimizer import FrozenPAdamW, FrozenPAdamWState
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
  numerator_dynamic: float | None = None
  numerator_frozen: float | None = None
  denominator: float | None = None


def _tree_zeros_like(tree):
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_sqnorm(tree):
  leaves = jax.tree_util.tree_leaves(tree)
  return sum((jnp.sum(jnp.square(leaf)) for leaf in leaves),
             start=jnp.asarray(0.0))


class FullPyTreeReplayAccumulator:
  """Online full-parameter replay without retaining a gradient trajectory."""

  def __init__(
      self, *, params: Any, dynamic_optimizer: optax.GradientTransformation,
      dynamic_state: Any, frozen_state: FrozenPAdamWState,
      learning_rate: float, beta1: float, weight_decay: float,
  ):
    self.dynamic_optimizer = dynamic_optimizer
    self.frozen_optimizer = FrozenPAdamW(learning_rate, beta1, weight_decay)
    self.dynamic_clean_params = params
    self.dynamic_noisy_params = params
    self.dynamic_clean_state = dynamic_state
    self.dynamic_noisy_state = dynamic_state
    self.frozen_clean_params = params
    self.frozen_noisy_params = params
    self.frozen_clean_state = frozen_state
    self.frozen_noisy_state = frozen_state
    self.p_star = frozen_state.p_star
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.rho = 1.0 - learning_rate * weight_decay
    self.delta_m = _tree_zeros_like(params)
    self.delta_theta = _tree_zeros_like(params)
    self.numerator_dynamic = jnp.asarray(0.0)
    self.numerator_frozen = jnp.asarray(0.0)
    self.denominator = jnp.asarray(0.0)

  @staticmethod
  def _apply(optimizer, gradients, state, params):
    updates, state = optimizer.update(gradients, state, params)
    return optax.apply_updates(params, updates), state

  def update(self, clipped_grad: Any, noise: Any) -> None:
    """Consume one exogenous clipped-gradient/noise PyTree pair."""
    private_grad = jax.tree_util.tree_map(
        lambda gradient, perturbation: gradient + perturbation,
        clipped_grad, noise)
    self.dynamic_clean_params, self.dynamic_clean_state = self._apply(
        self.dynamic_optimizer, clipped_grad, self.dynamic_clean_state,
        self.dynamic_clean_params)
    self.dynamic_noisy_params, self.dynamic_noisy_state = self._apply(
        self.dynamic_optimizer, private_grad, self.dynamic_noisy_state,
        self.dynamic_noisy_params)
    self.frozen_clean_params, self.frozen_clean_state = self._apply(
        self.frozen_optimizer, clipped_grad, self.frozen_clean_state,
        self.frozen_clean_params)
    self.frozen_noisy_params, self.frozen_noisy_state = self._apply(
        self.frozen_optimizer, private_grad, self.frozen_noisy_state,
        self.frozen_noisy_params)

    self.delta_m = jax.tree_util.tree_map(
        lambda old, perturbation: self.beta1 * old
        + (1.0 - self.beta1) * perturbation,
        self.delta_m, noise)
    global_t = self.frozen_clean_state.count
    correction = 1.0 - jnp.asarray(self.beta1) ** global_t
    self.delta_theta = jax.tree_util.tree_map(
        lambda old, p, moment: self.rho * old
        - self.learning_rate * p * moment / correction,
        self.delta_theta, self.p_star, self.delta_m)

    dynamic_delta = jax.tree_util.tree_map(
        lambda noisy, clean: noisy - clean,
        self.dynamic_noisy_params, self.dynamic_clean_params)
    frozen_delta = jax.tree_util.tree_map(
        lambda noisy, clean: noisy - clean,
        self.frozen_noisy_params, self.frozen_clean_params)
    dynamic_error = jax.tree_util.tree_map(
        lambda actual, linear: actual - linear,
        dynamic_delta, self.delta_theta)
    frozen_error = jax.tree_util.tree_map(
        lambda actual, linear: actual - linear,
        frozen_delta, self.delta_theta)
    self.numerator_dynamic += _tree_sqnorm(dynamic_error)
    self.numerator_frozen += _tree_sqnorm(frozen_error)
    self.denominator += _tree_sqnorm(self.delta_theta)

  def result(self) -> ReplayResult:
    denominator = float(self.denominator)
    floor = np.finfo(np.float64).tiny
    numerator_dynamic = float(self.numerator_dynamic)
    numerator_frozen = float(self.numerator_frozen)
    return ReplayResult(
        numerator_dynamic / max(denominator, floor),
        numerator_frozen / max(denominator, floor),
        numerator_dynamic, numerator_frozen, denominator)


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
  denominator = float(np.sum(linear_delta ** 2))
  numerator_dynamic = float(np.sum((dynamic_delta - linear_delta) ** 2))
  numerator_frozen = float(np.sum((frozen_delta - linear_delta) ** 2))
  return ReplayResult(
      gap(dynamic_delta, linear_delta), gap(frozen_delta, linear_delta),
      numerator_dynamic, numerator_frozen, denominator)


__all__ = [
    "FullPyTreeReplayAccumulator", "ReplayResult", "filter_latent_draws",
    "gap", "paired_replay",
]
