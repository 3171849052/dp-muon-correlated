"""Tests for exact EMA-then-Nesterov Muon pre-Q dynamics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dp_muon.optim import (
    init_muon_nesterov_state,
    muon_nesterov_step,
    nesterov_kernel_coef,
)


def _dense_toeplitz(coef: jax.Array) -> jax.Array:
  horizon = coef.shape[0]
  offsets = jnp.arange(horizon)[:, None] - jnp.arange(horizon)[None, :]
  return jnp.where(offsets >= 0, coef[jnp.maximum(offsets, 0)], 0.0)


def test_runtime_updates_equal_dense_nesterov_kernel_times_gradients():
  beta = 0.6
  gradients = jnp.array([1.2, -0.7, 3.0, 0.25], dtype=jnp.float32)
  state = init_muon_nesterov_state(gradients[0])
  updates = []
  for gradient in gradients:
    update, state = muon_nesterov_step(state, gradient, beta)
    updates.append(update)
  dense_h = _dense_toeplitz(nesterov_kernel_coef(len(gradients), beta))
  np.testing.assert_allclose(jnp.stack(updates), dense_h @ gradients, rtol=1e-6, atol=1e-6)


def test_fixed_lr_parameter_trajectory_equals_negative_workload_times_gradients():
  beta, learning_rate = 0.75, 0.1
  gradients = jnp.array([2.0, -1.0, 0.5, 3.0], dtype=jnp.float32)
  initial = jnp.array(4.0, dtype=jnp.float32)
  parameter = initial
  state = init_muon_nesterov_state(parameter)
  trajectory = []
  for gradient in gradients:
    update, state = muon_nesterov_step(state, gradient, beta)
    parameter = parameter - learning_rate * update
    trajectory.append(parameter - initial)
  workload = learning_rate * jnp.cumsum(nesterov_kernel_coef(len(gradients), beta))
  np.testing.assert_allclose(
      jnp.stack(trajectory), -_dense_toeplitz(workload) @ gradients, rtol=1e-6, atol=1e-6
  )


def test_pytree_multiple_leaves_and_jit_match_eager():
  gradient = {
      "matrix": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
      "vector": jnp.array([-1.0, 2.0], dtype=jnp.float32),
  }
  state = init_muon_nesterov_state(gradient)
  eager_update, eager_state = muon_nesterov_step(state, gradient, 0.9)
  jitted_update, jitted_state = jax.jit(muon_nesterov_step)(state, gradient, 0.9)
  for eager, jitted in zip(
      jax.tree_util.tree_leaves((eager_update, eager_state)),
      jax.tree_util.tree_leaves((jitted_update, jitted_state)),
      strict=True,
  ):
    np.testing.assert_allclose(eager, jitted)
  assert jitted_state.step == 1


def test_checkpoint_resume_matches_uninterrupted_steps():
  gradients = jnp.linspace(-1.0, 2.0, 10)
  state = init_muon_nesterov_state(gradients[0])
  uninterrupted = []
  for gradient in gradients:
    update, state = muon_nesterov_step(state, gradient, 0.85)
    uninterrupted.append(update)
  state = init_muon_nesterov_state(gradients[0])
  resumed = []
  for gradient in gradients[:4]:
    update, state = muon_nesterov_step(state, gradient, 0.85)
    resumed.append(update)
  checkpoint = state
  for gradient in gradients[4:]:
    update, checkpoint = muon_nesterov_step(checkpoint, gradient, 0.85)
    resumed.append(update)
  np.testing.assert_array_equal(jnp.stack(uninterrupted), jnp.stack(resumed))
  assert checkpoint.step == 10


@pytest.mark.parametrize("momentum", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_momentum_fails_fast(momentum):
  state = init_muon_nesterov_state(jnp.array(0.0))
  with pytest.raises(ValueError, match="momentum"):
    muon_nesterov_step(state, jnp.array(1.0), momentum)
