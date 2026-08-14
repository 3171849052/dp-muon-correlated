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


def _assert_tree_allclose(actual, expected):
  assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(actual),
      jax.tree_util.tree_leaves(expected),
      strict=True,
  ):
    np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-6)


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


def test_tuple_pytree_preserves_structure_and_nesterov_recurrence():
  beta = 0.6
  first_gradient = (
      jnp.array([1.0, -2.0], dtype=jnp.float32),
      jnp.array([[3.0]], dtype=jnp.float32),
  )
  second_gradient = (
      jnp.array([-0.5, 4.0], dtype=jnp.float32),
      jnp.array([[2.0]], dtype=jnp.float32),
  )
  state = init_muon_nesterov_state(first_gradient)
  first_update, state = muon_nesterov_step(state, first_gradient, beta)
  second_update, new_state = muon_nesterov_step(state, second_gradient, beta)
  assert jax.tree_util.tree_structure(first_update) == jax.tree_util.tree_structure(first_gradient)
  assert jax.tree_util.tree_structure(new_state.momentum) == jax.tree_util.tree_structure(second_gradient)
  expected_first = jax.tree_util.tree_map(lambda grad: (1.0 - beta**2) * grad, first_gradient)
  expected_momentum = jax.tree_util.tree_map(
      lambda first, second: beta * (1.0 - beta) * first + (1.0 - beta) * second,
      first_gradient,
      second_gradient,
  )
  expected_second = jax.tree_util.tree_map(
      lambda grad, momentum: (1.0 - beta) * grad + beta * momentum,
      second_gradient,
      expected_momentum,
  )
  _assert_tree_allclose(first_update, expected_first)
  _assert_tree_allclose(second_update, expected_second)
  _assert_tree_allclose(new_state.momentum, expected_momentum)


def test_nested_tuple_mixed_pytree_preserves_structure_and_values():
  beta = 0.25
  gradient = {
      "a": (
          jnp.array([1.0, 2.0], dtype=jnp.float32),
          jnp.array([-3.0], dtype=jnp.float32),
      ),
      "b": {"c": jnp.array([[4.0]], dtype=jnp.float32)},
  }
  state = init_muon_nesterov_state(gradient)
  update, new_state = muon_nesterov_step(state, gradient, beta)
  assert jax.tree_util.tree_structure(update) == jax.tree_util.tree_structure(gradient)
  assert jax.tree_util.tree_structure(new_state.momentum) == jax.tree_util.tree_structure(gradient)
  expected_update = jax.tree_util.tree_map(lambda grad: (1.0 - beta**2) * grad, gradient)
  expected_momentum = jax.tree_util.tree_map(lambda grad: (1.0 - beta) * grad, gradient)
  _assert_tree_allclose(update, expected_update)
  _assert_tree_allclose(new_state.momentum, expected_momentum)


def test_tuple_pytree_jit_matches_eager():
  gradient = (
      jnp.array([1.0, -2.0], dtype=jnp.float32),
      (jnp.array([3.0], dtype=jnp.float32),),
  )
  state = init_muon_nesterov_state(gradient)
  eager_update, eager_state = muon_nesterov_step(state, gradient, 0.8)
  jitted_update, jitted_state = jax.jit(muon_nesterov_step)(state, gradient, 0.8)
  _assert_tree_allclose(jitted_update, eager_update)
  _assert_tree_allclose(jitted_state.momentum, eager_state.momentum)
  np.testing.assert_array_equal(jitted_state.step, eager_state.step)


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
