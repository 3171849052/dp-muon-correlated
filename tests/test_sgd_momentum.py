import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.optim import init_sgd_momentum_state, sgd_momentum_step


def _assert_tree_allclose(actual, expected):
  for left, right in zip(
      jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
  ):
    np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-7)


def test_zero_momentum_is_exactly_sgd():
  gradient = {"a": jnp.array([1.0, -2.0]), "b": (jnp.array(3.0),)}
  state = init_sgd_momentum_state(gradient)
  velocity, next_state = sgd_momentum_step(state, gradient, momentum=0.0)
  _assert_tree_allclose(velocity, gradient)
  _assert_tree_allclose(next_state.velocity, gradient)
  assert int(next_state.step) == 1


def test_fixed_gradient_sequence_matches_hand_calculated_velocity():
  state = init_sgd_momentum_state(jnp.array(0.0, dtype=jnp.float32))
  actual = []
  for gradient in (1.0, -2.0, 3.0):
    velocity, state = sgd_momentum_step(
        state, jnp.array(gradient, dtype=jnp.float32), momentum=0.5
    )
    actual.append(velocity)
  np.testing.assert_allclose(jnp.stack(actual), jnp.array([1.0, -1.5, 2.25]))
  assert int(state.step) == 3


def test_jit_and_eager_step_match_for_nested_pytree():
  params = {"x": jnp.array([0.0, 0.0]), "y": (jnp.array([[0.0]]),)}
  gradient = {"x": jnp.array([1.0, -2.0]), "y": (jnp.array([[3.0]]),)}
  state = init_sgd_momentum_state(params)
  eager = sgd_momentum_step(state, gradient, 0.75)
  compiled = jax.jit(lambda s, g: sgd_momentum_step(s, g, 0.75))(state, gradient)
  _assert_tree_allclose(compiled[0], eager[0])
  _assert_tree_allclose(compiled[1].velocity, eager[1].velocity)
  np.testing.assert_array_equal(compiled[1].step, eager[1].step)
