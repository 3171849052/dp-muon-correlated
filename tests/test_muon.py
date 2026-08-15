import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.optim import classic_nesterov_momentum, init_muon_nesterov_state, muon_nesterov_step


def test_optax_classic_nesterov_matches_existing_muon_step_each_step():
  beta = 0.95
  gradients = [
      jnp.array([[1.0, -2.0], [0.5, 3.0]], jnp.float32),
      jnp.array([[-0.3, 1.2], [4.0, -1.0]], jnp.float32),
      jnp.array([[2.5, 0.0], [-1.5, 0.7]], jnp.float32),
  ]
  transformation = classic_nesterov_momentum(beta)
  optax_state = transformation.init(gradients[0])
  reference_state = init_muon_nesterov_state(gradients[0])
  for gradient in gradients:
    optax_update, optax_state = transformation.update(gradient, optax_state)
    reference_update, reference_state = muon_nesterov_step(
        reference_state, gradient, beta
    )
    np.testing.assert_allclose(optax_update, reference_update, rtol=1e-6, atol=1e-6)


def test_optax_classic_nesterov_jit_matches_eager():
  transformation = classic_nesterov_momentum(0.8)
  gradient = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
  state = transformation.init(gradient)
  eager = transformation.update(gradient, state)
  compiled = jax.jit(transformation.update)(gradient, state)
  np.testing.assert_allclose(compiled[0], eager[0])
