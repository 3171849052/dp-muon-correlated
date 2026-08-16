import math

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.analysis import cancellation_statistics
from dp_muon.optim import muon_q_stages, muon_transform


def test_linear_q_stage_delta_is_exactly_input_noise():
  u = jnp.array([[1.25, -0.5], [0.75, 2.0]], dtype=jnp.float32)
  e = jnp.array([[0.125, 0.25], [-0.5, 0.75]], dtype=jnp.float32)
  delta = muon_q_stages(u + e, use_bf16_ns=False)["linear"] - muon_q_stages(
      u, use_bf16_ns=False
  )["linear"]
  np.testing.assert_array_equal(delta, e)


def test_scale_stage_has_same_cancellation_as_ns_when_its_constant_is_one():
  # consistent_rms=1/sqrt(max(m,n)) makes the production scale factor one.
  u = np.array([
      [[1.0, -2.0], [0.25, 3.0]],
      [[-1.5, 0.75], [2.0, -0.5]],
  ], dtype=np.float32)
  e = np.array([
      [[[0.2, -0.1], [0.3, 0.4]], [[-0.2, 0.1], [-0.3, -0.4]]],
      [[[0.1, 0.2], [-0.4, 0.3]], [[-0.1, -0.2], [0.4, -0.3]]],
  ], dtype=np.float32)
  q = lambda value: muon_q_stages(
      value, ns_steps=3, consistent_rms=1 / math.sqrt(2), use_bf16_ns=False
  )
  clean = jax.vmap(q)(jnp.asarray(u))
  perturbed = jax.vmap(jax.vmap(q))(jnp.asarray(u)[None] + jnp.asarray(e))
  ns_delta = np.asarray(perturbed["ns"] - clean["ns"][None])
  scale_delta = np.asarray(perturbed["scale"] - clean["scale"][None])
  ns_stats = cancellation_statistics(ns_delta, np.ones(2))
  scale_stats = cancellation_statistics(scale_delta, np.ones(2))
  np.testing.assert_allclose(scale_stats["aggregate_R"], ns_stats["aggregate_R"], rtol=1e-6, atol=1e-6)


def test_full_q_matches_standard_muon_post_nesterov_tail():
  matrix = jnp.array([[1.0, -0.25, 2.0], [0.75, 3.0, -1.0]], dtype=jnp.float32)
  # beta=0 makes the first production Muon update's pre-Q input exactly matrix;
  # the negative sign is solely Optax's final learning-rate descent scale.
  production = muon_transform(
      learning_rate=1.0,
      weight_decay=0.0,
      momentum=0.0,
      ns_steps=3,
      consistent_rms=0.2,
      use_bf16_ns=True,
  )
  update, _ = production.update(matrix, production.init(matrix), matrix)
  expected = muon_q_stages(
      matrix, ns_steps=3, consistent_rms=0.2, use_bf16_ns=True
  )["scale"]
  np.testing.assert_allclose(-np.asarray(update), np.asarray(expected), rtol=1e-6, atol=1e-6)
