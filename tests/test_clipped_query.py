"""Tests for the fixed-normalization JAX Privacy clipping adapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dp_accounting import NeighboringRelation

from dp_muon.privacy import compute_query_sensitivity, make_clipped_gradient_query


def _linear_loss(params, batch):
  """One-example loss; keep_batch_dim supplies a leading axis of size one."""
  return params["scalar"] * batch["scalar"][0] + jnp.vdot(params["vector"], batch["vector"][0])


def _manual_clipped_query(params, batch, clip_norm, normalize_by):
  raw = jax.vmap(
      jax.grad(lambda p, scalar, vector: p["scalar"] * scalar + jnp.vdot(p["vector"], vector)),
      in_axes=(None, 0, 0),
  )(params, batch["scalar"], batch["vector"])
  squared_norm = sum(
      jnp.sum(leaf**2, axis=tuple(range(1, leaf.ndim))) for leaf in jax.tree_util.tree_leaves(raw)
  )
  factors = jnp.minimum(1.0, clip_norm / jnp.sqrt(squared_norm))
  return jax.tree_util.tree_map(
      lambda leaf: jnp.sum(
          leaf * factors.reshape((factors.shape[0],) + (1,) * (leaf.ndim - 1)), axis=0
      ) / normalize_by,
      raw,
  )


def test_matches_manual_per_example_global_clipping_reference():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {
      "scalar": jnp.array([3.0, 1.0]),
      "vector": jnp.array([[0.0, 4.0], [2.0, -2.0]]),
  }
  query = make_clipped_gradient_query(_linear_loss, clip_norm=4.0, normalize_by=5.0)
  actual = query(params, batch)
  expected = _manual_clipped_query(params, batch, clip_norm=4.0, normalize_by=5.0)
  jax.tree_util.tree_map(lambda a, e: np.testing.assert_allclose(a, e), actual, expected)


def test_clipping_is_global_across_parameter_leaves():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {"scalar": jnp.array([3.0]), "vector": jnp.array([[0.0, 4.0]])}
  query = make_clipped_gradient_query(_linear_loss, clip_norm=4.0, normalize_by=1.0)
  output = query(params, batch)
  # The full PyTree norm is 5, so a global clip at 4 uses factor 4/5.
  np.testing.assert_allclose(output["scalar"], 2.4)
  np.testing.assert_allclose(output["vector"], jnp.array([0.0, 3.2]))
  # Independent per-leaf clipping would have left (3, [0, 4]) unchanged.
  assert not np.isclose(float(output["scalar"]), 3.0)


def test_fixed_normalization_is_not_actual_batch_size():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {"scalar": jnp.array([1.0, 3.0]), "vector": jnp.zeros((2, 2))}
  query = make_clipped_gradient_query(_linear_loss, clip_norm=10.0, normalize_by=4.0)
  output = query(params, batch)
  np.testing.assert_allclose(output["scalar"], (1.0 + 3.0) / 4.0)


def test_sensitivity_matches_jax_privacy_and_m1_convention():
  clip_norm, normalize_by = 3.0, 8.0
  query = make_clipped_gradient_query(_linear_loss, clip_norm=clip_norm, normalize_by=normalize_by)
  assert query.sensitivity(NeighboringRelation.ADD_OR_REMOVE_ONE) == clip_norm / normalize_by
  assert query.sensitivity(NeighboringRelation.REPLACE_ONE) == 2 * clip_norm / normalize_by
  assert query.sensitivity(NeighboringRelation.ADD_OR_REMOVE_ONE) == compute_query_sensitivity(
      clip_norm, normalize_by, "add_remove"
  )
  assert query.sensitivity(NeighboringRelation.REPLACE_ONE) == compute_query_sensitivity(
      clip_norm, normalize_by, "replace_one"
  )


def test_query_is_jittable_and_matches_eager():
  params = {"scalar": jnp.array(0.0), "vector": jnp.zeros(2)}
  batch = {"scalar": jnp.array([3.0, 1.0]), "vector": jnp.array([[0.0, 4.0], [2.0, -2.0]])}
  query = make_clipped_gradient_query(_linear_loss, clip_norm=4.0, normalize_by=5.0)
  eager = query(params, batch)
  jitted = jax.jit(query)(params, batch)
  jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(a, b), eager, jitted)


@pytest.mark.parametrize("name,value", [
    ("clip_norm", 0.0),
    ("clip_norm", -1.0),
    ("clip_norm", jnp.nan),
    ("clip_norm", jnp.inf),
    ("normalize_by", 0.0),
    ("normalize_by", -1.0),
    ("normalize_by", jnp.nan),
    ("normalize_by", jnp.inf),
])
def test_invalid_public_clipping_configuration(name, value):
  kwargs = {"clip_norm": 1.0, "normalize_by": 1.0}
  kwargs[name] = value
  with pytest.raises(ValueError, match="finite positive"):
    make_clipped_gradient_query(_linear_loss, **kwargs)
