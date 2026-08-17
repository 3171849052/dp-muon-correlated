from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dp_muon.privacy import calibrate_nonamplified_iid
from dp_muon.training import (
    init_nonamplified_dpadamw_state,
    make_nonamplified_dpadamw_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
import dp_muon.training.nonamplified_dpadamw as dpadamw


def _params():
  return {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]], jnp.float32), "b": jnp.array([0.0, 0.0], jnp.float32)}


def _loss(params, batch):
  return jnp.sum(params["w"] * batch["x"][0]) + jnp.sum(params["b"])


def _calibration(noise_std=0.0):
  calibration = calibrate_nonamplified_iid(
      epsilon=2.0, delta=1e-5, clip_norm=100.0, normalize_by=1.0,
      adjacency="add_remove", max_participations=1,
  )
  return replace(calibration, iid_noise_std=noise_std)


def _make_step(noise_std=0.0):
  return make_nonamplified_dpadamw_train_step(
      _loss, _calibration(noise_std),
      learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
  )


def test_one_full_tree_noise_sample_and_reproducible_jit(monkeypatch):
  calls = []
  query_calls = []
  real_query_factory = dpadamw.make_clipped_gradient_query

  def query_factory(*args, **kwargs):
    query = real_query_factory(*args, **kwargs)
    def counted_query(*query_args, **query_kwargs):
      query_calls.append(None)
      return query(*query_args, **query_kwargs)
    return counted_query

  def noise(key, template, noise_std):
    calls.append((template, noise_std))
    return jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.25, template), key

  monkeypatch.setattr(dpadamw, "make_clipped_gradient_query", query_factory)
  monkeypatch.setattr(dpadamw, "_sample_iid_gaussian_noise", noise)
  step, optimizer = _make_step(0.25)
  initial = init_nonamplified_dpadamw_state(_params(), jax.random.key(3), optimizer)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  eager = step(initial, batch)
  repeated = step(initial, batch)
  # Exactly one clipped query call and one noise call per logical batch.
  assert len(calls) == 2
  assert len(query_calls) == 2
  # Noise tree matches the parameter PyTree structure.
  assert jax.tree_util.tree_structure(calls[0][0]) == jax.tree_util.tree_structure(initial.params)
  # AdamW only sees the private_grad by construction: the monkeypatched noise
  # adds a known constant to every leaf, and the optimizer receives the sum.
  # Repeated invocations with the same seed produce identical results.
  for actual, expected in zip(jax.tree_util.tree_leaves(repeated.params), jax.tree_util.tree_leaves(eager.params), strict=True):
    np.testing.assert_allclose(actual, expected)
  # Restore the JIT-compatible sampler and verify eager == JIT.
  monkeypatch.setattr(dpadamw, "_sample_iid_gaussian_noise", dpadamw.sample_iid_gaussian_noise)
  eager = step(initial, batch)
  compiled = jax.jit(step)(initial, batch)
  for actual, expected in zip(jax.tree_util.tree_leaves(compiled.params), jax.tree_util.tree_leaves(eager.params), strict=True):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_zero_noise_matches_clean_adamw_update():
  """DP-AdamW with noise_std=0 must produce the same result as plain optax.adamw."""
  step, step_optimizer = _make_step(0.0)
  initial = init_nonamplified_dpadamw_state(_params(), jax.random.key(4), step_optimizer)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  actual = step(initial, batch)
  # Independent reference: create a clean optax.adamw that is NOT the one
  # returned by make_nonamplified_dpadamw_train_step.
  reference_optimizer = optax.adamw(
      learning_rate=0.1, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0,
  )
  clean_grad = jax.grad(_loss)(initial.params, batch)
  ref_updates, ref_state = reference_optimizer.update(
      clean_grad, reference_optimizer.init(initial.params), initial.params
  )
  expected_params = jax.tree_util.tree_map(lambda p, u: p + u, initial.params, ref_updates)
  # Compare params leaf-by-leaf.
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(actual.params),
      jax.tree_util.tree_leaves(expected_params),
      strict=True,
  ):
    np.testing.assert_allclose(actual_leaf, expected_leaf)
  # Compare optimizer state numerically (not just structure).
  # Both states are optax AdamW states with the same hyper-parameters, so every
  # leaf (mu, nu, step) should be identical after one step on the same gradient.
  actual_leaves = list(jax.tree_util.tree_leaves(actual.optimizer_state))
  expected_leaves = list(jax.tree_util.tree_leaves(ref_state))
  assert len(actual_leaves) == len(expected_leaves)
  for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
    np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf))


def test_checkpointed_optimizer_resumes_identically(tmp_path):
  """Checkpoint resume must preserve AdamW moments, DP RNG, and step counter."""
  noise_std = 0.2
  step, optimizer = _make_step(noise_std)
  initial = init_nonamplified_dpadamw_state(_params(), jax.random.key(8), optimizer)
  batches = [{"x": jnp.array([value], jnp.float32)} for value in (1.0, -0.5, 2.0)]
  # A: uninterrupted training over all batches.
  uninterrupted = initial
  for batch in batches:
    uninterrupted = step(uninterrupted, batch)
  # B: run first batch, save checkpoint, load, run remaining.
  resumed = step(initial, batches[0])
  checkpoint = tmp_path / "dpadamw.pkl"
  save_checkpoint(
      checkpoint, state=resumed, current_step=1,
      experiment_config={"test": True}, artifact_identifiers={"algorithm": "dpadamw"},
  )
  loaded = load_checkpoint(checkpoint)["state"]
  for batch in batches[1:]:
    loaded = step(loaded, batch)
  # Verify all state fields are identical.
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(loaded.params),
      jax.tree_util.tree_leaves(uninterrupted.params),
      strict=True,
  ):
    np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf))
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(loaded.optimizer_state),
      jax.tree_util.tree_leaves(uninterrupted.optimizer_state),
      strict=True,
  ):
    np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf))
  np.testing.assert_array_equal(
      jax.random.key_data(loaded.rng_key), jax.random.key_data(uninterrupted.rng_key),
  )
  np.testing.assert_array_equal(
      np.asarray(loaded.step), np.asarray(uninterrupted.step),
  )


def test_learning_rate_parameter_validation():
  for invalid in (0.0, -1.0, float("nan")):
    with pytest.raises((ValueError, TypeError)):
      make_nonamplified_dpadamw_train_step(_loss, _calibration(), learning_rate=invalid)


def test_beta1_parameter_validation():
  for invalid in (1.0, -0.1, float("nan")):
    with pytest.raises((ValueError, TypeError)):
      make_nonamplified_dpadamw_train_step(_loss, _calibration(), learning_rate=0.1, beta1=invalid)


def test_beta2_parameter_validation():
  for invalid in (1.0, -0.1, float("nan")):
    with pytest.raises((ValueError, TypeError)):
      make_nonamplified_dpadamw_train_step(_loss, _calibration(), learning_rate=0.1, beta2=invalid)


def test_eps_parameter_validation():
  for invalid in (0.0, -1.0, float("nan")):
    with pytest.raises((ValueError, TypeError)):
      make_nonamplified_dpadamw_train_step(_loss, _calibration(), learning_rate=0.1, eps=invalid)


def test_weight_decay_parameter_validation():
  for invalid in (-0.1, float("nan")):
    with pytest.raises((ValueError, TypeError)):
      make_nonamplified_dpadamw_train_step(_loss, _calibration(), learning_rate=0.1, weight_decay=invalid)