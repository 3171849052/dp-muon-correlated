from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from dp_muon.privacy import calibrate_nonamplified_iid
from dp_muon.training import (
    init_nonamplified_dpmuon_state,
    make_nonamplified_dpmuon_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
import dp_muon.training.nonamplified_dpmuon as dpmuon


def _params():
  dense = lambda: {"kernel": jnp.ones((2, 2), jnp.float32), "bias": jnp.zeros((2,), jnp.float32)}
  return {
      "blocks": ({"attention": {name: dense() for name in ("query", "key", "value", "out")}, "mlp": {name: dense() for name in ("dense0", "dense1")}},),
      "patch_embedding": dense(),
      "head": dense(),
  }


def _loss(params, batch):
  return (
      jnp.sum(params["blocks"][0]["attention"]["query"]["kernel"])
      + jnp.sum(params["head"]["kernel"])
  ) * batch["x"][0]


def _calibration(noise_std):
  calibration = calibrate_nonamplified_iid(
      epsilon=2.0, delta=1e-5, clip_norm=100.0, normalize_by=1.0,
      adjacency="add_remove", max_participations=1,
  )
  return replace(calibration, iid_noise_std=noise_std)


def _make_step(noise_std=0.0):
  return make_nonamplified_dpmuon_train_step(
      _loss, _calibration(noise_std), muon_learning_rate=0.01,
      muon_weight_decay=0.0, momentum=0.95, ns_steps=5, consistent_rms=0.2,
      adamw_learning_rate=0.001, adamw_beta1=0.9, adamw_beta2=0.999,
      adamw_eps=1e-8, adamw_weight_decay=0.0, use_bf16_ns=False,
  )


def test_one_full_tree_noise_sample_and_reproducible_jit(monkeypatch):
  calls = []
  query_calls = []
  real_query_factory = dpmuon.make_clipped_gradient_query
  def query_factory(*args, **kwargs):
    query = real_query_factory(*args, **kwargs)
    def counted_query(*query_args, **query_kwargs):
      query_calls.append(None)
      return query(*query_args, **query_kwargs)
    return counted_query
  def noise(key, template, noise_std):
    calls.append((template, noise_std))
    return jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.25, template), key
  monkeypatch.setattr(dpmuon, "make_clipped_gradient_query", query_factory)
  monkeypatch.setattr(dpmuon, "_sample_iid_gaussian_noise", noise)
  step, optimizer = _make_step(0.25)
  initial = init_nonamplified_dpmuon_state(_params(), jax.random.key(3), optimizer)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  eager = step(initial, batch)
  repeated = step(initial, batch)
  assert len(calls) == 2
  assert len(query_calls) == 2
  assert jax.tree_util.tree_structure(calls[0][0]) == jax.tree_util.tree_structure(initial.params)
  for actual, expected in zip(jax.tree_util.tree_leaves(repeated.params), jax.tree_util.tree_leaves(eager.params), strict=True):
    np.testing.assert_allclose(actual, expected)
  # Restore the JIT-compatible sampler (the list-appending fake is eager-only).
  monkeypatch.setattr(dpmuon, "_sample_iid_gaussian_noise", dpmuon.sample_iid_gaussian_noise)
  eager = step(initial, batch)
  compiled = jax.jit(step)(initial, batch)
  for actual, expected in zip(jax.tree_util.tree_leaves(compiled.params), jax.tree_util.tree_leaves(eager.params), strict=True):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_zero_noise_matches_clean_partitioned_muon_adamw_update():
  step, optimizer = _make_step(0.0)
  initial = init_nonamplified_dpmuon_state(_params(), jax.random.key(4), optimizer)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  actual = step(initial, batch)
  clean_grad = jax.grad(_loss)(initial.params, batch)
  updates, expected_state = optimizer.update(clean_grad, initial.optimizer_state, initial.params)
  expected_params = jax.tree_util.tree_map(lambda p, u: p + u, initial.params, updates)
  for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual.params), jax.tree_util.tree_leaves(expected_params), strict=True):
    np.testing.assert_allclose(actual_leaf, expected_leaf)
  assert jax.tree_util.tree_structure(actual.optimizer_state) == jax.tree_util.tree_structure(expected_state)


def test_checkpointed_partitioned_optimizer_resumes_identically(tmp_path):
  step, optimizer = _make_step(0.0)
  initial = init_nonamplified_dpmuon_state(_params(), jax.random.key(8), optimizer)
  batches = [{"x": jnp.array([value], jnp.float32)} for value in (1.0, -0.5, 2.0)]
  uninterrupted = initial
  for batch in batches:
    uninterrupted = step(uninterrupted, batch)
  resumed = step(initial, batches[0])
  checkpoint = tmp_path / "dpmuon.pkl"
  save_checkpoint(checkpoint, state=resumed, current_step=1, experiment_config={"test": True}, artifact_identifiers={"algorithm": "dpmuon"})
  resumed = load_checkpoint(checkpoint)["state"]
  for batch in batches[1:]:
    resumed = step(resumed, batch)
  for actual, expected in zip(jax.tree_util.tree_leaves(resumed.params), jax.tree_util.tree_leaves(uninterrupted.params), strict=True):
    np.testing.assert_allclose(actual, expected)
