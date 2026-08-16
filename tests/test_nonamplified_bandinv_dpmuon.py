"""Tests for the naive full-tree BandInvMF DP-Muon composition."""

import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy, sample_bandinv_noise
from dp_muon.optim import ADAMW, MUON, vit_muon_parameter_labels
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training import (
    init_nonamplified_bandinv_dpmuon_state,
    make_nonamplified_bandinv_dpmuon_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
import dp_muon.training.nonamplified_bandinv_dpmuon as bandinv_dpmuon


def _params():
  dense = lambda: {
      "kernel": jnp.ones((2, 2), jnp.float32),
      "bias": jnp.zeros((2,), jnp.float32),
  }
  return {
      "blocks": ({
          "attention": {name: dense() for name in ("query", "key", "value", "out")},
          "mlp": {name: dense() for name in ("dense0", "dense1")},
      },),
      "head": dense(),
  }


def _loss(params, batch):
  return (
      jnp.sum(params["blocks"][0]["attention"]["query"]["kernel"])
      + jnp.sum(params["head"]["kernel"])
  ) * batch["x"][0]


def _artifacts(*, horizon=4, noising_coef=(1.0, -0.25)):
  coef = jnp.asarray(noising_coef, jnp.float32)
  sensitivity = toeplitz.compute_banded_inverse_sensitivity_squared(
      n=horizon, noising_coef=coef, min_sep=1, max_participations=None
  )
  # This intentionally is not the linear fixed-LR Nesterov workload: the
  # naive Muon trainer must validate privacy artefacts without that constraint.
  strategy = BandInvMFStrategy(
      horizon=horizon, bandwidth=len(noising_coef), min_sep=1,
      max_participations=None, workload_coef=jnp.ones((horizon,), jnp.float32),
      noising_coef=coef, strategy_coef=jnp.ones((horizon,), jnp.float32),
      sensitivity_squared=sensitivity, objective=jnp.array(0.0, jnp.float32),
  )
  calibration = calibrate_nonamplified_bandinv(
      epsilon=2.0, delta=1e-5, clip_norm=100.0, normalize_by=1.0,
      adjacency="add_remove", sensitivity_squared=float(sensitivity),
  )
  return strategy, calibration, ParticipationSpec(horizon, 1, None)


def _make_step(*, horizon=4, noising_coef=(1.0, -0.25)):
  strategy, calibration, participation = _artifacts(
      horizon=horizon, noising_coef=noising_coef
  )
  step, optimizer = make_nonamplified_bandinv_dpmuon_train_step(
      _loss, strategy, calibration, participation,
      muon_learning_rate=0.01, muon_weight_decay=0.0, momentum=0.95,
      ns_steps=5, consistent_rms=0.2, adamw_learning_rate=0.001,
      adamw_beta1=0.9, adamw_beta2=0.999, adamw_eps=1e-8,
      adamw_weight_decay=0.0, use_bf16_ns=False,
  )
  return step, optimizer, strategy, calibration


def _tree_allclose(actual, expected):
  assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
  for actual_leaf, expected_leaf in zip(
      jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
  ):
    if jnp.issubdtype(jnp.asarray(actual_leaf).dtype, jax.dtypes.prng_key):
      np.testing.assert_array_equal(
          jax.random.key_data(actual_leaf), jax.random.key_data(expected_leaf)
      )
    else:
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-7)


def _partition_counts(state):
  """The reused fixed partition has one Muon and one AdamW update counter."""
  counts = []
  for leaf in jax.tree_util.tree_leaves(state.optimizer_state):
    leaf = jnp.asarray(leaf)
    if leaf.shape == () and jnp.issubdtype(leaf.dtype, jnp.integer):
      counts.append(leaf)
  assert len(counts) == 2
  return tuple(counts)


def test_one_query_one_sampler_and_full_tree_template(monkeypatch):
  calls, query_calls = [], []
  real_query_factory = bandinv_dpmuon.make_clipped_gradient_query
  real_sampler = bandinv_dpmuon.sample_bandinv_noise

  def query_factory(*args, **kwargs):
    query = real_query_factory(*args, **kwargs)
    def counted(*query_args, **query_kwargs):
      query_calls.append(None)
      return query(*query_args, **query_kwargs)
    return counted

  def counted_sampler(*args):
    calls.append(args[1])
    return real_sampler(*args)

  monkeypatch.setattr(bandinv_dpmuon, "make_clipped_gradient_query", query_factory)
  monkeypatch.setattr(bandinv_dpmuon, "sample_bandinv_noise", counted_sampler)
  step, optimizer, strategy, _ = _make_step()
  initial = init_nonamplified_bandinv_dpmuon_state(
      _params(), strategy, jax.random.key(1), optimizer
  )
  result = step(initial, {"x": jnp.array([1.0], jnp.float32)})
  assert len(query_calls) == len(calls) == 1
  assert jax.tree_util.tree_structure(calls[0].buffer) == jax.tree_util.tree_structure(initial.params)
  assert jax.tree_util.tree_structure(result.params) == jax.tree_util.tree_structure(initial.params)
  labels = vit_muon_parameter_labels(initial.params)
  assert MUON in jax.tree_util.tree_leaves(labels)
  assert ADAMW in jax.tree_util.tree_leaves(labels)


def test_noise_is_added_before_the_standard_partitioned_optimizer(monkeypatch):
  step, optimizer, strategy, _ = _make_step()
  initial = init_nonamplified_bandinv_dpmuon_state(
      _params(), strategy, jax.random.key(2), optimizer
  )

  def fixed_noise(key, state, coef, std):
    zero_noise, new_state, new_key = sample_bandinv_noise(key, state, coef, 0.0)
    return jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * 0.25, zero_noise), new_state, new_key

  monkeypatch.setattr(bandinv_dpmuon, "sample_bandinv_noise", fixed_noise)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  actual = step(initial, batch)
  clipped_grad = jax.grad(_loss)(initial.params, batch)
  private_grad = jax.tree_util.tree_map(lambda leaf: leaf + 0.25, clipped_grad)
  updates, expected_optimizer_state = optimizer.update(
      private_grad, initial.optimizer_state, initial.params
  )
  expected_params = jax.tree_util.tree_map(
      lambda parameter, update: parameter + update, initial.params, updates
  )
  _tree_allclose(actual.params, expected_params)
  _tree_allclose(actual.optimizer_state, expected_optimizer_state)


def test_zero_noise_matches_clean_partitioned_muon_adamw_update(monkeypatch):
  step, optimizer, strategy, _ = _make_step()
  initial = init_nonamplified_bandinv_dpmuon_state(
      _params(), strategy, jax.random.key(3), optimizer
  )
  real_sampler = bandinv_dpmuon.sample_bandinv_noise

  def zero_noise(key, state, coef, std):
    return real_sampler(key, state, coef, 0.0)

  monkeypatch.setattr(bandinv_dpmuon, "sample_bandinv_noise", zero_noise)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  actual = step(initial, batch)
  clean_grad = jax.grad(_loss)(initial.params, batch)
  updates, expected_optimizer_state = optimizer.update(
      clean_grad, initial.optimizer_state, initial.params
  )
  expected_params = jax.tree_util.tree_map(
      lambda parameter, update: parameter + update, initial.params, updates
  )
  _tree_allclose(actual.params, expected_params)
  _tree_allclose(actual.optimizer_state, expected_optimizer_state)


def test_bandwidth_one_streaming_jit_and_steps_are_aligned():
  step, optimizer, strategy, calibration = _make_step(horizon=3, noising_coef=(1.0,))
  initial = init_nonamplified_bandinv_dpmuon_state(
      _params(), strategy, jax.random.key(4), optimizer
  )
  batch = {"x": jnp.array([0.5], jnp.float32)}
  eager = step(initial, batch)
  compiled = jax.jit(step)(initial, batch)
  _tree_allclose(compiled, eager)
  direct_noise, direct_noise_state, direct_key = sample_bandinv_noise(
      initial.rng_key, initial.noise_state, strategy.noising_coef,
      calibration.iid_noise_std,
  )
  clean_grad = jax.grad(_loss)(initial.params, batch)
  private_grad = jax.tree_util.tree_map(
      lambda gradient, noise: gradient + noise, clean_grad, direct_noise
  )
  updates, _ = optimizer.update(private_grad, initial.optimizer_state, initial.params)
  expected_params = jax.tree_util.tree_map(
      lambda parameter, update: parameter + update, initial.params, updates
  )
  _tree_allclose(eager.params, expected_params)
  _tree_allclose(eager.noise_state, direct_noise_state)
  _tree_allclose(eager.rng_key, direct_key)
  assert eager.noise_state.bandwidth == 1
  assert int(eager.step) == int(eager.noise_state.step) == 1
  assert tuple(map(int, _partition_counts(eager))) == (1, 1)


def test_checkpoint_resume_matches_uninterrupted_and_keeps_steps_in_sync(tmp_path):
  step, optimizer, strategy, _ = _make_step(horizon=4)
  batches = [{"x": jnp.array([value], jnp.float32)} for value in (1.0, -0.5, 0.25, 2.0)]
  initial = init_nonamplified_bandinv_dpmuon_state(
      _params(), strategy, jax.random.key(5), optimizer
  )
  uninterrupted = initial
  for batch in batches:
    uninterrupted = step(uninterrupted, batch)
  resumed = step(step(initial, batches[0]), batches[1])
  checkpoint = tmp_path / "bandinv-dpmuon.pkl"
  save_checkpoint(
      checkpoint, state=resumed, current_step=2,
      experiment_config={"test": True},
      artifact_identifiers={"algorithm": "dp-muon-correlated-naive"},
  )
  resumed = load_checkpoint(checkpoint)["state"]
  for batch in batches[2:]:
    resumed = step(resumed, batch)
  _tree_allclose(resumed, uninterrupted)
  assert int(resumed.step) == int(resumed.noise_state.step) == len(batches)
  assert tuple(map(int, _partition_counts(resumed))) == (len(batches),) * 2
