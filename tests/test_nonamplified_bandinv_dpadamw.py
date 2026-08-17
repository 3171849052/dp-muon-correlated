"""Tests for the naive full-tree BandInvMF DP-AdamW composition.

Verifies:
1. Exactly one clipped query and one sample_bandinv_noise per logical batch
2. Correlated noise PyTree matches params/gradient PyTree
3. Optimizer receives private_grad (clipped_grad + noise)
4. Zero-noise matches independent optax.adamw reference
5. Eager == jax.jit
6. Non-zero correlated noise checkpoint resume
7. Prefix-sum artifact does not collide with nesterov-trajectory
8. Same prefix-sum config can cache hit
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_privacy.matrix_factorization import toeplitz

from dp_muon.bandinvmf import BandInvMFStrategy, sample_bandinv_noise
from dp_muon.privacy import ParticipationSpec, calibrate_nonamplified_bandinv
from dp_muon.training import (
    init_nonamplified_bandinv_dpadamw_state,
    make_nonamplified_bandinv_dpadamw_train_step,
)
from dp_muon.training.checkpoint import load_checkpoint, save_checkpoint
import dp_muon.training.nonamplified_bandinv_dpadamw as bandinv_dpadamw


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
  step, optimizer = make_nonamplified_bandinv_dpadamw_train_step(
      _loss, strategy, calibration, participation,
      learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
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


def _adamw_update_count(state):
  """AdamW has a single count scalar (the step counter) in its state."""
  for leaf in jax.tree_util.tree_leaves(state.optimizer_state):
    leaf = jnp.asarray(leaf)
    if leaf.shape == () and jnp.issubdtype(leaf.dtype, jnp.integer):
      return int(leaf)
  return 0


# --- Test 1: exactly one query and one sampler per logical batch ---

def test_one_query_one_sampler_and_full_tree_template(monkeypatch):
  calls, query_calls = [], []
  real_query_factory = bandinv_dpadamw.make_clipped_gradient_query
  real_sampler = bandinv_dpadamw.sample_bandinv_noise

  def query_factory(*args, **kwargs):
    query = real_query_factory(*args, **kwargs)
    def counted(*query_args, **query_kwargs):
      query_calls.append(None)
      return query(*query_args, **query_kwargs)
    return counted

  def counted_sampler(*args):
    calls.append(args[1])
    return real_sampler(*args)

  monkeypatch.setattr(bandinv_dpadamw, "make_clipped_gradient_query", query_factory)
  monkeypatch.setattr(bandinv_dpadamw, "sample_bandinv_noise", counted_sampler)
  step, optimizer, strategy, _ = _make_step()
  initial = init_nonamplified_bandinv_dpadamw_state(
      _params(), strategy, jax.random.key(1), optimizer
  )
  result = step(initial, {"x": jnp.array([1.0], jnp.float32)})
  assert len(query_calls) == len(calls) == 1
  # Correlated noise tree structure matches params
  assert jax.tree_util.tree_structure(calls[0].buffer) == jax.tree_util.tree_structure(initial.params)
  assert jax.tree_util.tree_structure(result.params) == jax.tree_util.tree_structure(initial.params)


# --- Test 2: noise is added before the optimizer (private_grad = clipped_grad + noise) ---

def test_noise_is_added_before_optimizer(monkeypatch):
  step, optimizer, strategy, _ = _make_step()
  initial = init_nonamplified_bandinv_dpadamw_state(
      _params(), strategy, jax.random.key(2), optimizer
  )

  def fixed_noise(key, state, coef, std):
    zero_noise, new_state, new_key = sample_bandinv_noise(key, state, coef, 0.0)
    return jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * 0.25, zero_noise), new_state, new_key

  monkeypatch.setattr(bandinv_dpadamw, "sample_bandinv_noise", fixed_noise)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  actual = step(initial, batch)
  # Independent reference: compute private_grad = clipped_grad + 0.25, feed to AdamW
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


# --- Test 3: zero-noise matches clean AdamW update ---

def test_zero_noise_matches_clean_adamw(monkeypatch):
  step, optimizer, strategy, _ = _make_step()
  initial = init_nonamplified_bandinv_dpadamw_state(
      _params(), strategy, jax.random.key(3), optimizer
  )
  real_sampler = bandinv_dpadamw.sample_bandinv_noise

  def zero_noise(key, state, coef, std):
    return real_sampler(key, state, coef, 0.0)

  monkeypatch.setattr(bandinv_dpadamw, "sample_bandinv_noise", zero_noise)
  batch = {"x": jnp.array([1.0], jnp.float32)}
  actual = step(initial, batch)
  # Independent reference AdamW (not the one returned by the train step)
  ref_optimizer = optax.adamw(learning_rate=0.01, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
  ref_opt_state = ref_optimizer.init(initial.params)
  clean_grad = jax.grad(_loss)(initial.params, batch)
  updates, expected_opt_state = ref_optimizer.update(
      clean_grad, ref_opt_state, initial.params
  )
  expected_params = jax.tree_util.tree_map(
      lambda parameter, update: parameter + update, initial.params, updates
  )
  _tree_allclose(actual.params, expected_params)
  _tree_allclose(actual.optimizer_state, expected_opt_state)


# --- Test 4: bandwidth=1 streaming, JIT, step alignment ---

def test_bandwidth_one_streaming_jit_and_steps_are_aligned():
  step, optimizer, strategy, calibration = _make_step(horizon=3, noising_coef=(1.0,))
  initial = init_nonamplified_bandinv_dpadamw_state(
      _params(), strategy, jax.random.key(4), optimizer
  )
  batch = {"x": jnp.array([0.5], jnp.float32)}
  eager = step(initial, batch)
  compiled = jax.jit(step)(initial, batch)
  _tree_allclose(compiled, eager)
  # Verify the noise state is updated correctly
  direct_noise, direct_noise_state, direct_key = sample_bandinv_noise(
      initial.rng_key, initial.noise_state, strategy.noising_coef,
      calibration.iid_noise_std,
  )
  clean_grad = jax.grad(_loss)(initial.params, batch)
  private_grad = jax.tree_util.tree_map(
      lambda gradient, noise: gradient + noise, clean_grad, direct_noise
  )
  ref_optimizer = optax.adamw(learning_rate=0.01, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
  ref_opt_state = ref_optimizer.init(initial.params)
  updates, _ = ref_optimizer.update(private_grad, ref_opt_state, initial.params)
  expected_params = jax.tree_util.tree_map(
      lambda parameter, update: parameter + update, initial.params, updates
  )
  _tree_allclose(eager.params, expected_params)
  _tree_allclose(eager.noise_state, direct_noise_state)
  _tree_allclose(eager.rng_key, direct_key)
  assert eager.noise_state.bandwidth == 1
  assert int(eager.step) == int(eager.noise_state.step) == 1
  assert _adamw_update_count(eager) == 1


# --- Test 5: checkpoint resume matches uninterrupted ---

def test_checkpoint_resume_matches_uninterrupted(tmp_path):
  step, optimizer, strategy, _ = _make_step(horizon=4)
  batches = [{"x": jnp.array([value], jnp.float32)} for value in (1.0, -0.5, 0.25, 2.0)]
  initial = init_nonamplified_bandinv_dpadamw_state(
      _params(), strategy, jax.random.key(5), optimizer
  )
  uninterrupted = initial
  for batch in batches:
    uninterrupted = step(uninterrupted, batch)
  # Train 2 steps, save checkpoint, resume for 2 more
  resumed = step(step(initial, batches[0]), batches[1])
  checkpoint = tmp_path / "bandinv-dpadamw.pkl"
  save_checkpoint(
      checkpoint, state=resumed, current_step=2,
      experiment_config={"test": True},
      artifact_identifiers={"algorithm": "dp-adamw-correlated-naive"},
  )
  resumed = load_checkpoint(checkpoint)["state"]
  for batch in batches[2:]:
    resumed = step(resumed, batch)
  _tree_allclose(resumed, uninterrupted)
  assert int(resumed.step) == int(resumed.noise_state.step) == len(batches)
  assert _adamw_update_count(resumed) == len(batches)


# --- Test 6: prefix-sum artifact does not collide with nesterov-trajectory ---

def test_prefix_sum_artifact_path_differs_from_nesterov():
  from dp_muon.training.bandinvmf_strategy_manager import (
      prefix_sum_strategy_artifact_path,
      strategy_artifact_path,
  )
  ps_path = prefix_sum_strategy_artifact_path(
      "strategies", horizon=488, min_sep=97, max_participations=5,
      bandwidth=4, reduction="mean", max_optimizer_steps=1000,
  )
  n_path = strategy_artifact_path(
      "strategies", horizon=488, min_sep=97, max_participations=5,
      bandwidth=4, momentum=0.95, learning_rate=0.001,
      reduction="mean", max_optimizer_steps=1000,
  )
  assert ps_path != n_path
  assert "prefix-sum" in ps_path.name
  assert "nesterov-trajectory" in n_path.name


# --- Test 7: same prefix-sum config can cache hit ---

def test_prefix_sum_same_config_cache_hit(tmp_path, monkeypatch):
  from dp_muon.training.bandinvmf_strategy_manager import (
      PrefixSumBandInvMFFitRequest,
      get_or_fit_prefix_sum_strategy_snapshot,
      fit_bandinv_strategy,
  )
  request = PrefixSumBandInvMFFitRequest(
      horizon=10, min_sep=2, max_participations=3, bandwidth=2,
      reduction="mean", max_optimizer_steps=100,
      strategy_dir=str(tmp_path / "strategies"), force_refit=False,
  )
  # First call: fit
  snapshot1, action1 = get_or_fit_prefix_sum_strategy_snapshot(
      request, fit_strategy=fit_bandinv_strategy,
  )
  assert action1 == "fit"
  # Second call: reuse
  snapshot2, action2 = get_or_fit_prefix_sum_strategy_snapshot(
      request, fit_strategy=fit_bandinv_strategy,
  )
  assert action2 == "reuse"
  assert snapshot1.path == snapshot2.path
  assert snapshot1.sha256 == snapshot2.sha256


# --- Test 8: prefix-sum artifact does not match nesterov metadata ---

def test_prefix_sum_metadata_rejects_nesterov_artifact(tmp_path, monkeypatch):
  from dp_muon.bandinvmf import save_bandinv_strategy
  from dp_muon.training.bandinvmf_strategy_manager import (
      PrefixSumBandInvMFFitRequest,
      _load_compatible_prefix_sum_snapshot_unlocked,
      prefix_sum_strategy_artifact_path,
  )
  # Save a nesterov-trajectory artifact at the prefix-sum path
  strategy, _, _ = _artifacts(horizon=10, noising_coef=(1.0, -0.5))
  ps_path = prefix_sum_strategy_artifact_path(
      "strategies", horizon=10, min_sep=2, max_participations=3,
      bandwidth=2, reduction="mean", max_optimizer_steps=100,
  )
  save_bandinv_strategy(
      ps_path, strategy,
      reduction="mean", workload_type="nesterov-trajectory",
      momentum=0.95, learning_rate=0.01, max_optimizer_steps=100,
  )
  request = PrefixSumBandInvMFFitRequest(
      horizon=10, min_sep=2, max_participations=3, bandwidth=2,
      reduction="mean", max_optimizer_steps=100,
      strategy_dir="strategies", force_refit=False,
  )
  # The prefix-sum loader should not accept a nesterov-trajectory artifact
  from dp_muon.training.bandinvmf_strategy_manager import REPOSITORY_ROOT
  resolved_path = REPOSITORY_ROOT / ps_path
  snapshot = _load_compatible_prefix_sum_snapshot_unlocked(resolved_path, request)
  assert snapshot is None, "prefix-sum loader must reject nesterov-trajectory metadata"


# --- Test 9: optimizer state counts are correct for AdamW ---

def test_adamw_optimizer_state_has_one_count():
  step, optimizer, strategy, _ = _make_step(horizon=3, noising_coef=(1.0,))
  initial = init_nonamplified_bandinv_dpadamw_state(
      _params(), strategy, jax.random.key(6), optimizer
  )
  # AdamW state has a single count scalar
  count = _adamw_update_count(initial)
  assert count == 0
  result = step(initial, {"x": jnp.array([1.0], jnp.float32)})
  assert _adamw_update_count(result) == 1


# --- Test 10: state validation rejects non-matching state type ---

def test_state_validation_rejects_wrong_type():
  from dp_muon.training.nonamplified_bandinv_dpadamw import NonAmplifiedBandInvDPAdamWState, _validate_state
  from dp_muon.bandinvmf import BandInvMFNoiseState
  state = NonAmplifiedBandInvDPAdamWState(
      params=_params(),
      optimizer_state={},
      noise_state=BandInvMFNoiseState(
          buffer=_params(), cursor=jnp.array(0), step=jnp.array(0), bandwidth=1,
      ),
      rng_key=jax.random.key(0),
      step=jnp.array(0, dtype=jnp.int32),
  )
  _validate_state(state)  # should not raise

  # Invalid noise_state type
  import dataclasses
  bad_state = dataclasses.replace(state, noise_state=object())
  import pytest
  with pytest.raises(TypeError, match="must be a BandInvMFNoiseState"):
    _validate_state(bad_state)